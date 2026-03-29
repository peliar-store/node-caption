'use strict';
/**
 * audio-capture.js
 * Captures system audio via PulseAudio monitor source using ffi-napi bindings
 * to FFmpeg shared libraries. Emits Float32Array chunks at 16kHz mono.
 */

const ffi = require('ffi-napi');
const ref = require('ref-napi');
const { EventEmitter } = require('events');

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const AVMEDIA_TYPE_AUDIO = 1;
const AV_SAMPLE_FMT_S16 = 1;
const AV_SAMPLE_FMT_FLT = 3;
// Legacy channel layout constants (still valid in FFmpeg 6.1)
const AV_CH_LAYOUT_STEREO = BigInt(3);
const AV_CH_LAYOUT_MONO   = BigInt(4);

// Verified struct offsets (compiled against FFmpeg 6.1 headers)
const OFFSETS = {
  fmtctx:  { nb_streams: 44, streams: 48 },
  stream:  { codecpar: 16 },
  codecpar:{ codec_id: 4, codec_type: 0, sample_rate: 116, channels: 112, format: 28 },
  packet:  { data: 24, size: 32, stream_index: 36 },
  frame:   { extended_data: 96, linesize0: 64, nb_samples: 112, format: 116 },
};

// ---------------------------------------------------------------------------
// FFmpeg library bindings
// ---------------------------------------------------------------------------
const voidPtr = ref.refType(ref.types.void);
const voidPtrPtr = ref.refType(voidPtr);

const libavdevice = ffi.Library('libavdevice.so.60', {
  avdevice_register_all: ['void', []],
});

const libavformat = ffi.Library('libavformat.so.60', {
  av_find_input_format:        [voidPtr, ['string']],
  avformat_alloc_context:      [voidPtr, []],
  avformat_open_input:         ['int', [voidPtrPtr, 'string', voidPtr, voidPtrPtr]],
  avformat_find_stream_info:   ['int', [voidPtr, voidPtrPtr]],
  av_read_frame:               ['int', [voidPtr, voidPtr]],
  avformat_close_input:        ['void', [voidPtrPtr]],
  av_find_best_stream:         ['int', [voidPtr, 'int', 'int', 'int', voidPtrPtr, 'int']],
});

const libavcodec = ffi.Library('libavcodec.so.60', {
  avcodec_find_decoder:          [voidPtr, ['uint']],
  avcodec_alloc_context3:        [voidPtr, [voidPtr]],
  avcodec_parameters_to_context: ['int', [voidPtr, voidPtr]],
  avcodec_open2:                 ['int', [voidPtr, voidPtr, voidPtrPtr]],
  avcodec_send_packet:           ['int', [voidPtr, voidPtr]],
  avcodec_receive_frame:         ['int', [voidPtr, voidPtr]],
  av_packet_alloc:               [voidPtr, []],
  av_packet_unref:               ['void', [voidPtr]],
  av_frame_alloc:                [voidPtr, []],
  av_frame_free:                 ['void', [voidPtrPtr]],
  avcodec_free_context:          ['void', [voidPtrPtr]],
});

const libavutil = ffi.Library('libavutil.so.58', {
  av_dict_set:  ['int', [voidPtrPtr, 'string', 'string', 'int']],
  av_dict_free: ['void', [voidPtrPtr]],
});

const libswresample = ffi.Library('libswresample.so.4', {
  swr_alloc:             [voidPtr, []],
  swr_alloc_set_opts:    [voidPtr, [voidPtr, 'int64', 'int', 'int', 'int64', 'int', 'int', 'int', 'int64']],
  swr_init:              ['int', [voidPtr]],
  swr_convert:           ['int', [voidPtr, voidPtrPtr, 'int', voidPtrPtr, 'int']],
  swr_free:              ['void', [voidPtrPtr]],
});

// ---------------------------------------------------------------------------
// Helper: read a pointer value from a buffer at given byte offset
// Returns a Buffer representing the pointed-to memory (size=1 placeholder)
// ---------------------------------------------------------------------------
function readPtrAt(buf, offset) {
  const addr = buf.readBigUInt64LE(offset);
  if (addr === 0n) return null;
  return addr;
}

function addrToBuffer(addr, size) {
  // Create a buffer from a raw address (BigInt)
  // ref-napi: ref.reinterpret needs a Buffer, not a raw addr
  // We construct via ref.alloc then manually set the address
  const ptrBuf = Buffer.alloc(8);
  ptrBuf.writeBigUInt64LE(addr, 0);
  // Use ref to treat the address as a pointer
  const ptrRef = ref.readPointer(ptrBuf, 0, size);
  return ptrRef;
}

// ---------------------------------------------------------------------------
// startCapture(monitorSourceName) → EventEmitter
// Emits: 'pcm' (Float32Array mono 16kHz), 'error' (Error)
// Call .stop() to terminate
// ---------------------------------------------------------------------------
function startCapture(monitorSourceName) {
  const emitter = new EventEmitter();
  let running = true;

  emitter.stop = () => { running = false; };

  // Run capture in next tick so caller can attach listeners first
  setImmediate(() => {
    _captureLoop(monitorSourceName, emitter, () => running).catch(err => {
      emitter.emit('error', err);
    });
  });

  return emitter;
}

async function _captureLoop(sourceName, emitter, isRunning) {
  // 1. Register all input devices (includes pulse)
  libavdevice.avdevice_register_all();

  // 2. Find pulse input format
  const inputFmt = libavformat.av_find_input_format('pulse');
  if (inputFmt.isNull && inputFmt.isNull()) {
    throw new Error('Could not find pulse input format. Is FFmpeg built with --enable-libpulse?');
  }

  // 3. Set PulseAudio options: small fragment_size for low latency
  const optsRef = ref.alloc(voidPtr, ref.NULL);
  libavutil.av_dict_set(optsRef, 'fragment_size', '8192', 0);

  // 4. Open format context: avformat_open_input(&ctx, source, fmt, &opts)
  const fmtCtxRef = ref.alloc(voidPtr, ref.NULL);
  let ret = libavformat.avformat_open_input(fmtCtxRef, sourceName, inputFmt, optsRef);
  libavutil.av_dict_free(optsRef);
  if (ret < 0) throw new Error(`avformat_open_input failed: ${ret} (source=${sourceName})`);

  const fmtCtx = fmtCtxRef.deref();

  // 5. Find stream info
  ret = libavformat.avformat_find_stream_info(fmtCtx, ref.NULL);
  if (ret < 0) throw new Error(`avformat_find_stream_info failed: ${ret}`);

  // 6. Find best audio stream
  const audioStreamIdx = libavformat.av_find_best_stream(fmtCtx, AVMEDIA_TYPE_AUDIO, -1, -1, ref.NULL, 0);
  if (audioStreamIdx < 0) throw new Error('No audio stream found');

  // 7. Read streams array: AVFormatContext.streams is at offset 48, it's AVStream**
  //    fmtCtx is a Buffer/pointer to AVFormatContext
  const fmtCtxBuf = fmtCtx.reinterpret(256); // enough to read streams field at offset 48
  const streamsAddr = readPtrAt(fmtCtxBuf, OFFSETS.fmtctx.streams);

  // streams[audioStreamIdx] = *(streams + audioStreamIdx * sizeof(pointer))
  const streamPtrBuf = addrToBuffer(streamsAddr + BigInt(audioStreamIdx * 8), 8);
  const streamAddr = readPtrAt(streamPtrBuf, 0);
  const streamBuf = addrToBuffer(streamAddr, 64); // enough for codecpar at offset 16

  const codecparAddr = readPtrAt(streamBuf, OFFSETS.stream.codecpar);
  const codecparBuf = addrToBuffer(codecparAddr, 128); // enough for all fields

  const codecId = codecparBuf.readUInt32LE(OFFSETS.codecpar.codec_id);
  const srcSampleRate = codecparBuf.readInt32LE(OFFSETS.codecpar.sample_rate);
  const srcChannels = codecparBuf.readInt32LE(OFFSETS.codecpar.channels);

  // 8. Find and open decoder
  const decoder = libavcodec.avcodec_find_decoder(codecId);
  if (decoder.isNull && decoder.isNull()) throw new Error(`No decoder for codec_id=${codecId}`);

  const codecCtx = libavcodec.avcodec_alloc_context3(decoder);
  ret = libavcodec.avcodec_parameters_to_context(codecCtx, addrToBuffer(codecparAddr, 128));
  if (ret < 0) throw new Error(`avcodec_parameters_to_context failed: ${ret}`);

  ret = libavcodec.avcodec_open2(codecCtx, decoder, ref.NULL);
  if (ret < 0) throw new Error(`avcodec_open2 failed: ${ret}`);

  // 9. Setup resampler: srcChannels/srcSampleRate s16 → mono float32 16kHz
  const srcLayout = srcChannels === 1 ? AV_CH_LAYOUT_MONO : AV_CH_LAYOUT_STEREO;
  const swrCtx = libswresample.swr_alloc_set_opts(
    ref.NULL,
    AV_CH_LAYOUT_MONO,  // out channel layout
    AV_SAMPLE_FMT_FLT,  // out format
    16000,              // out sample rate
    srcLayout,          // in channel layout
    AV_SAMPLE_FMT_S16,  // in format
    srcSampleRate,      // in sample rate
    0, 0n               // logging
  );
  if (swrCtx.isNull && swrCtx.isNull()) throw new Error('swr_alloc_set_opts failed');
  ret = libswresample.swr_init(swrCtx);
  if (ret < 0) throw new Error(`swr_init failed: ${ret}`);

  // 10. Alloc packet and frame
  const pkt = libavcodec.av_packet_alloc();
  const frame = libavcodec.av_frame_alloc();

  const MAX_OUT_SAMPLES = 4096;
  const outBuf = Buffer.alloc(MAX_OUT_SAMPLES * 4); // float32
  const outBufPtr = ref.alloc(voidPtr, outBuf);

  // 11. Main read loop
  while (isRunning()) {
    ret = libavformat.av_read_frame(fmtCtx, pkt);
    if (ret < 0) {
      // AVERROR(EAGAIN) = -11, retry; else break
      if (ret === -11) { await _sleep(5); continue; }
      break;
    }

    const pktBuf = pkt.reinterpret(104);
    const streamIdx = pktBuf.readInt32LE(OFFSETS.packet.stream_index);

    if (streamIdx !== audioStreamIdx) {
      libavcodec.av_packet_unref(pkt);
      continue;
    }

    ret = libavcodec.avcodec_send_packet(codecCtx, pkt);
    libavcodec.av_packet_unref(pkt);
    if (ret < 0) continue;

    while (true) {
      ret = libavcodec.avcodec_receive_frame(codecCtx, frame);
      if (ret < 0) break; // EAGAIN or EOF

      const frameBuf = frame.reinterpret(480);
      const nbSamples = frameBuf.readInt32LE(OFFSETS.frame.nb_samples);
      if (nbSamples <= 0) continue;

      // Get pointer to first plane of input data
      const extDataAddr = readPtrAt(frameBuf, OFFSETS.frame.extended_data);
      const plane0Addr = readPtrAt(addrToBuffer(extDataAddr, 8), 0);
      const inPlaneRef = ref.alloc(voidPtr, addrToBuffer(plane0Addr, nbSamples * 4));

      // Resample to mono float32 16kHz
      const outSamples = Math.ceil(nbSamples * 16000 / srcSampleRate) + 64;
      const actualOut = libswresample.swr_convert(
        swrCtx,
        outBufPtr,
        Math.min(outSamples, MAX_OUT_SAMPLES),
        inPlaneRef,
        nbSamples
      );

      if (actualOut > 0) {
        const chunk = new Float32Array(outBuf.buffer.slice(0, actualOut * 4));
        emitter.emit('pcm', chunk);
      }
    }

    // Yield to event loop periodically to prevent blocking
    await _sleep(0);
  }

  // Cleanup
  const swrRef = ref.alloc(voidPtr, swrCtx);
  libswresample.swr_free(swrRef);
  const codecCtxRef = ref.alloc(voidPtr, codecCtx);
  libavcodec.avcodec_free_context(codecCtxRef);
  libavformat.avformat_close_input(fmtCtxRef);
}

function _sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

module.exports = { startCapture };
