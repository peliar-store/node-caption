'use strict';
/**
 * transcriber.js
 * Runs Whisper-base ONNX inference (encoder + decoder_model_merged).
 * Accepts Float32Array log-mel spectrogram [N_MELS * N_FRAMES] and returns text.
 *
 * Whisper-base architecture:
 *   - Encoder: input_features [1,80,3000] → last_hidden_state [1,1500,512]
 *   - Decoder (merged): uses use_cache_branch to switch between first/cached pass
 *     - 6 decoder layers, 8 attention heads, head_dim=64
 *     - Encoder cross-attention KV shape: [1,8,1500,64]
 *     - Decoder self-attention KV grows each step
 */

const ort = require('onnxruntime-node');
const fs  = require('fs');
const path = require('path');

// ---------------------------------------------------------------------------
// Whisper special token IDs (whisper-base multilingual)
// ---------------------------------------------------------------------------
const T = {
  EOT:           50257,
  SOT:           50258,
  LANG_EN:       50259,
  TRANSCRIBE:    50359,
  NO_TIMESTAMPS: 50363,
  NO_SPEECH:     50362,
  BLANK:         220,   // space token
};

const N_MELS   = 80;
const N_FRAMES = 3000;
const N_LAYERS = 6;   // whisper-base decoder layers
const N_HEADS  = 8;
const HEAD_DIM = 64;
const MAX_TOKENS = 224;
const NO_SPEECH_THRESHOLD = 0.6; // logit softmax prob above which to skip

class Transcriber {
  constructor(encoderPath, decoderPath, vocabPath) {
    this.encoderPath = encoderPath;
    this.decoderPath = decoderPath;

    const raw = JSON.parse(fs.readFileSync(vocabPath, 'utf8'));
    // Build id→token map
    this._idToToken = {};
    for (const [token, id] of Object.entries(raw)) {
      this._idToToken[id] = token;
    }

    this.encoder = null;
    this.decoder = null;
  }

  async init() {
    const opts = {
      executionProviders: ['cpu'],
    };
    this.encoder = await ort.InferenceSession.create(this.encoderPath, opts);
    this.decoder = await ort.InferenceSession.create(this.decoderPath, opts);
  }

  /**
   * Transcribe a log-mel spectrogram.
   * @param {Float32Array} melData  Flat array of length N_MELS*N_FRAMES (240000)
   * @returns {Promise<string>} transcribed text, or '' if no speech detected
   */
  async transcribe(melData) {
    if (!this.encoder || !this.decoder) throw new Error('Call init() first');

    // -----------------------------------------------------------------------
    // 1. Encoder forward pass
    // -----------------------------------------------------------------------
    const inputFeatures = new ort.Tensor('float32', melData, [1, N_MELS, N_FRAMES]);
    const encOut = await this.encoder.run({ input_features: inputFeatures });
    const encoderHidden = encOut.last_hidden_state; // [1, 1500, 512]

    // -----------------------------------------------------------------------
    // 2. Check no-speech probability using first decoder step
    // -----------------------------------------------------------------------
    // (We'll do this implicitly: if the greedy token is NO_SPEECH, return '')

    // -----------------------------------------------------------------------
    // 3. Decoder greedy loop
    // -----------------------------------------------------------------------
    // Initial prompt tokens: [SOT, LANG_EN, TRANSCRIBE, NO_TIMESTAMPS]
    const promptTokens = [T.SOT, T.LANG_EN, T.TRANSCRIBE, T.NO_TIMESTAMPS];

    let inputIds  = Int32Array.from(promptTokens);
    let useCacheB = false; // first step: full attention, no cache used
    let pastKV    = this._emptyPastKV();
    const generated = [];

    for (let step = 0; step < MAX_TOKENS; step++) {
      const seqLen = inputIds.length;

      const feeds = {
        input_ids: new ort.Tensor('int64',
          BigInt64Array.from(Array.from(inputIds).map(BigInt)),
          [1, seqLen]
        ),
        encoder_hidden_states: encoderHidden,
        use_cache_branch: new ort.Tensor('bool', [useCacheB], [1]),
        ...pastKV,
      };

      const out = await this.decoder.run(feeds);
      const logits = out.logits; // [1, seqLen, vocab_size]
      const vocabSize = logits.dims[2];

      // Argmax of last token position
      const lastOffset = (seqLen - 1) * vocabSize;
      let maxLogit = -Infinity;
      let nextToken = 0;
      for (let v = 0; v < vocabSize; v++) {
        const l = logits.data[lastOffset + v];
        if (l > maxLogit) { maxLogit = l; nextToken = v; }
      }

      // Handle no-speech on first step
      if (step === 0 && nextToken === T.NO_SPEECH) return '';

      // End of transcript
      if (nextToken === T.EOT) break;

      // Collect non-special tokens
      if (nextToken < 50257) {
        generated.push(nextToken);
      }

      // Update KV cache.
      // First pass (step==0): update all KV including encoder cross-attn.
      // Cached passes: only update decoder self-attn KV; encoder KV is stable.
      if (step === 0) {
        pastKV = this._extractAllKV(out);
      } else {
        pastKV = this._updateDecoderKV(pastKV, out);
      }
      inputIds  = Int32Array.from([nextToken]);
      useCacheB = true;
    }

    return this._decodeTokens(generated);
  }

  // -------------------------------------------------------------------------
  // Past key-value helpers
  // -------------------------------------------------------------------------
  _emptyPastKV() {
    const kv = {};
    // Encoder sequence length from encoder output: 1500 frames
    const ENC_SEQ = 1500;
    const encSize = N_HEADS * ENC_SEQ * HEAD_DIM; // 8*1500*64 = 768000
    for (let i = 0; i < N_LAYERS; i++) {
      // Decoder self-attention: seq dim = 0 initially
      kv[`past_key_values.${i}.decoder.key`]   = new ort.Tensor('float32', new Float32Array(0), [1, N_HEADS, 0, HEAD_DIM]);
      kv[`past_key_values.${i}.decoder.value`] = new ort.Tensor('float32', new Float32Array(0), [1, N_HEADS, 0, HEAD_DIM]);
      // Encoder cross-attention: must have seq=ENC_SEQ on first pass (even though model recomputes them
      // when use_cache_branch=false, it still reshapes the past tensors requiring this shape)
      kv[`past_key_values.${i}.encoder.key`]   = new ort.Tensor('float32', new Float32Array(encSize), [1, N_HEADS, ENC_SEQ, HEAD_DIM]);
      kv[`past_key_values.${i}.encoder.value`] = new ort.Tensor('float32', new Float32Array(encSize), [1, N_HEADS, ENC_SEQ, HEAD_DIM]);
    }
    return kv;
  }

  _extractAllKV(decoderOut) {
    // First pass: extract all KV including encoder cross-attn (now computed from encoder_hidden_states)
    const kv = {};
    for (let i = 0; i < N_LAYERS; i++) {
      kv[`past_key_values.${i}.decoder.key`]   = decoderOut[`present.${i}.decoder.key`];
      kv[`past_key_values.${i}.decoder.value`] = decoderOut[`present.${i}.decoder.value`];
      kv[`past_key_values.${i}.encoder.key`]   = decoderOut[`present.${i}.encoder.key`];
      kv[`past_key_values.${i}.encoder.value`] = decoderOut[`present.${i}.encoder.value`];
    }
    return kv;
  }

  _updateDecoderKV(prevKV, decoderOut) {
    // Cached pass: only update decoder self-attention KV (grows each step).
    // Encoder cross-attention KV stays stable from the first pass.
    const kv = { ...prevKV };
    for (let i = 0; i < N_LAYERS; i++) {
      kv[`past_key_values.${i}.decoder.key`]   = decoderOut[`present.${i}.decoder.key`];
      kv[`past_key_values.${i}.decoder.value`] = decoderOut[`present.${i}.decoder.value`];
    }
    return kv;
  }

  // -------------------------------------------------------------------------
  // BPE token decoding (GPT-2 byte-level encoding used by Whisper)
  // -------------------------------------------------------------------------
  _decodeTokens(tokenIds) {
    // Join token strings, then undo GPT-2 byte-level encoding
    // Ġ (U+0120) represents a leading space in GPT-2 BPE
    // Ċ (U+010A) represents newline
    const parts = tokenIds.map(id => this._idToToken[id] || '');
    let text = parts.join('');

    // GPT-2 unicode → actual bytes reversal
    text = _gpt2ByteDecode(text);

    return text.trim();
  }
}

// ---------------------------------------------------------------------------
// GPT-2 byte-level BPE unicode → string decoding
// Whisper/GPT-2 maps each byte 0-255 to a unique unicode char to avoid
// invalid UTF-8 sequences. We reverse this mapping.
// ---------------------------------------------------------------------------
function _buildBytesDecoder() {
  // This is the canonical GPT-2 bytes_to_unicode() mapping in reverse
  const bs = [];
  const cs = [];
  // Printable ASCII ranges that map to themselves
  for (let i = '!'.charCodeAt(0); i <= '~'.charCodeAt(0); i++) { bs.push(i); cs.push(i); }
  for (let i = '¡'.charCodeAt(0); i <= '¬'.charCodeAt(0); i++) { bs.push(i); cs.push(i); }
  for (let i = '®'.charCodeAt(0); i <= 'ÿ'.charCodeAt(0); i++) { bs.push(i); cs.push(i); }
  // Remaining bytes 0-255 map to chars starting at 256
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  // Build reverse: unicode codepoint → byte value
  const decoder = {};
  for (let i = 0; i < bs.length; i++) {
    decoder[String.fromCodePoint(cs[i])] = bs[i];
  }
  return decoder;
}

const _bytesDecoder = _buildBytesDecoder();

function _gpt2ByteDecode(text) {
  // text is a sequence of unicode chars from the GPT-2 BPE encoding
  // convert back to UTF-8 bytes then decode as UTF-8
  const bytes = [];
  for (const ch of text) {
    const b = _bytesDecoder[ch];
    if (b !== undefined) {
      bytes.push(b);
    }
    // unknown chars: skip (shouldn't happen with valid vocab)
  }
  return Buffer.from(bytes).toString('utf8');
}

module.exports = { Transcriber };
