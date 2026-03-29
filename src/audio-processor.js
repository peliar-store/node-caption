'use strict';
/**
 * audio-processor.js
 * Accumulates raw PCM Float32 chunks (16kHz mono) from audio-capture,
 * then emits log-mel spectrogram tensors [1, 80, 3000] for Whisper.
 *
 * Whisper preprocessing spec:
 *   - N_FFT = 400 samples (25ms at 16kHz)
 *   - HOP = 160 samples (10ms)
 *   - N_MELS = 80
 *   - N_FRAMES = 3000  → covers 30 seconds
 *   - FFT is done with 512-point (next power of 2), window padded to 512
 *   - Mel filterbank: 80 triangular filters, HTK scale, 0–8000 Hz
 *   - Normalization: log10(max(energy, 1e-10)), then (x - max + 8) / 8 clamped to [-1, 1]
 */

const { EventEmitter } = require('events');
const FFT = require('fft.js');

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const SAMPLE_RATE     = 16000;
const N_FFT           = 400;   // physical window length (whisper spec)
const FFT_SIZE        = 512;   // next power of 2, used for fft.js
const HOP_LENGTH      = 160;
const N_MELS          = 80;
const N_FRAMES        = 3000;

// Emit a mel chunk every STEP_SAMPLES new samples; keep OVERLAP_SAMPLES of history
const CHUNK_SECONDS   = 3;
const OVERLAP_SECONDS = 1;
const CHUNK_SAMPLES   = SAMPLE_RATE * CHUNK_SECONDS;   // 48000
const STEP_SAMPLES    = SAMPLE_RATE * (CHUNK_SECONDS - OVERLAP_SECONDS); // 32000

// ---------------------------------------------------------------------------
// Precomputed tables (built once at module load)
// ---------------------------------------------------------------------------
const _hannWindow = _buildHannWindow(N_FFT);
const _melFilters = _buildMelFilterbank();
const _fftInst    = new FFT(FFT_SIZE);

function _buildHannWindow(n) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
  }
  return w;
}

function _hzToMel(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}

function _melToHz(mel) {
  return 700 * (Math.pow(10, mel / 2595) - 1);
}

function _buildMelFilterbank() {
  const fMin = 0;
  const fMax = 8000;
  const melMin = _hzToMel(fMin);
  const melMax = _hzToMel(fMax);

  // N_MELS + 2 mel center frequencies (including edges)
  const melPoints = new Array(N_MELS + 2);
  for (let i = 0; i < N_MELS + 2; i++) {
    melPoints[i] = _melToHz(melMin + (i * (melMax - melMin)) / (N_MELS + 1));
  }

  // FFT frequency bins: using N_FFT as the resolution basis (not FFT_SIZE)
  // This matches whisper.cpp's approach: freq[k] = k * sr / N_FFT
  const numBins = N_FFT / 2 + 1; // 201
  const fftFreqs = new Array(numBins);
  for (let k = 0; k < numBins; k++) {
    fftFreqs[k] = (k * SAMPLE_RATE) / N_FFT;
  }

  // Build triangular filters
  const filters = new Array(N_MELS);
  for (let m = 0; m < N_MELS; m++) {
    const lower  = melPoints[m];
    const center = melPoints[m + 1];
    const upper  = melPoints[m + 2];
    const row = new Float32Array(numBins);
    for (let k = 0; k < numBins; k++) {
      const f = fftFreqs[k];
      if (f >= lower && f <= center) {
        row[k] = (f - lower) / (center - lower);
      } else if (f > center && f <= upper) {
        row[k] = (upper - f) / (upper - center);
      }
      // else 0 (default)
    }
    filters[m] = row;
  }
  return filters;
}

// ---------------------------------------------------------------------------
// Mel spectrogram computation
// ---------------------------------------------------------------------------
function computeLogMel(samples) {
  // samples: Float32Array of at least CHUNK_SAMPLES; we use first CHUNK_SAMPLES
  const numBins = N_FFT / 2 + 1; // 201
  const melOut = new Float32Array(N_MELS * N_FRAMES);

  // fft.js works with interleaved real/imag arrays
  const fftInput  = new Array(FFT_SIZE * 2).fill(0);
  const fftOutput = new Array(FFT_SIZE * 2).fill(0);

  for (let frame = 0; frame < N_FRAMES; frame++) {
    const offset = frame * HOP_LENGTH;

    // Fill first N_FFT samples with windowed audio, rest with 0
    for (let i = 0; i < FFT_SIZE; i++) {
      const real = i < N_FFT && (offset + i) < samples.length
        ? samples[offset + i] * _hannWindow[i]
        : 0;
      fftInput[2 * i]     = real;
      fftInput[2 * i + 1] = 0;
    }

    _fftInst.transform(fftOutput, fftInput);

    // Compute power spectrum for first numBins (201) bins
    // then apply mel filterbank to get energy per mel band
    for (let m = 0; m < N_MELS; m++) {
      let energy = 0;
      const filter = _melFilters[m];
      for (let k = 0; k < numBins; k++) {
        const re = fftOutput[2 * k];
        const im = fftOutput[2 * k + 1];
        energy += filter[k] * (re * re + im * im);
      }
      melOut[m * N_FRAMES + frame] = Math.log10(Math.max(energy, 1e-10));
    }
  }

  // Whisper normalization: find global max, then (x - max + 8) / 8, clamped to [-1, 1]
  let maxVal = -Infinity;
  for (let i = 0; i < melOut.length; i++) {
    if (melOut[i] > maxVal) maxVal = melOut[i];
  }
  for (let i = 0; i < melOut.length; i++) {
    melOut[i] = Math.max((melOut[i] - maxVal + 8) / 8, -1);
  }

  return melOut; // Float32Array of length N_MELS * N_FRAMES = 240000
}

// ---------------------------------------------------------------------------
// AudioProcessor: buffers PCM and emits mel tensors on sliding window
// ---------------------------------------------------------------------------
class AudioProcessor extends EventEmitter {
  constructor() {
    super();
    // Ring buffer: hold 2 * CHUNK_SAMPLES to support overlap
    this._ring     = new Float32Array(CHUNK_SAMPLES * 2);
    this._writePos = 0;   // absolute write position
    this._lastEmit = 0;   // absolute position of last emit
  }

  /**
   * Push a new Float32Array of mono 16kHz PCM samples.
   * Emits 'mel' event with Float32Array [N_MELS * N_FRAMES] when enough data.
   */
  push(chunk) {
    const ringLen = this._ring.length;
    for (let i = 0; i < chunk.length; i++) {
      this._ring[this._writePos % ringLen] = chunk[i];
      this._writePos++;
    }

    // Emit once we have at least CHUNK_SAMPLES total and enough new samples since last emit
    while (
      this._writePos >= CHUNK_SAMPLES &&
      this._writePos - this._lastEmit >= STEP_SAMPLES
    ) {
      const samples = this._extractChunk();
      const mel = computeLogMel(samples);
      this.emit('mel', mel);
      this._lastEmit = this._writePos;
    }
  }

  _extractChunk() {
    const out = new Float32Array(CHUNK_SAMPLES);
    const ringLen = this._ring.length;
    const start = this._writePos - CHUNK_SAMPLES;
    for (let i = 0; i < CHUNK_SAMPLES; i++) {
      out[i] = this._ring[(start + i + ringLen * 2) % ringLen];
    }
    return out;
  }
}

module.exports = { AudioProcessor, computeLogMel };
