# node-caption: Electron Live Caption App

## Context
Build a minimal Electron app that captures all system audio via PulseAudio monitor source, transcribes it in real-time using Whisper ONNX (whisper-base), and displays rolling captions in a transparent always-on-top overlay — imitating Google's live caption behavior.

**User choices:** whisper-base model (74MB encoder + 208MB merged decoder), PulseAudio monitor source on Linux.

## Tech Stack
- **Audio capture**: `ffi-napi` calling FFmpeg system `.so` libs directly (no native addon)
- **Transcription**: `onnxruntime-node` + whisper-base ONNX (encoder + decoder_model_merged)
- **FFT**: `fft.js` (pure JS, 512-point FFT, zero-pad 400-sample Hann window)
- **Frontend**: transparent, click-through, always-on-top BrowserWindow — caption text only
- **IPC**: `contextBridge` → `ipcRenderer.on('caption', ...)` → DOM update

## System Facts (verified)
- FFmpeg 6.1.1 with `--enable-libpulse` installed
- Libs: `libavdevice.so.60`, `libavformat.so.60`, `libavcodec.so.60`, `libavutil.so.58`, `libswresample.so.4`
- PipeWire 1.0.5 running PulseAudio compat; monitor source confirmed working
- Node 18.19.1 / npm 9.2.0
- ONNX models in `/home/juan-fernandez/Downloads/`: `encoder_model.onnx` (82MB), `decoder_model_merged.onnx` (208MB)

## File Structure
```
node-caption/
├── package.json
├── main.js
├── preload.js
├── renderer/
│   ├── index.html
│   └── renderer.js
├── src/
│   ├── audio-capture.js
│   ├── audio-processor.js
│   └── transcriber.js
├── assets/
│   ├── encoder_model.onnx    (symlink from Downloads)
│   └── decoder_model_merged.onnx
└── vocab/
    └── whisper-vocab.json
```

## Implementation Order
1. package.json + directory structure
2. assets/ symlinks + vocab/whisper-vocab.json download script
3. src/audio-capture.js (ffi-napi FFmpeg/PulseAudio)
4. src/audio-processor.js (ring buffer, mel spectrogram)
5. src/transcriber.js (ONNX whisper inference)
6. main.js + preload.js
7. renderer/index.html + renderer/renderer.js
