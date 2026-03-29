'use strict';
const { app, BrowserWindow, ipcMain, screen } = require('electron');
const path = require('path');

const { startCapture } = require('./src/audio-capture');
const { AudioProcessor } = require('./src/audio-processor');
const { Transcriber } = require('./src/transcriber');

// Default monitor source — user can override via env var
const MONITOR_SOURCE = process.env.CAPTION_SOURCE ||
  'alsa_output.pci-0000_02_02.0.analog-stereo.monitor';

let win = null;
let capture = null;
let transcribing = false;

function createWindow() {
  const { width: sw, height: sh } = screen.getPrimaryDisplay().workAreaSize;

  win = new BrowserWindow({
    width:       Math.min(900, sw),
    height:      130,
    x:           Math.floor((sw - Math.min(900, sw)) / 2),
    y:           sh - 145,
    transparent: true,
    frame:       false,
    alwaysOnTop: true,
    skipTaskbar: true,
    focusable:   false,
    resizable:   false,
    hasShadow:   false,
    webPreferences: {
      preload:          path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration:  false,
      sandbox:          false,
    },
  });

  win.setIgnoreMouseEvents(true);
  win.loadFile(path.join(__dirname, 'renderer', 'index.html'));
}

app.whenReady().then(async () => {
  createWindow();

  const transcriber = new Transcriber(
    path.join(__dirname, 'assets', 'encoder_model.onnx'),
    path.join(__dirname, 'assets', 'decoder_model_merged.onnx'),
    path.join(__dirname, 'vocab', 'whisper-vocab.json')
  );

  console.log('[caption] Loading Whisper ONNX models...');
  await transcriber.init();
  console.log('[caption] Models loaded. Starting audio capture on:', MONITOR_SOURCE);

  const processor = new AudioProcessor();

  processor.on('mel', async (melData) => {
    if (transcribing) return; // drop chunk if previous inference still running
    transcribing = true;
    try {
      const text = await transcriber.transcribe(melData);
      if (text && win && !win.isDestroyed()) {
        win.webContents.send('caption', text);
      }
    } catch (err) {
      console.error('[caption] Transcription error:', err.message);
    } finally {
      transcribing = false;
    }
  });

  capture = startCapture(MONITOR_SOURCE);

  capture.on('pcm', (chunk) => {
    processor.push(chunk);
  });

  capture.on('error', (err) => {
    console.error('[caption] Audio capture error:', err.message);
    if (win && !win.isDestroyed()) {
      win.webContents.send('caption', `[Error: ${err.message}]`);
    }
  });
});

app.on('before-quit', () => {
  if (capture) capture.stop();
});

// Keep the app running even if the window is closed accidentally
app.on('window-all-closed', (e) => {
  e.preventDefault();
});
