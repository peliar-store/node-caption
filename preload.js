'use strict';
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('captionAPI', {
  onCaption: (callback) => {
    ipcRenderer.on('caption', (_event, text) => callback(text));
  },
});
