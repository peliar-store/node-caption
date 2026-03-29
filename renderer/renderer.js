'use strict';
/**
 * renderer.js — Caption display logic
 *
 * Behavior (mimics Google Live Captions):
 *  - New text from main process is appended to the current line
 *  - When a line gets long enough or ends with sentence punctuation, start a new line
 *  - Keep at most 3 lines visible; oldest lines pushed off top
 *  - Each line auto-expires after LINE_TTL_MS of no updates
 *  - Older lines are dimmed so the newest line stands out
 */

const MAX_LINES     = 3;
const LINE_TTL_MS   = 7000;  // remove line after this many ms of silence
const NEW_LINE_CHARS = 60;   // start new line when current exceeds this length

const captionsEl = document.getElementById('captions');
let lines = []; // [{el, text, timer}]

window.captionAPI.onCaption((text) => {
  if (!text) return;

  // Decide whether to append to current line or start fresh
  if (
    lines.length === 0 ||
    lines[lines.length - 1].text.length >= NEW_LINE_CHARS ||
    _endsWithSentence(lines[lines.length - 1].text)
  ) {
    _addLine(text);
  } else {
    _appendToLine(lines[lines.length - 1], text);
  }

  _trimLines();
  _updateDimming();
});

function _addLine(text) {
  const el = document.createElement('div');
  el.className = 'caption-line';
  el.textContent = text;
  captionsEl.appendChild(el);

  const entry = { el, text, timer: null };
  entry.timer = setTimeout(() => _removeLine(entry), LINE_TTL_MS);
  lines.push(entry);
}

function _appendToLine(entry, text) {
  clearTimeout(entry.timer);
  // Whisper output already has leading space from BPE decoding when appropriate
  entry.text += text;
  entry.el.textContent = entry.text;
  entry.timer = setTimeout(() => _removeLine(entry), LINE_TTL_MS);
}

function _removeLine(entry) {
  entry.el.remove();
  lines = lines.filter(l => l !== entry);
  _updateDimming();
}

function _trimLines() {
  while (lines.length > MAX_LINES) {
    const oldest = lines.shift();
    clearTimeout(oldest.timer);
    oldest.el.remove();
  }
}

function _updateDimming() {
  lines.forEach((l, i) => {
    const isNewest = i === lines.length - 1;
    l.el.classList.toggle('dimmed', !isNewest);
  });
}

function _endsWithSentence(text) {
  return /[.!?]\s*$/.test(text.trimEnd());
}
