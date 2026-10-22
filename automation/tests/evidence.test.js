import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { evidenceContentType, resolveEvidenceFile, scanEvidence } from '../server/evidence.js';

test('evidence scan classifies xAI, mask diagnostics, reports and models', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'trainer-evidence-'));
  fs.mkdirSync(path.join(root, 'classification_run', 'plots'), { recursive: true });
  fs.writeFileSync(path.join(root, 'classification_run', 'plots', 'gradcam.jpg'), 'x');
  fs.writeFileSync(path.join(root, 'classification_run', 'plots', 'mask_background.png'), 'x');
  fs.writeFileSync(path.join(root, 'classification_run', 'report.html'), '<html></html>');
  fs.writeFileSync(path.join(root, 'classification_run', 'best.pt'), 'x');
  const value = scanEvidence(root, { family: 'classification' });
  assert.deepEqual(new Set(value.items.map((item) => item.kind)), new Set(['xai', 'mask-diagnostic', 'report', 'model']));
  fs.rmSync(root, { recursive: true, force: true });
});

test('evidence file resolver rejects traversal', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'trainer-evidence-'));
  fs.writeFileSync(path.join(root, 'report.html'), '<html></html>');
  assert.equal(resolveEvidenceFile(root, 'report.html').path, fs.realpathSync(path.join(root, 'report.html')));
  assert.throws(() => resolveEvidenceFile(root, '../outside'), /outside the runs directory/);
  assert.equal(evidenceContentType('preview.png'), 'image/png');
  fs.rmSync(root, { recursive: true, force: true });
});

