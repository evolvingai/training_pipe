import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { saveSyncedEvidence } from '../server/evidence.js';

test('remote agents can sync only bounded classifier xAI evidence', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'trainer-evidence-sync-'));
  const saved = saveSyncedEvidence(root, {
    relativePath: 'classifier_xai_sample/xai/manifest.json',
    data: Buffer.from('{"images":[]}').toString('base64'),
  });
  assert.equal(saved.relativePath, 'classifier_xai_sample/xai/manifest.json');
  assert.equal(fs.readFileSync(path.join(root, saved.relativePath), 'utf8'), '{"images":[]}');
  assert.throws(() => saveSyncedEvidence(root, {
    relativePath: '../outside.json', data: Buffer.from('x').toString('base64'),
  }), /only sample xAI data/);
  assert.throws(() => saveSyncedEvidence(root, {
    relativePath: 'detector/model.pt', data: Buffer.from('x').toString('base64'),
  }), /only sample xAI data/);
  assert.throws(() => saveSyncedEvidence(root, {
    relativePath: 'classifier_xai_sample/reports/xai.html', data: Buffer.from('x').toString('base64'),
  }), /only sample xAI data/);
  fs.rmSync(root, { recursive: true, force: true });
});

