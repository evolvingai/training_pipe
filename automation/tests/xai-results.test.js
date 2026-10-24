import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { listClassificationXaiResults, resolveClassificationXaiAsset } from '../server/xai-results.js';

function sampleRun(root) {
  const run = path.join(root, 'classifier_xai_sample');
  for (const directory of ['inputs', 'gradcam', 'activations']) {
    fs.mkdirSync(path.join(run, 'xai', directory), { recursive: true });
  }
  fs.writeFileSync(path.join(run, 'xai', 'inputs', 'input.jpg'), 'input');
  fs.writeFileSync(path.join(run, 'xai', 'gradcam', 'signed.png'), 'map');
  fs.writeFileSync(path.join(run, 'xai', 'activations', 'stages.jpg'), 'activations');
  fs.writeFileSync(path.join(run, 'xai', 'manifest.json'), JSON.stringify({
    format: 'trainer-classification-xai-v1',
    architecture: 'convnext-tiny',
    images: [{
      token: 'internal-file-token', predicted: 'cat', confidence: 0.91, target: 'cat', target_score: 0.91,
      layer: 'stage_4', topk: [{ class: 'cat', confidence: 0.91 }],
      input: 'xai/inputs/input.jpg', signed_map: 'xai/gradcam/signed.png', activations: 'xai/activations/stages.jpg',
    }],
  }));
  return run;
}

test('xAI API model exposes sample results without file names or paths', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'trainer-xai-results-'));
  sampleRun(root);
  const runs = listClassificationXaiResults(root);
  assert.equal(runs.length, 1);
  assert.equal(runs[0].samples[0].label, 'Sample 1');
  assert.equal(runs[0].samples[0].predicted, 'cat');
  assert.match(runs[0].samples[0].inputUrl, /\/api\/xai\/classification\//);
  assert.doesNotMatch(JSON.stringify(runs), /manifest\.json|metrics\.json|input\.jpg|signed\.png|stages\.jpg|internal-file-token/);
  fs.rmSync(root, { recursive: true, force: true });
});

test('xAI asset resolver serves only the images belonging to the sample result', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'trainer-xai-assets-'));
  sampleRun(root);
  const input = resolveClassificationXaiAsset(root, 'classifier_xai_sample', 0, 'input');
  assert.equal(input.contentType, 'image/jpeg');
  assert.throws(() => resolveClassificationXaiAsset(root, '../outside', 0, 'input'), /invalid xAI run/);
  assert.throws(() => resolveClassificationXaiAsset(root, 'classifier_xai_sample', 0, 'manifest'), /asset not found/);
  fs.rmSync(root, { recursive: true, force: true });
});
