import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { discoverRuns } from '../server/runs.js';

test('run discovery includes direct classifier and xAI artifacts', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'trainer-runs-'));
  fs.mkdirSync(path.join(root, 'classifier_train', 'cfg'), { recursive: true });
  fs.writeFileSync(path.join(root, 'classifier_train', 'cfg', 'args.json'), JSON.stringify({ command: 'train', arch: 'convnext-tiny', device: 'cpu' }));
  fs.writeFileSync(path.join(root, 'classifier_train', 'metrics.json'), '{}');
  fs.mkdirSync(path.join(root, 'sample_xai', 'xai'), { recursive: true });
  fs.writeFileSync(path.join(root, 'sample_xai', 'xai', 'manifest.json'), JSON.stringify({ format: 'trainer-classification-xai-v1', architecture: 'convnext-tiny', images: [] }));
  const runs = discoverRuns(root);
  assert.equal(runs.length, 2);
  assert.ok(runs.every((run) => run.family === 'classification'));
  assert.deepEqual(new Set(runs.map((run) => run.action)), new Set(['train', 'xai']));
  fs.rmSync(root, { recursive: true, force: true });
});

test('run discovery expands a VM smoke matrix into every model family', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'trainer-runs-'));
  const matrix = path.join(root, 'vm_smoke_matrix');
  fs.mkdirSync(matrix, { recursive: true });
  fs.writeFileSync(path.join(matrix, 'summary.json'), JSON.stringify({
    format: 'trainer-vm-smoke-matrix-v1', device: 'cuda:0', families: {
      detection: { status: 'passed', seconds: 1 },
      classification: { status: 'passed', seconds: 2 },
      'video-classification': { status: 'passed', seconds: 3 },
      vlm: { status: 'failed', seconds: 4, error: 'fixture failure' },
    },
  }));
  const runs = discoverRuns(root);
  assert.deepEqual(new Set(runs.map((run) => run.family)), new Set(['detection', 'classification', 'video', 'vlm']));
  assert.equal(runs.find((run) => run.family === 'vlm').status, 'failed');
  fs.rmSync(root, { recursive: true, force: true });
});
