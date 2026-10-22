import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { makeSession, sessionUser, validLogin } from '../server/auth.js';
import { FileStore } from '../server/store.js';

test('signed sessions authenticate and reject tampering', () => {
  const token = makeSession('operator', 'secret');
  const request = { headers: { cookie: `trainer_session=${token}` } };
  assert.equal(sessionUser(request, 'secret'), 'operator');
  assert.equal(sessionUser({ headers: { cookie: `trainer_session=${token}x` } }, 'secret'), null);
  assert.equal(validLogin('operator', 'pass', 'operator', 'pass'), true);
  assert.equal(validLogin('operator', 'bad', 'operator', 'pass'), false);
});

test('store does not give a CUDA job to a CPU-only agent', async (context) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'trainer-store-'));
  context.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const store = new FileStore(root);
  const base = { name: 'job', type: 'detect_train', targetAgentId: null, createdAt: Date.now(), status: 'queued' };
  await store.createJob({ ...base, id: 'cuda', params: { device: 'cuda' } });
  assert.equal(await store.claimJob('cpu', { capabilities: { gpus: [] } }), null);
  const claimed = await store.claimJob('gpu', { capabilities: { gpus: [{ name: 'L4' }] } });
  assert.equal(claimed.id, 'cuda');
  assert.equal(claimed.status, 'running');
});
