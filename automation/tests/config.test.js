import assert from 'node:assert/strict';
import test from 'node:test';

import { config, parseManagedTargets } from '../server/config.js';

test('control plane defaults to listening on all interfaces', () => {
  assert.equal(config.host, '0.0.0.0');
});

test('managed CPU and GPU targets remain visible before heartbeats', () => {
  const targets = parseManagedTargets(JSON.stringify([
    { agentId: 'cpu-vm', label: 'CPU VM', cpus: 8 },
    { agentId: 'gpu-l4', label: 'L4 VM', gpus: [{ name: 'NVIDIA L4', memoryTotalMb: 23034 }] },
  ]));
  assert.deepEqual(Object.keys(targets), ['cpu-vm', 'gpu-l4']);
  assert.deepEqual(targets['cpu-vm'].gpus, []);
  assert.equal(targets['gpu-l4'].gpus[0].name, 'NVIDIA L4');
  assert.equal(targets['gpu-l4'].at, 0);
});

test('managed target ids are validated', () => {
  assert.throws(() => parseManagedTargets('[{"agentId":"bad id"}]'), /invalid agentId/);
  assert.throws(() => parseManagedTargets('{}'), /JSON array/);
});
