import assert from 'node:assert/strict';
import test from 'node:test';

import { makePowerDriver } from '../server/power.js';

test('GCE driver reads status and calls the official start/stop endpoints', async (context) => {
  const originalFetch = globalThis.fetch;
  const calls = [];
  context.after(() => { globalThis.fetch = originalFetch; });
  globalThis.fetch = async (url, options = {}) => {
    calls.push({ url: String(url), method: options.method || 'GET', authorization: options.headers?.authorization });
    return new Response(JSON.stringify({ status: 'RUNNING', machineType: 'zones/z/machineTypes/n2-standard-8' }), {
      status: 200, headers: { 'content-type': 'application/json' },
    });
  };
  const driver = makePowerDriver({
    powerDriver: 'gce',
    gce: { project: 'project', zone: 'zone', instance: 'worker', accessToken: 'token' },
  });
  assert.deepEqual(await driver.status(), { configured: true, state: 'running', detail: 'n2-standard-8' });
  await driver.start();
  await driver.stop();
  assert.deepEqual(calls.map((call) => call.url), [
    'https://compute.googleapis.com/compute/v1/projects/project/zones/zone/instances/worker',
    'https://compute.googleapis.com/compute/v1/projects/project/zones/zone/instances/worker/start',
    'https://compute.googleapis.com/compute/v1/projects/project/zones/zone/instances/worker/stop',
  ]);
  assert.ok(calls.every((call) => call.authorization === 'Bearer token'));
});

test('incomplete cloud configuration stays safely disabled', async () => {
  const driver = makePowerDriver({ powerDriver: 'azure', azure: {} });
  assert.equal(driver.configured, false);
  assert.equal((await driver.status()).state, 'unknown');
  await assert.rejects(driver.start(), /not configured/);
});
