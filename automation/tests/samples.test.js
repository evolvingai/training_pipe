import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { resolveSample, saveSample } from '../server/samples.js';

test('test samples are validated, stored, and resolved by opaque id', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'trainer-sample-'));
  const jpeg = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10]);
  const sample = saveSample(root, {
    name: '../cat photo.jpeg', type: 'image/jpeg', data: jpeg.toString('base64'),
  });
  assert.match(sample.id, /^[a-f0-9]{32}$/);
  assert.equal(sample.name, 'cat_photo.jpg');
  assert.deepEqual(fs.readFileSync(resolveSample(root, sample.id).path), jpeg);
  assert.throws(() => resolveSample(root, '../outside'), /invalid test sample id/);
  fs.rmSync(root, { recursive: true, force: true });
});

test('test sample content must match its declared image type', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'trainer-sample-'));
  assert.throws(() => saveSample(root, {
    name: 'not-an-image.jpg', type: 'image/jpeg', data: Buffer.from('plain text').toString('base64'),
  }), /does not match/);
  fs.rmSync(root, { recursive: true, force: true });
});
