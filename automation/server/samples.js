import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const SAMPLE_ID = /^[A-Za-z0-9_-]{12,80}$/;
const MAX_SAMPLE_BYTES = 12 * 1024 * 1024;
const TYPES = {
  'image/jpeg': { extension: '.jpg', signatures: [[0xff, 0xd8, 0xff]] },
  'image/png': { extension: '.png', signatures: [[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]] },
  'image/webp': { extension: '.webp', signatures: [[0x52, 0x49, 0x46, 0x46]] },
};

function sampleDir(dataDir) {
  return path.join(path.resolve(dataDir), 'samples');
}

function hasSignature(buffer, signatures) {
  return signatures.some((signature) => signature.every((byte, index) => buffer[index] === byte));
}

function safeName(value, extension) {
  const stem = path.basename(String(value || 'sample'), path.extname(String(value || 'sample')))
    .replace(/[^A-Za-z0-9._-]/g, '_').slice(0, 90) || 'sample';
  return `${stem}${extension}`;
}

export function saveSample(dataDir, input) {
  const type = String(input?.type || '').toLowerCase();
  const spec = TYPES[type];
  if (!spec) throw new Error('test sample must be a JPEG, PNG, or WebP image');
  const encoded = String(input?.data || '').replace(/^data:[^;,]+;base64,/, '');
  if (!encoded || !/^[A-Za-z0-9+/=\s]+$/.test(encoded)) throw new Error('test sample data is invalid');
  const buffer = Buffer.from(encoded, 'base64');
  if (!buffer.length || buffer.length > MAX_SAMPLE_BYTES) throw new Error('test sample must be 12 MiB or smaller');
  if (!hasSignature(buffer, spec.signatures)) throw new Error(`test sample content does not match ${type}`);
  if (type === 'image/webp' && buffer.subarray(8, 12).toString('ascii') !== 'WEBP') {
    throw new Error('test sample content does not match image/webp');
  }
  const id = crypto.randomBytes(16).toString('hex');
  const directory = sampleDir(dataDir);
  fs.mkdirSync(directory, { recursive: true });
  const fileName = `${id}${spec.extension}`;
  const meta = {
    id, name: safeName(input?.name, spec.extension), type, size: buffer.length,
    fileName, createdAt: Date.now(),
  };
  fs.writeFileSync(path.join(directory, fileName), buffer, { flag: 'wx' });
  fs.writeFileSync(path.join(directory, `${id}.json`), JSON.stringify(meta, null, 2), { flag: 'wx' });
  return meta;
}

export function resolveSample(dataDir, id) {
  const value = String(id || '');
  if (!SAMPLE_ID.test(value)) throw new Error('invalid test sample id');
  const directory = sampleDir(dataDir);
  const metaPath = path.join(directory, `${value}.json`);
  const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
  const file = path.join(directory, path.basename(meta.fileName));
  const stat = fs.statSync(file);
  if (!stat.isFile()) throw new Error('test sample is not a file');
  return { ...meta, path: file, stat };
}

export const sampleLimits = { maxBytes: MAX_SAMPLE_BYTES, requestBytes: 17 * 1024 * 1024 };
