import fs from 'node:fs';
import path from 'node:path';

const MAX_SYNC_BYTES = 12 * 1024 * 1024;

export function saveSyncedEvidence(root, input) {
  const relativePath = String(input?.relativePath || '').replaceAll('\\', '/');
  if (!/^[A-Za-z0-9_.-]+\/(?:xai\/(?:inputs|gradcam|activations)\/[^/]+\.(?:png|jpe?g|webp)|xai\/manifest\.json)$/i.test(relativePath)) {
    throw new Error('only sample xAI data can be synced');
  }
  const encoded = String(input?.data || '');
  if (!encoded || !/^[A-Za-z0-9+/=\s]+$/.test(encoded)) throw new Error('synced xAI data is invalid');
  const buffer = Buffer.from(encoded, 'base64');
  if (!buffer.length || buffer.length > MAX_SYNC_BYTES) throw new Error('synced xAI data must be 12 MiB or smaller');
  const absoluteRoot = path.resolve(root);
  const destination = path.resolve(absoluteRoot, relativePath);
  if (!destination.startsWith(`${absoluteRoot}${path.sep}`)) throw new Error('synced evidence path is outside the runs directory');
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  const temporary = `${destination}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, buffer);
  fs.renameSync(temporary, destination);
  return { relativePath, size: buffer.length };
}

export const evidenceSyncLimits = { requestBytes: 17 * 1024 * 1024 };

