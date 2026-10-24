import fs from 'node:fs';
import path from 'node:path';

const IMAGE_TYPES = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
};

function readManifest(runDirectory) {
  try {
    const value = JSON.parse(fs.readFileSync(path.join(runDirectory, 'xai', 'manifest.json'), 'utf8'));
    return value?.format === 'trainer-classification-xai-v1' && Array.isArray(value.images) ? value : null;
  } catch {
    return null;
  }
}

function safeRunDirectory(root, runId) {
  if (!/^[A-Za-z0-9_.-]+$/.test(String(runId || ''))) throw new Error('invalid xAI run');
  const absoluteRoot = fs.realpathSync(path.resolve(root));
  const directory = fs.realpathSync(path.join(absoluteRoot, runId));
  if (path.dirname(directory) !== absoluteRoot) throw new Error('xAI run is outside the runs directory');
  return directory;
}

function sampleResult(runId, item, index) {
  const base = `/api/xai/classification/${encodeURIComponent(runId)}/samples/${index}`;
  return {
    id: index,
    label: `Sample ${index + 1}`,
    predicted: String(item.predicted || ''),
    confidence: Number(item.confidence || 0),
    target: String(item.target || item.predicted || ''),
    targetScore: Number(item.target_score || 0),
    layer: String(item.layer || ''),
    topK: (item.topk || []).map((score) => ({
      class: String(score.class || ''), confidence: Number(score.confidence || 0),
    })),
    inputUrl: `${base}/input`,
    signedMapUrl: `${base}/signed-map`,
    activationsUrl: `${base}/activations`,
  };
}

export function listClassificationXaiResults(root, { runId = '', limit = 100 } = {}) {
  const absoluteRoot = path.resolve(root);
  if (!fs.existsSync(absoluteRoot)) return [];
  let entries;
  try { entries = fs.readdirSync(absoluteRoot, { withFileTypes: true }); } catch { return []; }
  return entries
    .filter((entry) => entry.isDirectory() && !entry.name.startsWith('.') && (!runId || entry.name === runId))
    .flatMap((entry) => {
      const directory = path.join(absoluteRoot, entry.name);
      const manifest = readManifest(directory);
      if (!manifest?.images?.length) return [];
      let updatedAt = 0;
      try { updatedAt = fs.statSync(path.join(directory, 'xai', 'manifest.json')).mtimeMs; } catch { /* ignore */ }
      return [{
        id: entry.name,
        name: entry.name,
        architecture: String(manifest.architecture || 'classification model'),
        createdAt: String(manifest.created_at || ''),
        updatedAt,
        samples: manifest.images.map((item, index) => sampleResult(entry.name, item, index)),
      }];
    })
    .sort((left, right) => right.updatedAt - left.updatedAt)
    .slice(0, Math.min(Math.max(Number(limit) || 100, 1), 200));
}

export function resolveClassificationXaiAsset(root, runId, sampleIndex, asset) {
  const directory = safeRunDirectory(root, runId);
  const manifest = readManifest(directory);
  const item = manifest?.images?.[Number(sampleIndex)];
  if (!item) throw new Error('xAI sample not found');
  const key = { input: 'input', 'signed-map': 'signed_map', activations: 'activations' }[asset];
  if (!key || !item[key]) throw new Error('xAI asset not found');
  const candidate = path.resolve(directory, String(item[key]));
  if (!candidate.startsWith(`${directory}${path.sep}`)) throw new Error('xAI asset is outside the run directory');
  const resolved = fs.realpathSync(candidate);
  if (!resolved.startsWith(`${directory}${path.sep}`)) throw new Error('xAI asset symlink is outside the run directory');
  const extension = path.extname(resolved).toLowerCase();
  if (!IMAGE_TYPES[extension]) throw new Error('unsupported xAI asset type');
  const stat = fs.statSync(resolved);
  if (!stat.isFile()) throw new Error('xAI asset is not a file');
  return { path: resolved, stat, contentType: IMAGE_TYPES[extension] };
}
