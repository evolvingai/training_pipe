import fs from 'node:fs';
import path from 'node:path';

const MAX_FILES_PER_RUN = 2_000;

function json(file) {
  try { return JSON.parse(fs.readFileSync(file, 'utf8')); } catch { return null; }
}

function runStats(directory) {
  const stack = [{ directory, depth: 0 }];
  let files = 0;
  let bytes = 0;
  let updatedAt = fs.statSync(directory).mtimeMs;
  while (stack.length && files < MAX_FILES_PER_RUN) {
    const current = stack.pop();
    let entries;
    try { entries = fs.readdirSync(current.directory, { withFileTypes: true }); } catch { continue; }
    for (const entry of entries) {
      if (entry.name.startsWith('.')) continue;
      const full = path.join(current.directory, entry.name);
      if (entry.isDirectory()) {
        if (current.depth < 5) stack.push({ directory: full, depth: current.depth + 1 });
        continue;
      }
      if (!entry.isFile()) continue;
      files += 1;
      try {
        const stat = fs.statSync(full);
        bytes += stat.size;
        updatedAt = Math.max(updatedAt, stat.mtimeMs);
      } catch { /* file disappeared during inventory */ }
    }
  }
  return { files, bytes, updatedAt };
}

function normalizedFamily(value) {
  const key = String(value || '').toLowerCase().replaceAll('_', '-');
  if (['detect', 'detector', 'detection'].includes(key)) return 'detection';
  if (['classification', 'image', 'image-classification'].includes(key)) return 'classification';
  if (['video', 'action', 'action-recognition', 'video-classification'].includes(key)) return 'video';
  if (['vlm', 'vision-language'].includes(key)) return 'vlm';
  return '';
}

function familyFromText(value) {
  const text = String(value || '').toLowerCase();
  if (/(vlm|adapter|lora)/.test(text)) return 'vlm';
  if (/(video|action|clip|r2plus1d)/.test(text)) return 'video';
  if (/(classifier|classification|classify|convnext)/.test(text)) return 'classification';
  if (/(detect|detector|yolo|trainer-[nsmxl]|boxes)/.test(text)) return 'detection';
  return '';
}

function smokeMatrixRuns(directory, relativePath, summary) {
  if (summary?.format !== 'trainer-vm-smoke-matrix-v1') return [];
  return Object.entries(summary.families || {}).map(([rawFamily, detail]) => {
    const family = normalizedFamily(rawFamily) || familyFromText(rawFamily) || 'unknown';
    const familyPath = path.join(directory, rawFamily);
    const stats = fs.existsSync(familyPath) ? runStats(familyPath) : runStats(directory);
    return {
      id: `discovered:${relativePath}:${rawFamily}`,
      name: `${path.basename(directory)} · ${rawFamily}`,
      family,
      action: 'smoke',
      status: detail?.status === 'passed' ? 'completed' : detail?.status === 'failed' ? 'failed' : 'completed',
      path: fs.existsSync(familyPath) ? familyPath : directory,
      relativePath: `${relativePath}/${rawFamily}`,
      architecture: detail?.architecture || detail?.arch || '',
      device: summary.device || '',
      files: stats.files,
      bytes: stats.bytes,
      updatedAt: stats.updatedAt,
      seconds: Number(detail?.seconds || 0),
      error: detail?.error || null,
      source: 'discovered',
      hasEvidence: false,
    };
  });
}

function inspectRun(root, directory) {
  const relativePath = path.relative(root, directory).split(path.sep).join('/');
  const name = path.basename(directory);
  const args = json(path.join(directory, 'cfg', 'args.json'));
  const manifest = json(path.join(directory, 'xai', 'manifest.json'));
  const metrics = json(path.join(directory, 'metrics.json'));
  const summary = json(path.join(directory, 'summary.json'));
  const matrix = smokeMatrixRuns(directory, relativePath, summary);
  if (matrix.length) return matrix;
  const contractNames = ['model.json', 'contract.json', 'export.json'];
  const contract = contractNames.map((item) => json(path.join(directory, item))).find(Boolean) || null;
  const has = (relative) => fs.existsSync(path.join(directory, relative));
  const hasCheckpoint = has('best.pt') || has('last.pt') || has('weights/best.pt') || has('weights/last.pt');
  const hasAdapter = has('adapter/adapter_config.json') || has('adapter_config.json');
  const hasExport = has('model.onnx') || has('model.engine') || Boolean(contract);
  const hasPrediction = has('predictions.csv') || has('xai/predictions.csv');
  if (!args && !manifest && !metrics && !summary && !hasCheckpoint && !hasAdapter && !hasExport && !hasPrediction) return [];

  const family = normalizedFamily(args?.task)
    || normalizedFamily(metrics?.task)
    || familyFromText(manifest?.format)
    || familyFromText(contract?.format)
    || (args ? 'classification' : '')
    || (hasAdapter ? 'vlm' : '')
    || (has('cfg/args.yaml') || has('reports/report.md') ? 'detection' : '')
    || (has('weights/best.pt') || has('weights/last.pt') ? 'classification' : '')
    || ((has('best.pt') || has('last.pt')) && metrics?.best_accuracy != null ? 'video' : '')
    || familyFromText(name)
    || 'unknown';
  const action = manifest ? 'xai'
    : String(args?.command || '')
      || (hasExport ? 'export' : hasAdapter ? 'tune' : hasCheckpoint ? 'train' : hasPrediction ? 'predict' : 'evaluate');
  const stats = runStats(directory);
  return [{
    id: `discovered:${relativePath}`,
    name,
    family,
    action,
    status: 'completed',
    path: directory,
    relativePath,
    architecture: args?.arch || manifest?.architecture || contract?.architecture || metrics?.architecture || '',
    device: args?.device || metrics?.device || '',
    files: stats.files,
    bytes: stats.bytes,
    updatedAt: stats.updatedAt,
    seconds: 0,
    error: null,
    source: 'discovered',
    hasEvidence: Boolean(manifest || has('reports/report.html') || has('reports/report.md')),
  }];
}

export function discoverRuns(root, { limit = 500 } = {}) {
  const absoluteRoot = path.resolve(root);
  if (!fs.existsSync(absoluteRoot)) return [];
  let entries;
  try { entries = fs.readdirSync(absoluteRoot, { withFileTypes: true }); } catch { return []; }
  return entries
    .filter((entry) => entry.isDirectory() && !entry.name.startsWith('.'))
    .flatMap((entry) => inspectRun(absoluteRoot, path.join(absoluteRoot, entry.name)))
    .sort((left, right) => Number(right.updatedAt || 0) - Number(left.updatedAt || 0))
    .slice(0, Math.min(Math.max(Number(limit) || 500, 1), 1_000));
}
