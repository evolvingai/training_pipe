import fs from 'node:fs';
import path from 'node:path';

const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp', '.gif']);
const DOCUMENT_EXTENSIONS = new Set(['.html', '.json', '.csv', '.toml', '.txt']);
const MODEL_EXTENSIONS = new Set(['.pt', '.pth', '.onnx', '.engine', '.safetensors']);
const MAX_SCAN_FILES = 2_000;

function familyFrom(relativePath) {
  const value = relativePath.toLowerCase();
  if (/(vlm|adapter|lora)/.test(value)) return 'vlm';
  if (/(video|action|clip)/.test(value)) return 'video';
  if (/(classifier|classification|classify)/.test(value)) return 'classification';
  if (/(detect|detection|boxes|yolo)/.test(value) || value.includes('/weights/')) return 'detection';
  return 'unknown';
}

function kindFrom(relativePath) {
  const value = relativePath.toLowerCase();
  const extension = path.extname(value);
  if (/mask|background|foreground/.test(value) && IMAGE_EXTENSIONS.has(extension)) return 'mask-diagnostic';
  if (/grad.?cam|heatmap|saliency|attention|activation|xai/.test(value) && IMAGE_EXTENSIONS.has(extension)) return 'xai';
  if (/augment|mosaic|mixup|copy.?paste/.test(value) && IMAGE_EXTENSIONS.has(extension)) return 'augmentation';
  if (IMAGE_EXTENSIONS.has(extension)) return 'visual';
  if (/report/.test(value) && extension === '.html') return 'report';
  if (/metric|threshold|confusion|history/.test(value) && DOCUMENT_EXTENSIONS.has(extension)) return 'metrics';
  if (/prediction|result/.test(value) && DOCUMENT_EXTENSIONS.has(extension)) return 'predictions';
  if (MODEL_EXTENSIONS.has(extension) || value.endsWith('/adapter_config.json')) return 'model';
  if (DOCUMENT_EXTENSIONS.has(extension)) return 'document';
  return null;
}

export function scanEvidence(root, { family = '', kind = '', query = '', limit = 200 } = {}) {
  const absoluteRoot = path.resolve(root);
  if (!fs.existsSync(absoluteRoot)) return { root: absoluteRoot, items: [], scanned: 0 };
  const stack = [{ directory: absoluteRoot, depth: 0 }];
  const items = [];
  let scanned = 0;
  const needle = String(query || '').trim().toLowerCase();
  const maximum = Math.min(Math.max(Number(limit) || 200, 1), 500);
  while (stack.length && scanned < MAX_SCAN_FILES) {
    const { directory, depth } = stack.pop();
    let entries;
    try { entries = fs.readdirSync(directory, { withFileTypes: true }); } catch { continue; }
    for (const entry of entries) {
      if (entry.name.startsWith('.')) continue;
      const full = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        if (depth < 6) stack.push({ directory: full, depth: depth + 1 });
        continue;
      }
      if (!entry.isFile()) continue;
      scanned += 1;
      const relativePath = path.relative(absoluteRoot, full).split(path.sep).join('/');
      const itemKind = kindFrom(relativePath);
      if (!itemKind) continue;
      const itemFamily = familyFrom(relativePath);
      if (family && family !== 'all' && itemFamily !== family && itemFamily !== 'unknown') continue;
      if (kind && kind !== 'all' && itemKind !== kind) continue;
      if (needle && !relativePath.toLowerCase().includes(needle)) continue;
      let stat;
      try { stat = fs.statSync(full); } catch { continue; }
      items.push({
        relativePath,
        name: entry.name,
        family: itemFamily,
        kind: itemKind,
        size: stat.size,
        modifiedAt: stat.mtimeMs,
        previewable: IMAGE_EXTENSIONS.has(path.extname(entry.name).toLowerCase()) || entry.name.endsWith('.html'),
        url: `/api/evidence/file?path=${encodeURIComponent(relativePath)}`,
      });
    }
  }
  items.sort((left, right) => right.modifiedAt - left.modifiedAt || left.relativePath.localeCompare(right.relativePath));
  return { root: absoluteRoot, items: items.slice(0, maximum), scanned };
}

export function resolveEvidenceFile(root, relativePath) {
  const absoluteRoot = fs.realpathSync(path.resolve(root));
  const candidate = path.resolve(absoluteRoot, String(relativePath || ''));
  if (!candidate.startsWith(`${absoluteRoot}${path.sep}`)) throw new Error('evidence path is outside the runs directory');
  const real = fs.realpathSync(candidate);
  if (!real.startsWith(`${absoluteRoot}${path.sep}`)) throw new Error('evidence symlink is outside the runs directory');
  const stat = fs.statSync(real);
  if (!stat.isFile()) throw new Error('evidence path is not a file');
  return { path: real, stat };
}

export function evidenceContentType(filename) {
  const extension = path.extname(filename).toLowerCase();
  return {
    '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp', '.gif': 'image/gif',
    '.html': 'text/html; charset=utf-8', '.json': 'application/json; charset=utf-8', '.csv': 'text/csv; charset=utf-8',
    '.toml': 'text/plain; charset=utf-8', '.txt': 'text/plain; charset=utf-8', '.onnx': 'application/octet-stream',
    '.pt': 'application/octet-stream', '.pth': 'application/octet-stream', '.engine': 'application/octet-stream',
    '.safetensors': 'application/octet-stream',
  }[extension] || 'application/octet-stream';
}

