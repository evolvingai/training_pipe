import fs from 'node:fs';
import path from 'node:path';

const FAMILY_TYPES = {
  detect_evaluate: 'detection',
  classifier_evaluate: 'classification',
  video_evaluate: 'video',
  vlm_evaluate: 'vlm',
};

const finite = (value) => Number.isFinite(Number(value)) ? Number(value) : null;

function readJson(file) {
  try {
    const value = JSON.parse(fs.readFileSync(file, 'utf8'));
    return value && typeof value === 'object' && !Array.isArray(value) ? value : null;
  } catch {
    return null;
  }
}

function firstJson(artifacts, candidates) {
  for (const artifact of artifacts || []) {
    const root = path.resolve(String(artifact));
    for (const candidate of candidates) {
      const value = readJson(path.join(root, candidate));
      if (value) return value;
    }
  }
  return null;
}

function sourceLabel(value, fallback) {
  const source = String(value || '').trim();
  if (!source) return { label: fallback, source: '' };
  const base = path.basename(source);
  const parent = path.basename(path.dirname(source));
  return { label: parent && parent !== '.' ? `${parent}/${base}` : base, source };
}

function classRows(perClass = {}, family) {
  return Object.entries(perClass || {}).slice(0, 1000).map(([name, value]) => ({
    name: String(name),
    precision: finite(value?.precision),
    recall: finite(value?.recall),
    f1: finite(value?.f1),
    accuracy: finite(value?.accuracy),
    support: finite(value?.support),
    ap: finite(value?.ap),
    ap50: finite(value?.AP50 ?? value?.ap50),
    map5095: finite(value?.['AP50-95'] ?? value?.map5095),
    family,
  }));
}

function matrix(value) {
  if (!Array.isArray(value)) return [];
  return value.slice(0, 100).map((row) => Array.isArray(row)
    ? row.slice(0, 100).map((cell) => finite(cell) ?? 0)
    : []);
}

function safeText(value, maximum = 2000) {
  return String(value || '').slice(0, maximum);
}

export function familyFromJobType(type) {
  return FAMILY_TYPES[String(type)] || null;
}

export function evaluationFromJob(job) {
  const family = familyFromJobType(job?.type);
  if (!family || job?.status !== 'completed') return null;
  const params = job.params || {};
  const data = sourceLabel(params.data, 'evaluation dataset');
  const checkpointValue = params.checkpoint || params.adapter || params.model;
  const checkpoint = sourceLabel(checkpointValue, params.arch || 'model');
  const split = String(params.split || 'validation');
  const base = {
    id: String(job.id), jobId: String(job.id), jobName: String(job.name || job.id),
    family, architecture: String(params.arch || params.model || ''),
    endedAt: Number(job.endedAt || job.updatedAt || 0), agentId: String(job.agentId || job.targetAgentId || ''),
    dataset: { key: `${data.source || data.label}::${split}`, ...data, split },
    checkpoint,
  };

  if (family === 'detection') {
    const payload = firstJson(job.artifacts, ['logs/val_results.json', 'val_results.json']);
    if (!payload) return null;
    const metrics = payload.global || payload.metrics || payload;
    const classes = classRows(payload.per_class || payload.perClass, family);
    return { ...base, samples: finite(payload.samples ?? metrics.samples), primary: {
      key: 'map5095', label: 'mAP50–95', value: finite(metrics['mAP50-95'] ?? metrics.map5095),
    }, metrics: {
      map50: finite(metrics.mAP50 ?? metrics.map50), map5095: finite(metrics['mAP50-95'] ?? metrics.map5095),
      precision: finite(metrics.precision), recall: finite(metrics.recall), loss: finite(metrics.val_loss ?? metrics.loss),
    }, classes, confusionMatrix: matrix(payload.confusion_matrix) };
  }

  const payload = firstJson(job.artifacts, ['metrics.json']);
  if (!payload) return null;
  if (family === 'classification') {
    const classes = classRows(payload.per_class, family);
    const multiLabel = payload.classification_mode === 'multi-label';
    return { ...base, architecture: String(payload.arch || base.architecture), samples: finite(payload.samples),
      primary: { key: multiLabel ? 'map' : 'macroF1', label: multiLabel ? 'mAP' : 'Macro F1',
        value: finite(multiLabel ? payload.map : payload.macro_f1) },
      metrics: { loss: finite(payload.loss), accuracy: finite(payload.accuracy),
        macroPrecision: finite(payload.macro_precision), macroRecall: finite(payload.macro_recall),
        macroF1: finite(payload.macro_f1), microF1: finite(payload.micro_f1), map: finite(payload.map) },
      classes, confusionMatrix: matrix(payload.confusion_matrix) };
  }
  if (family === 'video') {
    const classes = classRows(payload.per_class, family);
    return { ...base, architecture: String(payload.arch || base.architecture), samples: finite(payload.samples),
      primary: { key: 'accuracy', label: 'Accuracy', value: finite(payload.accuracy) },
      metrics: { loss: finite(payload.loss), accuracy: finite(payload.accuracy) },
      classes, confusionMatrix: matrix(payload.confusion_matrix) };
  }
  const tokenAccuracy = finite(payload.token_accuracy);
  const loss = finite(payload.loss);
  return { ...base, architecture: String(payload.model || base.architecture), samples: finite(payload.samples),
    primary: { key: tokenAccuracy == null ? 'loss' : 'tokenAccuracy',
      label: tokenAccuracy == null ? 'Loss' : 'Token accuracy', value: tokenAccuracy ?? loss,
      direction: tokenAccuracy == null ? 'lower' : 'higher' },
    metrics: { loss, perplexity: finite(payload.perplexity), tokenAccuracy,
      evaluatedTokens: finite(payload.evaluated_tokens) }, classes: [], confusionMatrix: [] };
}

export function sanitizeEvaluationRecord(value) {
  if (!value || typeof value !== 'object' || !Object.values(FAMILY_TYPES).includes(value.family)) return null;
  const classes = Array.isArray(value.classes) ? value.classes.slice(0, 1000).map((item) => ({
    name: safeText(item?.name, 200), precision: finite(item?.precision), recall: finite(item?.recall),
    f1: finite(item?.f1), accuracy: finite(item?.accuracy), support: finite(item?.support), ap: finite(item?.ap),
    ap50: finite(item?.ap50), map5095: finite(item?.map5095), family: value.family,
  })) : [];
  const metrics = {};
  for (const [key, item] of Object.entries(value.metrics || {}).slice(0, 50)) metrics[safeText(key, 100)] = finite(item);
  return {
    id: safeText(value.id, 200), jobId: safeText(value.jobId, 200), jobName: safeText(value.jobName, 300),
    family: value.family, architecture: safeText(value.architecture, 500), endedAt: finite(value.endedAt),
    agentId: safeText(value.agentId, 300), samples: finite(value.samples),
    dataset: { key: safeText(value.dataset?.key), label: safeText(value.dataset?.label, 500),
      source: safeText(value.dataset?.source), split: safeText(value.dataset?.split, 100) },
    checkpoint: { label: safeText(value.checkpoint?.label, 500), source: safeText(value.checkpoint?.source) },
    primary: { key: safeText(value.primary?.key, 100), label: safeText(value.primary?.label, 100),
      value: finite(value.primary?.value), direction: value.primary?.direction === 'lower' ? 'lower' : 'higher' },
    metrics, classes, confusionMatrix: matrix(value.confusionMatrix),
  };
}
