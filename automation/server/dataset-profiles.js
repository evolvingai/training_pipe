import fs from 'node:fs';
import path from 'node:path';

const FAMILY_TYPES = {
  detect_dataset_profile: 'detection',
  classifier_dataset_profile: 'classification',
  video_dataset_profile: 'video',
  vlm_dataset_profile: 'vlm',
};

const finite = (value) => Number.isFinite(Number(value)) ? Number(value) : 0;
const safeText = (value, maximum = 2000) => String(value == null ? '' : value).slice(0, maximum);

function numericObject(value, limit = 100) {
  const output = {};
  for (const [key, item] of Object.entries(value || {}).slice(0, limit)) {
    output[safeText(key, 100)] = finite(item);
  }
  return output;
}

export function familyFromDatasetProfileJob(type) {
  return FAMILY_TYPES[String(type)] || null;
}

export function sanitizeDatasetProfile(value) {
  if (!value || typeof value !== 'object' || value.schema !== 'trainer-dataset-profile-v1') return null;
  if (!Object.values(FAMILY_TYPES).includes(value.family)) return null;
  const summary = numericObject(value.summary, 50);
  const splits = Array.isArray(value.splits) ? value.splits.slice(0, 20).map((split) => ({
    name: safeText(split?.name, 100), samples: finite(split?.samples), images: finite(split?.images),
    annotations: finite(split?.annotations), labeled: finite(split?.labeled), unlabeled: finite(split?.unlabeled),
    classes: finite(split?.classes), invalid: finite(split?.invalid), crowd: finite(split?.crowd),
  })) : [];
  const classes = Array.isArray(value.classes) ? value.classes.slice(0, 1000).map((item) => ({
    id: safeText(item?.id, 100), name: safeText(item?.name, 300), total: finite(item?.total),
    splits: numericObject(item?.splits, 20),
  })) : [];
  return {
    schema: value.schema, family: value.family, format: safeText(value.format, 100),
    name: safeText(value.name, 300), source: safeText(value.source, 2000),
    generatedAt: finite(value.generatedAt), summary, splits, classes,
    boxSizes: numericObject(value.boxSizes, 10), extensions: numericObject(value.extensions, 50),
    text: numericObject(value.text, 50), jobId: safeText(value.jobId, 200),
    jobName: safeText(value.jobName, 300), agentId: safeText(value.agentId, 300),
  };
}

export function datasetProfileFromJob(job) {
  const family = familyFromDatasetProfileJob(job?.type);
  if (!family || job?.status !== 'completed') return null;
  for (const artifact of job.artifacts || []) {
    const file = path.join(path.resolve(String(artifact)), 'dataset-profile.json');
    try {
      const payload = JSON.parse(fs.readFileSync(file, 'utf8'));
      const profile = sanitizeDatasetProfile({ ...payload, family, jobId: job.id, jobName: job.name,
        agentId: job.agentId || job.targetAgentId });
      if (profile) return profile;
    } catch {
      // Missing and malformed profiles are not published to the UI.
    }
  }
  return null;
}
