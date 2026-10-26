import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { datasetProfileFromJob, familyFromDatasetProfileJob, sanitizeDatasetProfile } from '../server/dataset-profiles.js';

test('dataset profile jobs map to all Trainer model families', () => {
  assert.equal(familyFromDatasetProfileJob('detect_dataset_profile'), 'detection');
  assert.equal(familyFromDatasetProfileJob('classifier_dataset_profile'), 'classification');
  assert.equal(familyFromDatasetProfileJob('video_dataset_profile'), 'video');
  assert.equal(familyFromDatasetProfileJob('vlm_dataset_profile'), 'vlm');
  assert.equal(familyFromDatasetProfileJob('detect_train'), null);
});

test('completed profile jobs publish a bounded visualization record', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'trainer-dataset-profile-'));
  fs.writeFileSync(path.join(root, 'dataset-profile.json'), JSON.stringify({
    schema: 'trainer-dataset-profile-v1', family: 'detection', format: 'coco', name: 'val2017',
    source: '/data/coco/annotations/instances_val2017.json', generatedAt: 123,
    summary: { samples: 5000, annotations: 36781, classes: 80, invalid: 0 },
    splits: [{ name: 'val2017', samples: 5000, annotations: 36781, classes: 80 }],
    classes: [{ id: 1, name: 'person', total: 11004, splits: { val2017: 11004 } }],
    boxSizes: { small: 16800, medium: 15000, large: 4981 }, extensions: {}, text: {},
  }));
  const profile = datasetProfileFromJob({ id: 'job-1', name: 'coco-profile', type: 'detect_dataset_profile',
    status: 'completed', artifacts: [root], agentId: 'cpu-vm' });
  assert.equal(profile.format, 'coco');
  assert.equal(profile.summary.annotations, 36781);
  assert.equal(profile.classes[0].name, 'person');
  assert.equal(profile.jobId, 'job-1');
  assert.equal(profile.agentId, 'cpu-vm');
  fs.rmSync(root, { recursive: true, force: true });
});

test('dataset profile sanitizer rejects unrecognized payloads and caps classes', () => {
  assert.equal(sanitizeDatasetProfile({ family: 'detection' }), null);
  const profile = sanitizeDatasetProfile({ schema: 'trainer-dataset-profile-v1', family: 'classification',
    classes: Array.from({ length: 1200 }, (_, index) => ({ id: index, name: `class-${index}`, total: index })),
  });
  assert.equal(profile.classes.length, 1000);
  assert.equal(profile.classes[0].id, '0');
});
