import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { evaluationFromJob, familyFromJobType, sanitizeEvaluationRecord } from '../server/evaluations.js';

function fixture(type, metrics, relative = 'metrics.json') {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'trainer-evaluation-'));
  const file = path.join(root, relative);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(metrics));
  return { id: `${type}-1`, type, name: `${type}-run`, status: 'completed', endedAt: 123,
    agentId: 'cpu-vm', params: { data: '/data/validation', checkpoint: '/runs/model/best.pt', model: 'org/vlm' },
    artifacts: [root] };
}

test('evaluation records are emitted only for completed evaluate jobs', () => {
  assert.equal(familyFromJobType('classifier_evaluate'), 'classification');
  assert.equal(familyFromJobType('classifier_train'), null);
  assert.equal(evaluationFromJob({ type: 'classifier_evaluate', status: 'failed' }), null);
});

test('classification and video evaluation metrics become structured UI records', () => {
  const classifier = evaluationFromJob(fixture('classifier_evaluate', {
    loss: .4, accuracy: .8, macro_precision: .81, macro_recall: .79, macro_f1: .80, samples: 20,
    per_class: { cat: { precision: .9, recall: .8, f1: .85, support: 10 } }, confusion_matrix: [[8, 2], [2, 8]],
  }));
  assert.equal(classifier.primary.key, 'macroF1');
  assert.equal(classifier.metrics.accuracy, .8);
  assert.equal(classifier.classes[0].name, 'cat');
  assert.deepEqual(classifier.confusionMatrix[0], [8, 2]);

  const video = evaluationFromJob(fixture('video_evaluate', {
    loss: .2, accuracy: .9, samples: 12, per_class: { swim: { accuracy: .75, support: 4 } },
  }));
  assert.equal(video.primary.key, 'accuracy');
  assert.equal(video.classes[0].accuracy, .75);
});

test('COCO multi-label classification uses mAP and exposes AP and micro F1', () => {
  const classifier = evaluationFromJob(fixture('classifier_evaluate', {
    classification_mode: 'multi-label', loss: .15, accuracy: .01, micro_f1: .42, macro_f1: .18, map: .31,
    samples: 1000, per_class: { person: { precision: .7, recall: .6, f1: .65, ap: .72, support: 540 } },
  }));
  assert.equal(classifier.primary.key, 'map');
  assert.equal(classifier.primary.label, 'mAP');
  assert.equal(classifier.primary.value, .31);
  assert.equal(classifier.metrics.microF1, .42);
  assert.equal(classifier.classes[0].ap, .72);
  assert.deepEqual(classifier.confusionMatrix, []);
});

test('detection and VLM evaluation metrics have family-specific primary scores', () => {
  const detector = evaluationFromJob(fixture('detect_evaluate', {
    global: { mAP50: .7, 'mAP50-95': .5, precision: .8, recall: .6 },
    per_class: { fish: { precision: .8, recall: .6, AP50: .7, 'AP50-95': .5 } },
  }, 'logs/val_results.json'));
  assert.equal(detector.primary.label, 'mAP50–95');
  assert.equal(detector.classes[0].map5095, .5);

  const vlm = evaluationFromJob(fixture('vlm_evaluate', {
    model: 'org/vlm', samples: 5, loss: 1.2, perplexity: 3.32, token_accuracy: .62, evaluated_tokens: 104,
  }));
  assert.equal(vlm.primary.key, 'tokenAccuracy');
  assert.equal(vlm.metrics.evaluatedTokens, 104);
});

test('remote evaluation payloads are bounded and sanitized', () => {
  const record = sanitizeEvaluationRecord({
    id: 'eval', jobId: 'job', jobName: 'run', family: 'classification', dataset: { key: 'data::val', split: 'val' },
    checkpoint: { label: 'best.pt' }, primary: { key: 'macroF1', label: 'Macro F1', value: .9 },
    metrics: { macroF1: .9 }, classes: [{ name: 'cat', recall: .8 }], confusionMatrix: [[4]],
  });
  assert.equal(record.metrics.macroF1, .9);
  assert.equal(record.classes[0].name, 'cat');
  assert.deepEqual(record.confusionMatrix, [[4]]);
  assert.equal(sanitizeEvaluationRecord({ family: 'unknown' }), null);
});
