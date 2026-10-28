import assert from 'node:assert/strict';
import test from 'node:test';

import { buildRecipe } from '../server/recipes.js';

const defaults = { codeDir: '/opt/trainer', pythonBin: '/venv/bin/python', runsDir: '/runs', device: 'cpu' };

test('CPU detector recipe produces an exact, shell-free command', () => {
  const recipe = buildRecipe('detect_train', {
    name: 'cpu smoke; touch /tmp/nope', data: '/data/dataset.toml', model: 'trainer-n', epochs: 2,
  }, defaults);
  assert.equal(recipe.name, 'cpu_smoke__touch__tmp_nope');
  assert.deepEqual(recipe.commands[0].argv.slice(0, 4), ['/venv/bin/python', '-m', 'trainer', 'train']);
  assert.ok(recipe.commands[0].argv.includes('device=cpu'));
  assert.ok(recipe.commands[0].argv.includes('batch=2'));
  assert.equal(recipe.commands[0].argv.some((part) => part === ';'), false);
});

test('all model families can profile COCO or custom datasets on a worker', () => {
  const cases = [
    ['detect_dataset_profile', 'detection', '/data/coco/annotations/instances_val2017.json'],
    ['classifier_dataset_profile', 'classification', '/data/species'],
    ['video_dataset_profile', 'video', '/data/actions'],
    ['vlm_dataset_profile', 'vlm', '/data/vlm/validation.jsonl'],
  ];
  for (const [type, family, data] of cases) {
    const recipe = buildRecipe(type, { name: `${family}-profile`, data }, defaults);
    assert.deepEqual(recipe.commands[0].argv.slice(0, 2), ['/venv/bin/python', '/opt/trainer/automation/dataset_profile.py']);
    assert.ok(recipe.commands[0].argv.includes('--family'));
    assert.ok(recipe.commands[0].argv.includes(family));
    assert.ok(recipe.commands[0].argv.includes(data));
    assert.ok(recipe.commands[0].argv.includes(`/runs/${family}-profile/dataset-profile.json`));
    assert.deepEqual(recipe.artifacts, [`/runs/${family}-profile`]);
  }
});

test('CUDA jobs preserve explicit device and larger defaults', () => {
  const recipe = buildRecipe('detect_train', {
    name: 'gpu', data: '/data/dataset.toml', model: 'trainer-s', device: 'cuda:1',
  }, defaults);
  assert.ok(recipe.commands[0].argv.includes('device=cuda:1'));
  assert.ok(recipe.commands[0].argv.includes('batch=16'));
  assert.ok(recipe.commands[0].argv.includes('imgsz=640'));
});

test('latest detector architectures and experiment controls are passed through', () => {
  const recipe = buildRecipe('detect_train', {
    name: 'p2-distill', data: '/data/dataset.toml', model: 'trainer-p2-n', device: 'cpu',
    optimizer: 'musgd', resume: '/runs/base/weights/last.pt', resumeMode: 'branch',
    regMax: 12, regChannels: 64, assignmentMetric: 'ciou',
    teacher: '/runs/teacher/weights/best.pt', distillFeatureWeight: 1,
    distillClsWeight: 0.5, distillBoxWeight: 0.25, distillTemperature: 3,
    muonWeight: 0.15, sgdWeight: 0.9, muonBeta: 0.8, muonNsSteps: 4,
    muonSchedule: 'linear', muonTaperStart: 2, muonTaperEnd: 4,
    muonBatchNs: true, profile: true, lossDiagnostics: true,
  }, defaults);
  const argv = recipe.commands[0].argv;
  for (const value of [
    'model=trainer-p2-n', 'optimizer=musgd', 'resume=/runs/base/weights/last.pt',
    'resume_mode=branch', 'reg_max=12', 'reg_channels=64', 'assignment_metric=ciou',
    'teacher=/runs/teacher/weights/best.pt', 'distill_feature_weight=1',
    'distill_cls_weight=0.5', 'distill_box_weight=0.25', 'distill_temperature=3',
    'muon_weight=0.15', 'sgd_weight=0.9', 'muon_beta=0.8', 'muon_ns_steps=4',
    'muon_schedule=linear', 'muon_taper_start=2', 'muon_taper_end=4',
    'muon_batch_ns=true', 'profile=true', 'loss_diagnostics=true',
  ]) assert.ok(argv.includes(value), value);

  const legacy = buildRecipe('detect_train', {
    name: 'legacy', data: '/data/dataset.toml', model: 'trainer-legacy-s', device: 'cpu',
  }, defaults);
  assert.ok(legacy.commands[0].argv.includes('model=trainer-legacy-s'));
});

test('detector evaluation can select raw weights and multi-label NMS', () => {
  const recipe = buildRecipe('detect_evaluate', {
    name: 'raw-eval', data: '/data/dataset.toml', checkpoint: '/runs/train/weights/best.pt',
    evalWeights: 'raw', nmsMultiLabel: true,
  }, defaults);
  assert.ok(recipe.commands[0].argv.includes('eval_weights=raw'));
  assert.ok(recipe.commands[0].argv.includes('nms_multi_label=true'));
});

test('classifier recipes expose the detector-parity lifecycle', () => {
  const recipe = buildRecipe('classifier_train', {
    name: 'species', data: '/data/species', augmentationConfig: '/data/robust.toml',
    mosaic: 0.5, mixup: 0.2, copyPaste: 0.1, copyPasteIou: '0.1,0.55', blur: 0.3,
  }, defaults);
  assert.deepEqual(recipe.commands[0].argv.slice(0, 4), ['/venv/bin/python', '-m', 'trainer', 'train']);
  assert.ok(recipe.commands[0].argv.includes('task=classification'));
  assert.ok(recipe.commands[0].argv.includes('arch=convnext-tiny'));
  assert.ok(recipe.commands[0].argv.includes('project=/runs'));
  assert.ok(recipe.commands[0].argv.includes('augmentation_config=/data/robust.toml'));
  assert.ok(recipe.commands[0].argv.includes('mosaic=0.5'));
  assert.ok(recipe.commands[0].argv.includes('mixup=0.2'));
  assert.ok(recipe.commands[0].argv.includes('copy_paste=0.1'));
  assert.ok(recipe.commands[0].argv.includes('copy_paste_iou=0.1,0.55'));
  assert.ok(recipe.commands[0].argv.includes('blur=0.3'));

  const finetune = buildRecipe('classifier_finetune', {
    name: 'species-ft', data: '/data/species', checkpoint: '/runs/species/weights/best.pt', freeze: 2,
  }, defaults);
  assert.ok(finetune.commands[0].argv.includes('finetune'));
  assert.ok(finetune.commands[0].argv.includes('freeze=2'));

  const evaluation = buildRecipe('classifier_evaluate', {
    name: 'species-eval', data: '/data/species', checkpoint: '/runs/species/weights/best.pt',
  }, defaults);
  assert.deepEqual(evaluation.commands.map((item) => item.id), ['evaluate', 'tune-thresh', 'report']);
  assert.ok(evaluation.commands[2].argv.includes('thresholds=/runs/species-eval/thresholds.json'));

  const tune = buildRecipe('classifier_tune', { name: 'species-tune', data: '/data/species' }, defaults);
  assert.ok(tune.commands[0].argv.includes('tune'));
  assert.ok(tune.commands[0].argv.includes('iterations=10'));

  const xai = buildRecipe('classifier_xai', {
    name: 'species-xai', source: '/data/species/val', checkpoint: '/runs/species/weights/best.pt',
    target: 'salmon', maxImages: 12, activationChannels: 4,
  }, defaults);
  assert.ok(xai.commands[0].argv.includes('xai'));
  assert.ok(xai.commands[0].argv.includes('task=classification'));
  assert.ok(xai.commands[0].argv.includes('target=salmon'));
  assert.ok(xai.commands[0].argv.includes('max_images=12'));
  assert.ok(xai.commands[0].argv.includes('activation_channels=4'));
  assert.deepEqual(xai.artifacts, ['/runs/species-xai']);

  const uploadedSample = buildRecipe('classifier_xai', {
    name: 'one-sample', source: 'sample://0123456789abcdef',
    checkpoint: '/runs/species/weights/best.pt', maxImages: 1,
  }, defaults);
  assert.ok(uploadedSample.commands[0].argv.includes('source=sample://0123456789abcdef'));
  assert.ok(uploadedSample.commands[0].argv.includes('max_images=1'));
});

test('video/action recipes expose clip sampling and a CPU-safe default', () => {
  const recipe = buildRecipe('video_train', {
    name: 'actions', data: '/data/actions', augmentationConfig: '/data/video-augment.toml', fliplr: 0.7,
  }, defaults);
  assert.deepEqual(recipe.commands[0].argv.slice(0, 4), ['/venv/bin/python', '-m', 'trainer', 'train']);
  assert.ok(recipe.commands[0].argv.includes('task=video-classification'));
  assert.ok(recipe.commands[0].argv.includes('arch=r2plus1d-tiny'));
  assert.ok(recipe.commands[0].argv.includes('clip_len=8'));
  assert.ok(recipe.commands[0].argv.includes('augmentation_config=/data/video-augment.toml'));
  assert.ok(recipe.commands[0].argv.includes('fliplr=0.7'));
  assert.deepEqual(recipe.artifacts, ['/runs/actions']);
});

test('VLM tuning and merge recipes create portable adapter commands', () => {
  const tune = buildRecipe('vlm_tune', {
    name: 'caption_lora', data: '/data/vlm/train.jsonl', model: 'org/small-vlm', device: 'cpu',
  }, defaults);
  assert.deepEqual(tune.commands[0].argv.slice(0, 4), ['/venv/bin/python', '-m', 'trainer', 'tune']);
  assert.ok(tune.commands[0].argv.includes('task=vlm'));
  assert.ok(tune.commands[0].argv.includes('arch=hf-lora'));
  assert.ok(tune.commands[0].argv.some((item) => item.startsWith('lora_r=')));
  assert.ok(tune.artifacts.includes('/runs/caption_lora/adapter'));

  const merge = buildRecipe('vlm_merge', {
    name: 'merged', model: 'org/small-vlm', adapter: '/runs/caption_lora/adapter',
  }, defaults);
  assert.ok(merge.commands[0].argv.includes('merge'));
  assert.ok(merge.commands[0].argv.includes('adapter=/runs/caption_lora/adapter'));

  const evaluation = buildRecipe('vlm_evaluate', {
    name: 'vlm-eval', data: '/data/vlm/validation.jsonl', model: 'org/small-vlm',
    adapter: '/runs/caption_lora/adapter', maxSamples: 12,
  }, defaults);
  assert.ok(evaluation.commands[0].argv.includes('evaluate'));
  assert.ok(evaluation.commands[0].argv.includes('task=vlm'));
  assert.ok(evaluation.commands[0].argv.includes('max_samples=12'));
  assert.deepEqual(evaluation.artifacts, ['/runs/vlm-eval']);
});

test('detector evaluation writes validation metrics into its evaluation run', () => {
  const recipe = buildRecipe('detect_evaluate', {
    name: 'detector-eval', data: '/data/dataset.toml', checkpoint: '/runs/train/weights/best.pt',
  }, defaults);
  assert.ok(recipe.commands[0].argv.includes('project=/runs'));
  assert.ok(recipe.commands[0].argv.includes('name=detector-eval'));
  assert.ok(recipe.commands[0].argv.includes('exist_ok=true'));
});

test('detector export uses a deterministic artifact directory', () => {
  const recipe = buildRecipe('detect_export', {
    name: 'deployment', checkpoint: '/runs/train/weights/best.pt', format: 'onnx',
  }, defaults);
  assert.ok(recipe.commands[0].argv.includes('out=/runs/deployment/model'));
  assert.deepEqual(recipe.artifacts, ['/runs/deployment']);
});

test('every model family exposes ONNX and TensorRT export recipes', () => {
  const detector = buildRecipe('detect_export', {
    name: 'detector-trt', checkpoint: '/runs/detector/weights/best.pt', format: 'tensorrt', device: 'cuda:0',
  }, defaults);
  assert.ok(detector.commands[0].argv.includes('format=tensorrt'));

  const classifier = buildRecipe('classifier_export', {
    name: 'classifier-onnx', checkpoint: '/runs/classifier/weights/best.pt', format: 'onnx',
  }, defaults);
  assert.ok(classifier.commands[0].argv.includes('task=classification'));
  assert.ok(classifier.commands[0].argv.includes('format=onnx'));

  const video = buildRecipe('video_export', {
    name: 'video-trt', checkpoint: '/runs/video/best.pt', format: 'tensorrt', device: 'cuda:0', half: true,
  }, defaults);
  assert.ok(video.commands[0].argv.includes('task=video-classification'));
  assert.ok(video.commands[0].argv.includes('format=tensorrt'));
  assert.ok(video.commands[0].argv.includes('half=true'));

  const vlm = buildRecipe('vlm_export', {
    name: 'vlm-trt', model: 'org/vlm', adapter: '/runs/vlm/adapter', format: 'tensorrt', device: 'cuda:0',
  }, defaults);
  assert.ok(vlm.commands[0].argv.includes('task=vlm'));
  assert.ok(vlm.commands[0].argv.includes('export'));
  assert.ok(vlm.commands[0].argv.includes('format=tensorrt'));
  assert.ok(vlm.commands[0].argv.includes('opset=18'));
  assert.ok(vlm.commands[0].argv.includes('adapter=/runs/vlm/adapter'));
});

test('TensorRT export cannot be queued to a CPU-only device', () => {
  assert.throws(() => buildRecipe('classifier_export', {
    name: 'bad-trt', checkpoint: '/runs/model.pt', format: 'tensorrt', device: 'cpu',
  }, defaults), /requires a CUDA device/);
  assert.throws(() => buildRecipe('vlm_export', {
    name: 'bad-vlm-trt', model: 'org/vlm', format: 'tensorrt', device: 'cpu',
  }, defaults), /requires a CUDA device/);
});

test('invalid devices and model aliases are rejected', () => {
  assert.throws(() => buildRecipe('detect_train', { name: 'x', data: '/d', model: 'yolo', device: 'cpu' }, defaults), /model/);
  assert.throws(() => buildRecipe('detect_train', { name: 'x', data: '/d', model: 'trainer-n', device: 'cuda; reboot' }, defaults), /device/);
  assert.throws(() => buildRecipe('vlm_tune', { name: 'x', data: '/d.jsonl', model: 'org/model', dtype: 'int8' }, defaults), /dtype/);
});
