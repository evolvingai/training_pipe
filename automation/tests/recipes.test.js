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

test('CUDA jobs preserve explicit device and larger defaults', () => {
  const recipe = buildRecipe('detect_train', {
    name: 'gpu', data: '/data/dataset.toml', model: 'trainer-s', device: 'cuda:1',
  }, defaults);
  assert.ok(recipe.commands[0].argv.includes('device=cuda:1'));
  assert.ok(recipe.commands[0].argv.includes('batch=16'));
  assert.ok(recipe.commands[0].argv.includes('imgsz=640'));
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
});

test('detector export uses a deterministic artifact directory', () => {
  const recipe = buildRecipe('detect_export', {
    name: 'deployment', checkpoint: '/runs/train/weights/best.pt', format: 'onnx',
  }, defaults);
  assert.ok(recipe.commands[0].argv.includes('out=/runs/deployment/model'));
  assert.deepEqual(recipe.artifacts, ['/runs/deployment']);
});

test('invalid devices and model aliases are rejected', () => {
  assert.throws(() => buildRecipe('detect_train', { name: 'x', data: '/d', model: 'yolo', device: 'cpu' }, defaults), /model/);
  assert.throws(() => buildRecipe('detect_train', { name: 'x', data: '/d', model: 'trainer-n', device: 'cuda; reboot' }, defaults), /device/);
  assert.throws(() => buildRecipe('vlm_tune', { name: 'x', data: '/d.jsonl', model: 'org/model', dtype: 'int8' }, defaults), /dtype/);
});
