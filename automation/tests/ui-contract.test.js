import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const html = fs.readFileSync(new URL('../public/index.html', import.meta.url), 'utf8');
const app = fs.readFileSync(new URL('../public/app.js', import.meta.url), 'utf8');

test('UI keeps the Trainer shell and explicit family switch', () => {
  for (const section of ['Dashboard', 'New Run', 'Export', 'Models', 'xAI', 'Data', 'History']) {
    assert.match(html, new RegExp(`>${section}<`));
  }
  assert.match(html, /data-family="detection"/);
  assert.match(html, /data-family="classification"/);
  assert.match(html, /data-family="video"/);
  assert.match(html, /data-family="vlm"/);
});

test('all four Trainer families expose independent workflows', () => {
  assert.match(app, /\['finetune', 'Finetune'\]/);
  assert.match(app, /\['tune', 'Tune'\]/);
  assert.match(app, /data-type="classifier_\$\{mode\}"/);
  assert.match(app, /classifier_finetune_/);
  assert.match(app, /classifier_tune_/);
  assert.match(app, /Early-stop patience/);
  assert.match(app, /Thresholds JSON/);
  assert.match(app, /\['xai', 'xAI'\]/);
  assert.match(app, /SIGNED GRAD-CAM|signed Grad-CAM/i);
  assert.match(app, /Channels per stage/);
  assert.match(app, /data-type="video_\$\{mode\}"/);
  assert.match(app, /data-type="vlm_tune"/);
  assert.match(app, /data-job-type="vlm_merge"/);
  assert.match(app, /data-type="vlm_export"/);
  assert.match(app, /Export ONNX/);
  assert.match(app, /Export TensorRT/);
  assert.match(app, /TensorRT requires a CUDA target/);
  assert.match(app, /data-type="detect_/);
  assert.match(app, /convnext-tiny/);
  assert.match(app, /r2plus1d-tiny/);
  assert.match(app, /hf-lora/);
  assert.match(app, /Shared augmentation/);
  assert.match(app, /External augmentation TOML/);
  assert.match(app, /Mosaic probability/);
  assert.match(app, /Motion blur probability/);
});

test('compute targets visibly distinguish GPU and CPU-only machines', () => {
  assert.match(app, /computeKind/);
  assert.match(app, /CPU-only/);
  assert.match(app, /compute-kind/);
  assert.match(app, /machineOptionLabel/);
  assert.match(app, /Compute targets/);
});

test('history combines queued jobs with runs discovered on trainer machines', () => {
  assert.match(app, /api\('\/runs\?limit=1000'\)/);
  assert.match(app, /All model families/);
  assert.match(app, /Control-plane jobs/);
  assert.match(app, /Discovered runs/);
  assert.match(app, /vm_smoke_matrix\/summary\.json/);
});

test('registry and sample-level xAI flows remain available', () => {
  assert.match(app, /renderModels/);
  assert.match(app, /Interactive model flow/);
  assert.match(app, /trainer\.architecture/);
  assert.match(app, /renderEvidence/);
  assert.match(app, /drawClassificationGradcam/);
  assert.match(app, /Test one image/);
  assert.match(app, /Run test sample \+ xAI/);
  assert.match(app, /Prediction \+ ConvNeXt xAI/);
  assert.match(app, /sample:\/\//);
  assert.match(app, /Activation intensity/);
  assert.match(app, /Red supports the selected class; blue opposes it/i);
  assert.match(app, /setTimeout\(resolve, 350\)/);
  assert.doesNotMatch(app, /Curated run outputs/);
  assert.doesNotMatch(app, /Open artifact/);
  assert.doesNotMatch(app, /\/api\/evidence/);
  assert.doesNotMatch(app, /manifest\.json/);
  assert.doesNotMatch(app, /metrics\.json/);
});
