import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const html = fs.readFileSync(new URL('../public/index.html', import.meta.url), 'utf8');
const app = fs.readFileSync(new URL('../public/app.js', import.meta.url), 'utf8');

test('UI keeps the gender-automation shell and explicit family switch', () => {
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
  assert.match(app, /data-type="vlm_merge"/);
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

test('registry and xAI flows are available to every Trainer family', () => {
  assert.match(app, /renderModels/);
  assert.match(app, /Interactive model flow/);
  assert.match(app, /trainer\.architecture/);
  assert.match(app, /renderEvidence/);
  assert.match(app, /Mask\/background diagnostics/);
  assert.match(app, /bindEvidenceImageRetry/);
  assert.match(app, /drawClassificationGradcam/);
  assert.match(app, /Activation intensity/);
  assert.match(app, /Red supports the selected class; blue opposes it/i);
  assert.match(app, /setTimeout\(resolve, 350\)/);
});
