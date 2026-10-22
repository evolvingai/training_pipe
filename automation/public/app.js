/* Trainer Automation — same control-panel interaction model as gender_automation. */

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

let CONFIG = null;
let CURRENT_USER = null;
let jobTimer = null;

const FAMILY_KEY = 'trainer.family';
const TARGET_KEY = 'trainer.targetAgent';
const FAMILIES = {
  detection: { label: 'Detection', task: 'detection', prefix: 'detect_', data: 'YOLO labels + dataset TOML', export: 'Detector ONNX / TensorRT' },
  classification: { label: 'Image Classification', task: 'classification', prefix: 'classifier_', data: 'Class-folder image datasets', export: 'Image classifier ONNX + class contract' },
  video: { label: 'Video / Action Classification', task: 'video-classification', prefix: 'video_', data: 'Class-folder video datasets', export: 'Video classifier ONNX + clip contract' },
  vlm: { label: 'VLM Tuning', task: 'vlm', prefix: 'vlm_', data: 'Image-text JSONL datasets', export: 'Portable LoRA adapter or merged VLM' },
};

async function api(path, options = {}) {
  const request = () => fetch(`/api${path}`, {
    headers: { 'content-type': 'application/json', ...(options.headers || {}) },
    ...options,
    body: options.body == null || typeof options.body === 'string' ? options.body : JSON.stringify(options.body),
  });
  let response;
  try {
    response = await request();
  } catch (error) {
    if (String(options.method || 'GET').toUpperCase() !== 'GET') throw error;
    await new Promise((resolve) => setTimeout(resolve, 350));
    response = await request();
  }
  if (response.status === 401 && path !== '/login') {
    showLogin();
    throw new Error('authentication required');
  }
  const value = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
  return value;
}

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (character) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[character]);
}

function fmtTime(value) {
  return value ? new Date(value).toLocaleString() : '—';
}

function fmtDuration(start, end) {
  if (!start) return '—';
  const seconds = Math.max(0, Math.floor(((end || Date.now()) - start) / 1000));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return hours ? `${hours}h ${minutes}m` : minutes ? `${minutes}m ${seconds % 60}s` : `${seconds}s`;
}

function badge(status) {
  return `<span class="badge ${esc(status)}">${esc(String(status).replaceAll('_', ' '))}</span>`;
}

function toast(message, isError = false) {
  const element = $('#toast');
  element.textContent = message;
  element.className = `toast${isError ? ' err' : ''}`;
  clearTimeout(element._timer);
  element._timer = setTimeout(() => element.classList.add('hidden'), 5000);
}

function currentFamily() {
  const saved = localStorage.getItem(FAMILY_KEY);
  return Object.hasOwn(FAMILIES, saved) ? saved : 'detection';
}

function familyLabel() {
  return FAMILIES[currentFamily()].label;
}

function familyMatches(job) {
  return String(job.type).startsWith(FAMILIES[currentFamily()].prefix);
}

function jobFamily(job) {
  const entry = Object.entries(FAMILIES).find(([, value]) => String(job.type).startsWith(value.prefix));
  return entry ? entry[1].label : 'Unknown';
}

function checkpointMatches(path) {
  const value = String(path).toLowerCase();
  if (currentFamily() === 'detection') return value.includes('/weights/');
  if (currentFamily() === 'vlm') return value.endsWith('/adapter') || value.includes('/vlm_');
  if (currentFamily() === 'video') return value.includes('/video') || value.includes('/action');
  return !value.includes('/weights/') && !value.includes('/video') && !value.includes('/action') && !value.endsWith('/adapter');
}

function applyFamilyChrome() {
  const family = currentFamily();
  document.title = `${familyLabel()} · Trainer Automation`;
  $$('#family-switch button').forEach((button) => button.classList.toggle('active', button.dataset.family === family));
}

function setFamily(family) {
  localStorage.setItem(FAMILY_KEY, Object.hasOwn(FAMILIES, family) ? family : 'detection');
  applyFamilyChrome();
  if ((location.hash || '#/dashboard') !== '#/dashboard') location.hash = '#/dashboard';
  else route();
}

function machineList(agents) {
  return Object.values(agents || {})
    .map((agent) => ({ ...agent, online: Date.now() - Number(agent.at || 0) < 90_000 }))
    .sort((left, right) => Number(right.online) - Number(left.online) || Number(right.at) - Number(left.at));
}

function selectedTarget() {
  const element = $('#job-machine');
  if (element?.value) {
    localStorage.setItem(TARGET_KEY, element.value);
    return element.value;
  }
  return localStorage.getItem(TARGET_KEY) || '';
}

function gpuHtml(agent) {
  const gpus = Array.isArray(agent?.gpus) ? agent.gpus : [];
  if (!gpus.length) return '<div class="gpu-bar">CPU-only · no NVIDIA GPU reported</div>';
  return gpus.map((gpu, index) => {
    const total = gpu.memoryTotalMb ?? gpu.memTotal ?? '—';
    const free = gpu.memoryFreeMb ?? '—';
    return `<div class="gpu-bar">cuda:${esc(gpu.index ?? index)} ${esc(gpu.name || 'GPU')} · CC ${esc(gpu.computeCapability || '—')} · ${esc(free)}/${esc(total)} MiB free</div>`;
  }).join('');
}

function computeKind(agent) {
  return Array.isArray(agent?.gpus) && agent.gpus.length ? 'GPU' : 'CPU-only';
}

function machineName(agent) {
  return agent?.label || agent?.host || agent?.agentId || 'machine';
}

function machineStatus(agent) {
  if (!agent.online) return 'offline';
  return agent.activeJobId ? 'busy' : 'ready';
}

function machineOptionLabel(machine) {
  const gpu = machine.gpus?.[0];
  const hardware = gpu ? gpu.name || 'GPU' : `${machine.cpus || '—'} vCPU`;
  return `${machineName(machine)} · ${computeKind(machine)} · ${hardware} · ${machineStatus(machine)}`;
}

function machineCardHtml(machine) {
  const status = machineStatus(machine);
  const kind = computeKind(machine);
  const provider = [machine.provider, machine.machineType].filter(Boolean).join(' · ') || 'Unspecified provider';
  return `<article class="compute-card ${machine.online ? 'online' : 'offline'}">
    <div class="compute-head"><div><div class="compute-title">${esc(machineName(machine))}</div><div class="compute-sub">${esc(provider)}</div></div>
      <div class="row"><span class="compute-kind ${kind === 'GPU' ? 'gpu' : 'cpu'}">${kind}</span>${badge(status)}</div></div>
    <dl class="kv compact"><dt>Agent</dt><dd>${esc(machine.agentId)}</dd><dt>Host</dt><dd>${esc(machine.host || 'not connected')}</dd>
      <dt>CPU / memory</dt><dd>${esc(machine.cpus || '—')} cores · ${esc(machine.memoryFreeMb || '—')}/${esc(machine.memoryTotalMb || '—')} MiB free</dd>
      <dt>Heartbeat</dt><dd>${fmtTime(machine.at)}</dd>${machine.activeJobId ? `<dt>Active job</dt><dd>${esc(machine.activeJobId)}</dd>` : ''}</dl>
    ${gpuHtml(machine)}
    ${machine.disk ? `<div class="disk-line">${esc(machine.disk.freeKb)} KB free at ${esc(machine.disk.mount)}</div>` : ''}
  </article>`;
}

function machinePickerHtml(agents) {
  const machines = machineList(agents);
  if (!machines.length) {
    return `<div class="card"><h3>Target</h3><p class="muted">No agent has heartbeated yet. Jobs remain queued until a compatible agent comes online.</p></div>`;
  }
  const saved = localStorage.getItem(TARGET_KEY);
  const selected = machines.find((machine) => machine.agentId === saved) || machines.find((machine) => machine.online) || machines[0];
  localStorage.setItem(TARGET_KEY, selected.agentId);
  return `<div class="card" id="machine-card">
    <h3>Target</h3>
    <p class="muted">Jobs run only on the selected machine. CPU agents cannot claim an explicit CUDA job.</p>
    <label class="field"><span>Machine</span><select id="job-machine">
      ${machines.map((machine) => `<option value="${esc(machine.agentId)}" ${machine.agentId === selected.agentId ? 'selected' : ''}>${esc(machineOptionLabel(machine))}</option>`).join('')}
    </select></label>
    <div id="machine-hardware">${gpuHtml(selected)}</div>
  </div>`;
}

function bindMachinePicker(agents) {
  $('#job-machine')?.addEventListener('change', (event) => {
    localStorage.setItem(TARGET_KEY, event.target.value);
    const machine = machineList(agents).find((item) => item.agentId === event.target.value);
    if ($('#machine-hardware')) $('#machine-hardware').innerHTML = gpuHtml(machine);
  });
}

function deviceOptions(selected = '') {
  const current = selected || CONFIG.defaults.device || 'cpu';
  return [
    ['cpu', 'CPU'], ['auto', 'Auto'], ['cuda', 'CUDA (first GPU)'],
    ['cuda:0', 'CUDA 0'], ['cuda:1', 'CUDA 1'], ['mps', 'Apple MPS'],
  ].map(([value, label]) => `<option value="${value}" ${value === current ? 'selected' : ''}>${label}</option>`).join('');
}

function field(name, label, { value = '', placeholder = '', type = 'text', required = false, min, step } = {}) {
  const attributes = [
    `name="${name}"`, `type="${type}"`, `value="${esc(value)}"`, `placeholder="${esc(placeholder)}"`,
    required ? 'required' : '', min != null ? `min="${min}"` : '', step != null ? `step="${step}"` : '',
  ].filter(Boolean).join(' ');
  return `<label class="field"><span>${label}</span><input ${attributes} /></label>`;
}

function architecturePlugins(family = currentFamily()) {
  const task = FAMILIES[family]?.task;
  return (CONFIG?.registry?.architectures || []).filter((architecture) => architecture.task === task);
}

function architectureField(name, label, { value = '', optional = false } = {}) {
  const plugins = architecturePlugins();
  const saved = localStorage.getItem(`trainer.architecture.${currentFamily()}`);
  const selected = saved && plugins.some((plugin) => plugin.name === saved) ? saved : value;
  if (!plugins.length) return field(name, label, { value: selected, required: !optional });
  return `<label class="field"><span>${esc(label)}</span><select name="${esc(name)}" ${optional ? '' : 'required'}>
    ${optional ? '<option value="">Use checkpoint architecture</option>' : ''}
    ${plugins.map((plugin) => `<option value="${esc(plugin.name)}" ${plugin.name === selected ? 'selected' : ''}>${esc(plugin.name)} · ${esc(plugin.family)}</option>`).join('')}
  </select></label>`;
}

function commonRunFields(defaultName) {
  return `<div class="card">
    <h3>Experiment</h3>
    ${field('name', 'Run name', { value: defaultName, required: true })}
    <div class="grid2">
      <label class="field"><span>Device</span><select name="device">${deviceOptions()}</select></label>
      ${field('project', 'Runs directory', { value: CONFIG.defaults.runsDir })}
    </div>
  </div>`;
}

function advancedFields() {
  return `<details class="adv"><summary>Advanced paths and execution</summary>
    <div class="grid2">
      ${field('codeDir', 'Training code directory', { value: CONFIG.defaults.codeDir })}
      ${field('pythonBin', 'Python executable', { value: CONFIG.defaults.pythonBin })}
    </div>
    <label class="inline-check"><input type="checkbox" name="dryRun" /> Dry run — print commands without executing them</label>
  </details>`;
}

function augmentationFields({ multiSample = true } = {}) {
  return `<details class="adv"><summary>Shared augmentation</summary>
    <p class="muted">Uses the task-neutral Trainer augmentation pipeline. Leave fields empty for task defaults, or load an external TOML profile.</p>
    ${field('augmentationConfig', 'External augmentation TOML', { placeholder: '/data/augmentations/robust.toml' })}
    <label class="field"><span>Camera profile</span><select name="augProfile"><option value="">Task default</option><option value="generic">Generic</option><option value="topdown">Top-down</option><option value="frontal">Frontal</option></select></label>
    ${multiSample ? `<h4>Multi-image</h4><div class="adv-grid">
      ${field('mosaic', 'Mosaic probability', { type: 'number', min: 0, step: 0.01 })}
      ${field('mosaic9', 'Mosaic-9 probability', { type: 'number', min: 0, step: 0.01 })}
      ${field('mixup', 'MixUp probability', { type: 'number', min: 0, step: 0.01 })}
      ${field('mixupAlpha', 'MixUp alpha', { type: 'number', min: 0, step: 0.01 })}
      ${field('copyPaste', 'CopyPaste / CutMix probability', { type: 'number', min: 0, step: 0.01 })}
      ${field('copyPasteIou', 'CopyPaste area/IoU range', { placeholder: '0.10,0.55' })}
    </div>` : '<p class="muted">Multi-image Mosaic, MixUp, and CopyPaste are disabled for sequence/text-paired inputs.</p>'}
    <h4>Geometry</h4><div class="adv-grid">
      ${field('degrees', 'Rotation degrees', { type: 'number', min: 0, step: 0.1 })}
      ${field('translate', 'Translation fraction', { type: 'number', min: 0, step: 0.01 })}
      ${field('scale', 'Scale jitter', { type: 'number', min: 0, step: 0.01 })}
      ${field('shear', 'Shear degrees', { type: 'number', min: 0, step: 0.1 })}
      ${field('perspective', 'Perspective', { type: 'number', min: 0, step: 0.001 })}
      ${field('flipud', 'Vertical flip probability', { type: 'number', min: 0, step: 0.01 })}
      ${field('fliplr', 'Horizontal flip probability', { type: 'number', min: 0, step: 0.01 })}
      ${field('squareCropJitter', 'Square-crop jitter', { type: 'number', min: 0, step: 0.01 })}
    </div><label class="inline-check"><input type="checkbox" name="squareCrop" /> Enable square crop</label>
    <h4>Color and sensor</h4><div class="adv-grid">
      ${field('hsvH', 'HSV hue', { type: 'number', min: 0, step: 0.001 })}
      ${field('hsvS', 'HSV saturation', { type: 'number', min: 0, step: 0.01 })}
      ${field('hsvV', 'HSV value', { type: 'number', min: 0, step: 0.01 })}
      ${field('brightnessContrast', 'Brightness/contrast probability', { type: 'number', min: 0, step: 0.01 })}
      ${field('brightness', 'Brightness magnitude', { type: 'number', min: 0, step: 0.01 })}
      ${field('contrast', 'Contrast magnitude', { type: 'number', min: 0, step: 0.01 })}
      ${field('blur', 'Motion blur probability', { type: 'number', min: 0, step: 0.01 })}
      ${field('noise', 'Noise probability', { type: 'number', min: 0, step: 0.01 })}
      ${field('jpeg', 'JPEG artifact probability', { type: 'number', min: 0, step: 0.01 })}
      ${field('erasing', 'Random erasing probability', { type: 'number', min: 0, step: 0.01 })}
    </div>
  </details>`;
}

function detectionForm(mode) {
  const stamp = new Date().toISOString().slice(0, 10).replaceAll('-', '_');
  const names = {
    train: `detect_train_${stamp}`, finetune: `detect_finetune_${stamp}`,
    evaluate: `detect_eval_${stamp}`, predict: `detect_predict_${stamp}`, tune: `detect_tune_${stamp}`,
  };
  const headers = {
    train: ['Train detector', 'Start a new detector from an architecture definition.'],
    finetune: ['Finetune detector', 'Continue from a checkpoint and optionally freeze early modules.'],
    evaluate: ['Evaluate detector', 'Run validation and build the full detector report.'],
    predict: ['Predict', 'Run detector inference and save annotated images.'],
    tune: ['Tune', 'Run detector hyperparameter evolution.'],
  };
  let body = '';
  if (mode === 'train') body = `<div class="card"><h3>Dataset and model</h3>
    ${field('data', 'Dataset TOML', { placeholder: '/data/dataset.toml', required: true })}
    ${architectureField('model', 'Model architecture', { value: 'trainer-n' })}</div>
    <div class="card"><h3>Training parameters</h3><div class="adv-grid">
      ${field('epochs', 'Epochs', { value: '100', type: 'number', min: 1 })}
      ${field('batch', 'Batch size', { value: '2', type: 'number', min: 1 })}
      ${field('imgsz', 'Image size', { value: '320', type: 'number', min: 32 })}
    </div></div>`;
  if (mode === 'finetune') body = `<div class="card"><h3>Checkpoint and dataset</h3>
    ${field('checkpoint', 'Source checkpoint', { placeholder: '/runs/base/weights/best.pt', required: true })}
    ${field('data', 'Dataset TOML', { placeholder: '/data/dataset.toml', required: true })}</div>
    <div class="card"><h3>Finetuning parameters</h3><div class="adv-grid">
      ${field('epochs', 'Epochs', { value: '30', type: 'number', min: 1 })}
      ${field('batch', 'Batch size', { value: '2', type: 'number', min: 1 })}
      ${field('imgsz', 'Image size', { value: '320', type: 'number', min: 32 })}
      ${field('freeze', 'Freeze first N modules', { value: '10', type: 'number', min: 0 })}
    </div></div>`;
  if (mode === 'evaluate') body = `<div class="card"><h3>Evaluation inputs</h3>
    ${field('checkpoint', 'Checkpoint', { placeholder: '/runs/experiment/weights/best.pt', required: true })}
    ${field('data', 'Dataset TOML', { placeholder: '/data/dataset.toml', required: true })}
    <p class="muted">This recipe runs validation first, then writes the HTML/CSV report into the named run folder.</p></div>`;
  if (mode === 'predict') body = `<div class="card"><h3>Inference inputs</h3>
    ${field('checkpoint', 'Checkpoint', { placeholder: '/runs/experiment/weights/best.pt', required: true })}
    ${field('source', 'Images or directory', { placeholder: '/data/images', required: true })}
    ${field('conf', 'Confidence threshold', { value: '0.25', type: 'number', min: 0, step: 0.01 })}</div>`;
  if (mode === 'tune') body = `<div class="card"><h3>Tuning inputs</h3>
    ${field('data', 'Dataset TOML', { placeholder: '/data/dataset.toml', required: true })}
    ${architectureField('model', 'Model architecture', { value: 'trainer-n' })}
    ${field('iterations', 'Tune iterations', { value: '30', type: 'number', min: 1 })}</div>`;
  return `<h2 class="page-title">${headers[mode][0]}</h2><p class="page-sub">${headers[mode][1]}</p>
    <form id="run-form" data-type="detect_${mode === 'evaluate' ? 'evaluate' : mode}">
      ${commonRunFields(names[mode])}${body}${['train', 'finetune'].includes(mode) ? augmentationFields() : ''}${advancedFields()}${runActions(headers[mode][0])}
    </form>`;
}

function classificationForm(mode) {
  const stamp = new Date().toISOString().slice(0, 10).replaceAll('-', '_');
  const names = {
    train: `classifier_train_${stamp}`, finetune: `classifier_finetune_${stamp}`,
    evaluate: `classifier_eval_${stamp}`, predict: `classifier_predict_${stamp}`, xai: `classifier_xai_${stamp}`, tune: `classifier_tune_${stamp}`,
  };
  const headers = {
    train: ['Train classifier', 'Train the selected architecture plugin from class folders.'],
    finetune: ['Finetune classifier', 'Transfer a checkpoint to the current classes and optionally freeze feature stages.'],
    evaluate: ['Evaluate classifier', 'Compute full metrics, tune per-class thresholds, and build the evaluation report.'],
    predict: ['Classify images', 'Write top-k predictions and apply optional per-class confidence thresholds.'],
    xai: ['Explain classifier', 'Generate signed Grad-CAM overlays and ConvNeXt stage activations like the fish xAI flow.'],
    tune: ['Tune classifier', 'Search learning rate, regularization, dropout, and label smoothing.'],
  };
  let body = '';
  if (mode === 'train') body = `<div class="card"><h3>Classification dataset</h3>
    ${field('data', 'Dataset root', { placeholder: '/data/species', required: true })}
    ${architectureField('arch', 'Architecture plugin', { value: 'convnext-tiny' })}
    <div class="pipeline"><code>train/class_name/*.jpg</code><span>or</span><code>class_name/*.jpg</code></div>
    <p class="muted">Explicit train/val folders are used as-is. A compact class-folder dataset is split deterministically per class.</p></div>
    <div class="card"><h3>Training parameters</h3><div class="adv-grid">
      ${field('epochs', 'Epochs', { value: '30', type: 'number', min: 1 })}
      ${field('batch', 'Batch size', { value: '8', type: 'number', min: 1 })}
      ${field('imgsz', 'Image size', { value: '160', type: 'number', min: 32 })}
      ${field('patience', 'Early-stop patience', { value: '20', type: 'number', min: 1 })}
      ${field('accumulate', 'Gradient accumulation', { value: '1', type: 'number', min: 1 })}
    </div>${field('resume', 'Resume checkpoint (optional)', { placeholder: '/runs/classifier/weights/last.pt' })}</div>`;
  if (mode === 'finetune') body = `<div class="card"><h3>Checkpoint and dataset</h3>
    ${field('checkpoint', 'Source checkpoint', { placeholder: '/runs/classifier/weights/best.pt', required: true })}
    ${field('data', 'Dataset root', { placeholder: '/data/species', required: true })}
    ${architectureField('arch', 'Architecture override (optional)', { optional: true })}</div>
    <div class="card"><h3>Finetuning parameters</h3><div class="adv-grid">
      ${field('epochs', 'Epochs', { value: '20', type: 'number', min: 1 })}
      ${field('batch', 'Batch size', { value: '8', type: 'number', min: 1 })}
      ${field('imgsz', 'Image size', { value: '160', type: 'number', min: 32 })}
      ${field('freeze', 'Freeze feature stages', { value: '4', type: 'number', min: 0 })}
      ${field('patience', 'Early-stop patience', { value: '10', type: 'number', min: 1 })}
    </div></div>`;
  if (mode === 'evaluate') body = `<div class="card"><h3>Evaluation inputs</h3>
    ${field('checkpoint', 'Classifier checkpoint', { placeholder: '/runs/classifier/weights/best.pt', required: true })}
    ${field('data', 'Labelled dataset root', { placeholder: '/data/species', required: true })}
    ${field('batch', 'Batch size', { value: '32', type: 'number', min: 1 })}
    <p class="muted">Produces metrics JSON, per-image CSV, confusion matrix, optimized class thresholds, and an HTML report.</p></div>`;
  if (mode === 'predict') body = `<div class="card"><h3>Prediction inputs</h3>
    ${field('checkpoint', 'Classifier checkpoint', { placeholder: '/runs/classifier/weights/best.pt', required: true })}
    ${field('source', 'Image or directory', { placeholder: '/data/images', required: true })}
    ${field('thresholds', 'Thresholds JSON (optional)', { placeholder: '/runs/classifier_eval/thresholds.json' })}
    <div class="adv-grid">
      ${field('batch', 'Batch size', { value: '32', type: 'number', min: 1 })}
      ${field('topk', 'Top-k predictions', { value: '3', type: 'number', min: 1 })}
      ${field('conf', 'Fallback confidence', { value: '0', type: 'number', min: 0, step: 0.01 })}
    </div></div>`;
  if (mode === 'xai') body = `<div class="card"><h3>ConvNeXt xAI inputs</h3>
    ${field('checkpoint', 'Classifier checkpoint', { placeholder: '/runs/classifier/weights/best.pt', required: true })}
    ${field('source', 'Image or directory', { placeholder: '/data/images', required: true })}
    <div class="adv-grid">
      ${field('target', 'Target class', { value: 'predicted', placeholder: 'predicted, class name, or index' })}
      ${field('maxImages', 'Maximum images', { value: '24', type: 'number', min: 1 })}
      ${field('activationChannels', 'Channels per stage', { value: '4', type: 'number', min: 1 })}
      ${field('intensity', 'Overlay intensity', { value: '0.72', type: 'number', min: 0.05, step: 0.05 })}
    </div><p class="muted">Red supports the selected class; blue opposes it. Trainer also exports the highest-energy feature channels at each ConvNeXt stage and an auditable JSON manifest.</p></div>`;
  if (mode === 'tune') body = `<div class="card"><h3>Tuning inputs</h3>
    ${field('data', 'Dataset root', { placeholder: '/data/species', required: true })}
    ${architectureField('arch', 'Architecture plugin', { value: 'convnext-tiny' })}
    <div class="adv-grid">
      ${field('iterations', 'Trials', { value: '10', type: 'number', min: 1 })}
      ${field('tuneEpochs', 'Epochs per trial', { value: '5', type: 'number', min: 1 })}
      ${field('batch', 'Batch size', { value: '8', type: 'number', min: 1 })}
      ${field('imgsz', 'Image size', { value: '160', type: 'number', min: 32 })}
    </div><p class="muted">Each trial is a normal Trainer run with checkpoints, metrics, curves, and a report.</p></div>`;
  return `<h2 class="page-title">${headers[mode][0]}</h2><p class="page-sub">${headers[mode][1]}</p>
    <form id="run-form" data-type="classifier_${mode}">
      ${commonRunFields(names[mode])}${body}${['train', 'finetune', 'tune'].includes(mode) ? augmentationFields() : ''}${advancedFields()}${runActions(headers[mode][0])}
    </form>`;
}

function videoForm(mode) {
  const stamp = new Date().toISOString().slice(0, 10).replaceAll('-', '_');
  const names = {
    train: `video_train_${stamp}`, evaluate: `video_eval_${stamp}`, predict: `video_predict_${stamp}`,
  };
  const headers = {
    train: ['Train video/action classifier', 'Learn action classes with the selected temporal architecture plugin.'],
    evaluate: ['Evaluate video classifier', 'Compute clip accuracy, per-class accuracy, and a confusion matrix.'],
    predict: ['Classify videos', 'Write the action class and confidence for each source video.'],
  };
  let body = '';
  if (mode === 'train') body = `<div class="card"><h3>Video dataset</h3>
    ${field('data', 'Dataset root', { placeholder: '/data/actions', required: true })}
    ${architectureField('arch', 'Architecture plugin', { value: 'r2plus1d-tiny' })}
    <div class="pipeline"><code>train/action_name/*.mp4</code><span>or</span><code>action_name/*.mp4</code></div>
    <p class="muted">Supports MP4, AVI, MOV, MKV, M4V, and WebM. Compact datasets are split deterministically per action.</p></div>
    <div class="card"><h3>Training parameters</h3><div class="adv-grid">
      ${field('epochs', 'Epochs', { value: '30', type: 'number', min: 1 })}
      ${field('batch', 'Batch size', { value: '2', type: 'number', min: 1 })}
      ${field('imgsz', 'Frame size', { value: '112', type: 'number', min: 32 })}
      ${field('clipLen', 'Frames per clip', { value: '8', type: 'number', min: 2 })}
    </div><p class="muted">The compact frame encoder + GRU baseline runs on CPU. Use CUDA for larger clips and faster iteration.</p></div>`;
  if (mode === 'evaluate') body = `<div class="card"><h3>Evaluation inputs</h3>
    ${field('checkpoint', 'Video checkpoint', { placeholder: '/runs/video_action/best.pt', required: true })}
    ${field('data', 'Labelled video dataset', { placeholder: '/data/actions', required: true })}
    ${field('batch', 'Batch size', { value: '4', type: 'number', min: 1 })}</div>`;
  if (mode === 'predict') body = `<div class="card"><h3>Prediction inputs</h3>
    ${field('checkpoint', 'Video checkpoint', { placeholder: '/runs/video_action/best.pt', required: true })}
    ${field('source', 'Video or directory', { placeholder: '/data/videos', required: true })}
    ${field('batch', 'Batch size', { value: '4', type: 'number', min: 1 })}</div>`;
  return `<h2 class="page-title">${headers[mode][0]}</h2><p class="page-sub">${headers[mode][1]}</p>
    <form id="run-form" data-type="video_${mode}">
      ${commonRunFields(names[mode])}${body}${mode === 'train' ? augmentationFields({ multiSample: false }) : ''}${advancedFields()}${runActions(headers[mode][0])}
    </form>`;
}

function vlmForm() {
  const stamp = new Date().toISOString().slice(0, 10).replaceAll('-', '_');
  return `<h2 class="page-title">Tune vision-language model</h2><p class="page-sub">Create a portable PEFT LoRA adapter from image, prompt, and answer examples.</p>
    <form id="run-form" data-type="vlm_tune">
      ${commonRunFields(`vlm_lora_${stamp}`)}
      <div class="warn-banner">VLM tuning can run on CPU for a small smoke test, but even compact models are much faster on CUDA. Start with a small model and a limited dataset before moving to an L4.</div>
      <div class="card"><h3>Model and dataset</h3>
        ${architectureField('arch', 'Architecture plugin', { value: 'hf-lora' })}
        ${field('model', 'Hugging Face model', { placeholder: 'organization/model-name', required: true })}
        ${field('data', 'Training JSONL', { placeholder: '/data/vlm/train.jsonl', required: true })}
        <div class="pipeline"><code>{"image":"images/1.jpg","prompt":"What is shown?","answer":"..."}</code></div>
      </div>
      <div class="card"><h3>LoRA tuning parameters</h3><div class="adv-grid">
        ${field('epochs', 'Epochs', { value: '1', type: 'number', min: 1 })}
        ${field('batch', 'Micro batch', { value: '1', type: 'number', min: 1 })}
        ${field('gradientAccumulation', 'Gradient accumulation', { value: '8', type: 'number', min: 1 })}
        ${field('lr', 'Learning rate', { value: '0.00002', type: 'number', min: 0, step: 0.000001 })}
        ${field('maxLength', 'Max tokens', { value: '1024', type: 'number', min: 32 })}
        ${field('loraR', 'LoRA rank', { value: '16', type: 'number', min: 1 })}
        ${field('loraAlpha', 'LoRA alpha', { value: '32', type: 'number', min: 1 })}
      </div>
      <div class="grid2">
        <label class="field"><span>Compute dtype</span><select name="dtype"><option value="auto">Auto</option><option value="bfloat16">bfloat16</option><option value="float16">float16</option><option value="float32">float32</option></select></label>
        ${field('targetModules', 'LoRA target modules', { value: 'q_proj,k_proj,v_proj,o_proj' })}
      </div>
      <label class="inline-check"><input type="checkbox" name="trustRemoteCode" /> Allow model repository custom code</label></div>
      ${augmentationFields({ multiSample: false })}${advancedFields()}${runActions('VLM tune')}
    </form>`;
}

function runActions(label) {
  return `<div class="row"><button type="button" class="btn" id="preview-btn">Preview commands</button><button type="submit" class="btn primary">Queue ${esc(label.toLowerCase())}</button></div><div id="preview-out" style="margin-top:18px"></div>`;
}

function formPayload(form) {
  const params = Object.fromEntries(new FormData(form));
  const dryRun = params.dryRun === 'on';
  delete params.dryRun;
  for (const key of Object.keys(params)) if (params[key] === '') delete params[key];
  const targetAgentId = selectedTarget();
  if (targetAgentId) params.targetAgentId = targetAgentId;
  return { type: form.dataset.type, params, dryRun };
}

function commandText(command) {
  const quote = (value) => /[\s"']/.test(value) ? JSON.stringify(value) : value;
  return `${command.cwd ? `cd ${quote(command.cwd)}\n` : ''}${(command.argv || []).map(quote).join(' ')}`;
}

function bindRunForm() {
  const form = $('#run-form');
  if (!form) return;
  $('#preview-btn')?.addEventListener('click', async () => {
    try {
      const payload = formPayload(form);
      const result = await api('/preview', { method: 'POST', body: payload });
      $('#preview-out').innerHTML = result.commands.map((command) => `<div class="cmd-title">${esc(command.title)}</div><div class="cmd-box">${esc(commandText(command))}</div>`).join('');
    } catch (error) { toast(error.message, true); }
  });
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    try {
      const { job } = await api('/jobs', { method: 'POST', body: formPayload(form) });
      toast(`Queued ${job.name}`);
      location.hash = `#/job/${job.id}`;
    } catch (error) { toast(error.message, true); }
  });
}

async function renderDashboard(main) {
  main.innerHTML = `<h2 class="page-title">Dashboard</h2><p class="page-sub">${familyLabel()} — training VM, active jobs, artifacts and queue</p><div id="dashboard-loading" class="card"><div class="muted">Loading…</div></div>`;
  const [{ driver }, { agents }, { jobs }, { inventory }] = await Promise.all([
    api('/vm').catch(() => ({ driver: { configured: false, state: 'unknown' } })),
    api('/agents'), api('/jobs?limit=80'), api('/inventory'),
  ]);
  const machines = machineList(agents);
  const online = machines.filter((machine) => machine.online);
  const familyJobs = jobs.filter(familyMatches);
  const checkpoints = Object.values(inventory || {}).flatMap((item) => item.checkpoints || []);
  main.innerHTML = `<h2 class="page-title">Dashboard</h2><p class="page-sub">${familyLabel()} — training VM, active jobs, artifacts and queue</p>
    <div id="dash-target">${machinePickerHtml(agents)}</div>
    <div class="grid2">
      <div class="card" id="vm-card"><h3>Training VM ${badge(driver.state === 'running' ? 'completed' : 'cancelled')}</h3>
        <div class="kv"><dt>Power driver</dt><dd>${esc(CONFIG.power.name)}</dd><dt>State</dt><dd>${esc(driver.state || 'unknown')}</dd><dt>Auto-shutdown</dt><dd>${CONFIG.autoShutdown ? 'enabled' : 'disabled'}</dd></div>
        ${CONFIG.power.configured ? '<div class="row" style="margin-top:16px"><button class="btn small" id="vm-start">Start VM</button><button class="btn small danger" id="vm-stop">Deallocate VM</button></div>' : '<p class="muted">Power control is not configured; manage this machine in Azure/GCP.</p>'}
      </div>
      <div class="card"><h3>Compute targets ${online.length ? badge('completed') : badge('cancelled')}</h3>
        ${machines.length ? `<div class="compute-grid">${machines.map(machineCardHtml).join('')}</div>` : '<p class="muted">No compute target is configured.</p>'}
      </div>
    </div>
    <div class="grid2">
      <div class="card"><h3>Data</h3><p class="muted">${FAMILIES[currentFamily()].data}</p><a class="btn" href="#/data">Open data guide</a></div>
      <div class="card"><h3>Export</h3><p class="muted">${FAMILIES[currentFamily()].export}</p><a class="btn" href="#/export">New export</a></div>
    </div>
    <div class="grid2">
      <div class="card"><h3>Model registry</h3><p class="muted">Installed architecture plugins, their contracts, and the exact model flow.</p><a class="btn" href="#/models">Open models</a></div>
      <div class="card"><h3>xAI and evidence</h3><p class="muted">Reports, overlays, augmentation previews, predictions, and mask/background diagnostics.</p><a class="btn" href="#/evidence">Open xAI</a></div>
    </div>
    <div class="card"><h3>Runs</h3>${checkpoints.length ? `<table><thead><tr><th>Checkpoint</th><th>Agent</th></tr></thead><tbody>${Object.entries(inventory).flatMap(([agentId, value]) => (value.checkpoints || []).filter(checkpointMatches).slice(-8).map((item) => `<tr><td><code>${esc(item)}</code></td><td>${esc(agentId)}</td></tr>`)).join('')}</tbody></table>` : '<div class="empty">No checkpoint inventory yet.</div>'}</div>
    <div class="card"><h3>Jobs</h3>${jobTable(familyJobs.slice(0, 10))}</div>`;
  bindMachinePicker(agents);
  bindJobRows();
  $('#vm-start')?.addEventListener('click', () => vmAction('start'));
  $('#vm-stop')?.addEventListener('click', () => vmAction('stop'));
}

async function vmAction(action) {
  if (action === 'stop' && !confirm('Deallocate/stop the training VM?')) return;
  try {
    await api(`/vm/${action}`, { method: 'POST', body: action === 'stop' ? {} : undefined });
    toast(`VM ${action} requested`);
    setTimeout(route, 800);
  } catch (error) { toast(error.message, true); }
}

function jobTable(jobs) {
  if (!jobs.length) return '<div class="empty">No jobs in this mode yet.</div>';
  return `<table><thead><tr><th>Run</th><th>Recipe</th><th>Status</th><th>Target</th><th>Created</th><th>Duration</th></tr></thead><tbody>
    ${jobs.map((job) => `<tr class="clickable job-row" data-id="${job.id}"><td><strong>${esc(job.name)}</strong></td><td>${esc(job.type)}</td><td>${badge(job.status)}</td><td>${esc(job.targetAgentId || 'any')}</td><td>${fmtTime(job.createdAt)}</td><td>${fmtDuration(job.startedAt, job.endedAt)}</td></tr>`).join('')}
  </tbody></table>`;
}

function bindJobRows() {
  $$('.job-row').forEach((row) => row.addEventListener('click', () => { location.hash = `#/job/${row.dataset.id}`; }));
}

async function renderNew(main) {
  const { agents } = await api('/agents');
  const family = currentFamily();
  const modes = family === 'detection'
    ? [['train', 'Train'], ['finetune', 'Finetune'], ['evaluate', 'Evaluate'], ['predict', 'Predict'], ['tune', 'Tune']]
    : family === 'classification'
      ? [['train', 'Train'], ['finetune', 'Finetune'], ['evaluate', 'Evaluate'], ['predict', 'Predict'], ['xai', 'xAI'], ['tune', 'Tune']]
    : family === 'vlm' ? [['tune', 'LoRA Tune']] : [['train', 'Train'], ['evaluate', 'Evaluate'], ['predict', 'Predict']];
  main.innerHTML = `<h2 class="page-title">New ${familyLabel()} Run</h2><p class="page-sub">Choose the operation, target machine and device, then preview or queue it.</p>
    <div id="new-target">${machinePickerHtml(agents)}</div>
    <div class="seg" id="mode-seg">${modes.map(([value, label], index) => `<button data-mode="${value}" class="${index === 0 ? 'active' : ''}">${label}</button>`).join('')}</div>
    <div id="mode-form"></div>`;
  bindMachinePicker(agents);
  const show = (mode) => {
    $('#mode-form').innerHTML = family === 'detection' ? detectionForm(mode)
      : family === 'classification' ? classificationForm(mode)
        : family === 'video' ? videoForm(mode) : vlmForm();
    bindRunForm();
  };
  show(modes[0][0]);
  $$('#mode-seg button').forEach((button) => button.addEventListener('click', () => {
    $$('#mode-seg button').forEach((item) => item.classList.toggle('active', item === button));
    show(button.dataset.mode);
  }));
}

async function renderExport(main) {
  const { agents } = await api('/agents');
  const family = currentFamily();
  const detection = family === 'detection';
  if (family === 'vlm') {
    main.innerHTML = `<h2 class="page-title">Merge VLM adapter</h2><p class="page-sub">Merge a Trainer LoRA adapter into its base model for standalone deployment.</p>
      ${machinePickerHtml(agents)}
      <form id="run-form" data-type="vlm_merge">
        ${commonRunFields(`vlm_merged_${new Date().toISOString().slice(0, 10).replaceAll('-', '_')}`)}
        <div class="card"><h3>Merge source</h3>
          ${field('model', 'Base model', { placeholder: 'organization/model-name', required: true })}
          ${field('adapter', 'LoRA adapter directory', { placeholder: '/runs/vlm_lora/adapter', required: true })}
          <label class="field"><span>Compute dtype</span><select name="dtype"><option value="auto">Auto</option><option value="bfloat16">bfloat16</option><option value="float16">float16</option><option value="float32">float32</option></select></label>
          <label class="inline-check"><input type="checkbox" name="trustRemoteCode" /> Allow model repository custom code</label>
        </div>${advancedFields()}${runActions('merge')}
      </form>`;
    bindMachinePicker(agents); bindRunForm(); return;
  }
  const image = family === 'classification';
  const prefix = detection ? 'detect' : image ? 'classifier' : 'video';
  main.innerHTML = `<h2 class="page-title">Export</h2><p class="page-sub">${detection ? 'Build an ONNX or TensorRT detector artifact.' : image ? 'Build image classifier ONNX and its class/tensor contract.' : 'Build video/action classifier ONNX and its clip/tensor contract.'}</p>
    ${machinePickerHtml(agents)}
    <form id="run-form" data-type="${prefix}_export">
      ${commonRunFields(`${prefix}_export_${new Date().toISOString().slice(0, 10).replaceAll('-', '_')}`)}
      <div class="card"><h3>Export source</h3>
        ${field('checkpoint', 'Checkpoint', { placeholder: detection ? '/runs/experiment/weights/best.pt' : image ? '/runs/classifier/best.pt' : '/runs/video_action/best.pt', required: true })}
        ${detection ? `<div class="grid2"><label class="field"><span>Format</span><select name="format"><option value="onnx">ONNX</option><option value="tensorrt">TensorRT</option></select></label>${field('imgsz', 'Image size', { value: '640', type: 'number', min: 32 })}</div><label class="inline-check"><input type="checkbox" name="half" /> FP16 TensorRT</label>` : `<p class="muted">Exports dynamic batch input <code>${image ? 'images' : 'clips'}</code> and output <code>logits</code>, plus a JSON deployment contract.</p>`}
      </div>${advancedFields()}${runActions('export')}
    </form>`;
  bindMachinePicker(agents);
  bindRunForm();
}

async function renderData(main) {
  const [{ inventory }, { agents }] = await Promise.all([api('/inventory'), api('/agents')]);
  const family = currentFamily();
  const detection = family === 'detection';
  const trees = {
    detection: `dataset.toml\nimages/train/*.jpg\nimages/val/*.jpg\nlabels/train/*.txt\nlabels/val/*.txt`,
    classification: `dataset/\n  train/<class>/*.jpg\n  val/<class>/*.jpg\n\n# or compact\ndataset/<class>/*.jpg`,
    video: `dataset/\n  train/<action>/*.mp4\n  val/<action>/*.mp4\n\n# or compact\ndataset/<action>/*.mp4`,
    vlm: `vlm/\n  train.jsonl\n  images/*.jpg\n\n# JSONL row\n{"image":"images/1.jpg","prompt":"...","answer":"..."}`,
  };
  const notes = {
    detection: 'The TOML maps train/val image sources and numeric class names. Labels use normalized YOLO cls cx cy w h rows.',
    classification: 'Folders define image class names. Compact datasets receive a deterministic per-class validation split.',
    video: 'Folders define action classes. Trainer samples a fixed number of frames and processes them with the selected temporal architecture plugin.',
    vlm: 'Each JSONL row points to a local image and contains the user prompt and assistant answer. Chat-style messages are also accepted.',
  };
  main.innerHTML = `<h2 class="page-title">Data</h2><p class="page-sub">${familyLabel()} dataset contract and machine inventory</p>
    <div class="card"><h3>${familyLabel()} layout</h3><div class="cmd-box">${esc(trees[family])}</div>
      <p class="muted">${notes[family]}</p></div>
    <div class="card"><h3>Compute and storage</h3>${machineList(agents).length ? `<div class="compute-grid">${machineList(agents).map(machineCardHtml).join('')}</div>` : '<div class="empty">No compute target is configured.</div>'}</div>
    <div class="card"><h3>Checkpoints</h3>${Object.keys(inventory || {}).length ? `<table><thead><tr><th>Agent</th><th>Path</th></tr></thead><tbody>${Object.entries(inventory).flatMap(([agentId, value]) => (value.checkpoints || []).map((checkpoint) => `<tr><td>${esc(agentId)}</td><td><code>${esc(checkpoint)}</code></td></tr>`)).join('')}</tbody></table>` : '<div class="empty">The agent has not published checkpoint inventory yet.</div>'}</div>`;
}

function fmtBytes(value) {
  const bytes = Number(value || 0);
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  return `${(bytes / 1024 ** 3).toFixed(1)} GiB`;
}

async function renderModels(main) {
  main.innerHTML = `<h2 class="page-title">Model registry</h2><p class="page-sub">Loading installed Trainer architecture plugins…</p>`;
  const payload = await api('/architectures');
  const task = FAMILIES[currentFamily()].task;
  const models = (payload.architectures || []).filter((model) => model.task === task);
  const taskInfo = (payload.tasks || []).find((item) => item.name === task) || {};
  const families = [...new Set(models.map((model) => model.family).filter(Boolean))].sort();
  const saved = localStorage.getItem(`trainer.architecture.${currentFamily()}`);
  let activeName = models.some((model) => model.name === saved) ? saved : taskInfo.defaultArchitecture || models[0]?.name;

  main.innerHTML = `<div class="model-registry-head"><div><span class="control-kicker">${esc(familyLabel())}</span><h2 class="page-title">Model registry</h2>
      <p class="page-sub">The registry is generated from the Python plugins installed on this control plane, including external entry-point plugins.</p></div>
      <div class="row"><a class="btn primary" href="#/new">New run</a><a class="btn" href="#/evidence">Open xAI</a></div></div>
    ${payload.warning ? `<div class="warn-banner">Registry warning: ${esc(payload.warning)}</div>` : ''}
    <section class="model-registry-kpis">
      <div><span>Architectures</span><strong>${models.length}</strong><small>installed for this family</small></div>
      <div><span>Families</span><strong>${families.length}</strong><small>architecture groups</small></div>
      <div><span>Actions</span><strong>${(taskInfo.actions || []).length}</strong><small>${esc((taskInfo.actions || []).join(' · ') || 'plugin-defined')}</small></div>
      <div><span>Default</span><strong class="registry-default">${esc(taskInfo.defaultArchitecture || '—')}</strong><small>used when no override is supplied</small></div>
    </section>
    <section class="card architecture-flow-card">
      <div class="row spread"><div><span class="control-kicker">Interactive model flow</span><h3 id="flow-title"></h3></div><span class="flow-note">Select a stage to inspect its contract</span></div>
      <div id="architecture-flow"></div>
      <div class="flow-explanation" id="flow-explanation"></div>
    </section>
    <section class="model-registry-tools card">
      <label class="field"><span>Find a model</span><input id="registry-query" placeholder="architecture, inputs, outputs, strategy…" /></label>
      <label class="field"><span>Family</span><select id="registry-family"><option value="all">All families</option>${families.map((family) => `<option value="${esc(family)}">${esc(family)}</option>`).join('')}</select></label>
    </section>
    <div class="row spread model-registry-count"><strong id="registry-count"></strong><span class="muted">Selecting a model also makes it the default in New Run.</span></div>
    <section id="model-registry-grid" class="model-registry-grid"></section>`;

  const activeModel = () => models.find((model) => model.name === activeName) || models[0];
  const selectStage = (index) => {
    const model = activeModel();
    if (!model) return;
    const stages = model.stages || [];
    const selected = Math.max(0, Math.min(index, stages.length - 1));
    $$('#architecture-flow .flow-stage-button').forEach((button, buttonIndex) => {
      button.classList.toggle('active', buttonIndex === selected);
      button.setAttribute('aria-pressed', String(buttonIndex === selected));
    });
    const descriptions = [model.inputContract, model.trainingStrategy, model.outputContract];
    const position = stages.length <= 1 ? 0 : selected / (stages.length - 1);
    const detail = position < 0.34 ? descriptions[0] : position > 0.66 ? descriptions[2] : descriptions[1];
    $('#flow-explanation').innerHTML = `<span>${esc(stages[selected] || 'stage')}</span><p>${esc(detail)}</p>`;
  };
  const drawFlow = () => {
    const model = activeModel();
    if (!model) {
      $('#architecture-flow').innerHTML = '<div class="empty">No architecture plugin is registered for this family.</div>';
      $('#flow-title').textContent = 'No installed model';
      return;
    }
    $('#flow-title').textContent = `${model.name} · ${model.family}`;
    $('#architecture-flow').innerHTML = `<div class="architecture-flow">${(model.stages || []).map((stage, index) => `<button type="button" class="flow-stage-button" data-stage-index="${index}"><span>${index + 1}</span><strong>${esc(stage)}</strong></button>`).join('<i aria-hidden="true">→</i>')}</div>`;
    $$('#architecture-flow .flow-stage-button').forEach((button) => button.addEventListener('click', () => selectStage(Number(button.dataset.stageIndex))));
    selectStage(0);
  };
  const drawRegistry = () => {
    const query = $('#registry-query').value.trim().toLowerCase();
    const family = $('#registry-family').value;
    const filtered = models.filter((model) => {
      if (family !== 'all' && model.family !== family) return false;
      if (!query) return true;
      return [model.name, model.family, model.description, model.inputContract, model.outputContract, model.trainingStrategy, ...(model.aliases || [])]
        .some((value) => String(value || '').toLowerCase().includes(query));
    });
    $('#registry-count').textContent = `${filtered.length} of ${models.length} architecture plugins`;
    $('#model-registry-grid').innerHTML = filtered.map((model) => `<article class="model-registry-card ${model.name === activeName ? 'selected' : ''}">
      <header><div><span class="model-family">${esc(model.family)}</span><h3>${esc(model.name)}</h3><code>${esc(model.task)}</code></div>${model.name === taskInfo.defaultArchitecture ? '<span class="model-kind">Default</span>' : ''}</header>
      <p class="model-description">${esc(model.description)}</p>
      <dl class="model-specs"><div><dt>Input</dt><dd>${esc(model.inputContract)}</dd></div><div><dt>Output</dt><dd>${esc(model.outputContract)}</dd></div></dl>
      <section><span>Training strategy</span><p>${esc(model.trainingStrategy)}</p></section>
      ${model.aliases?.length ? `<section><span>Aliases</span><p><code>${esc(model.aliases.join(', '))}</code></p></section>` : ''}
      <footer><button type="button" class="btn small primary use-architecture" data-architecture="${esc(model.name)}">Use in New Run</button><button type="button" class="btn small inspect-architecture" data-architecture="${esc(model.name)}">Inspect flow</button><a class="btn small" href="#/evidence">xAI</a></footer>
    </article>`).join('') || '<div class="empty">No architectures match these filters.</div>';
    $$('.inspect-architecture').forEach((button) => button.addEventListener('click', () => {
      activeName = button.dataset.architecture; drawFlow(); drawRegistry(); $('.architecture-flow-card')?.scrollIntoView({ behavior: 'smooth' });
    }));
    $$('.use-architecture').forEach((button) => button.addEventListener('click', () => {
      localStorage.setItem(`trainer.architecture.${currentFamily()}`, button.dataset.architecture);
      location.hash = '#/new';
    }));
  };
  $('#registry-query').addEventListener('input', drawRegistry);
  $('#registry-family').addEventListener('change', drawRegistry);
  drawFlow();
  drawRegistry();
}

function bindEvidenceImageRetry() {
  $$('.evidence-preview img').forEach((image) => image.addEventListener('error', () => {
    const retries = Number(image.dataset.retries || 0);
    if (retries >= 1) {
      image.closest('.evidence-preview')?.classList.add('failed');
      return;
    }
    image.dataset.retries = String(retries + 1);
    setTimeout(() => { image.src = `${image.dataset.source}${image.dataset.source.includes('?') ? '&' : '?'}retry=${Date.now()}`; }, 450);
  }));
}

const classificationXaiImageCache = new Map();

function evidenceFileUrl(relativePath) {
  return `/api/evidence/file?path=${encodeURIComponent(relativePath)}`;
}

function loadEvidenceCanvasImage(url, retry = 0) {
  const key = `${url}:${retry}`;
  if (!classificationXaiImageCache.has(key)) classificationXaiImageCache.set(key, new Promise((resolve, reject) => {
    const source = new Image();
    source.decoding = 'async';
    source.onload = () => resolve(source);
    source.onerror = () => {
      if (retry < 1) loadEvidenceCanvasImage(`${url}${url.includes('?') ? '&' : '?'}retry=${Date.now()}`, retry + 1).then(resolve, reject);
      else reject(new Error(`Unable to load ${url}`));
    };
    source.src = url;
  }));
  return classificationXaiImageCache.get(key);
}

async function drawClassificationGradcam(canvas, originalUrl, mapUrl, intensity) {
  const [original, map] = await Promise.all([
    loadEvidenceCanvasImage(originalUrl), loadEvidenceCanvasImage(mapUrl),
  ]);
  const scale = Math.min(1, 720 / Math.max(original.naturalWidth, original.naturalHeight));
  const width = Math.max(1, Math.round(original.naturalWidth * scale));
  const height = Math.max(1, Math.round(original.naturalHeight * scale));
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  context.drawImage(original, 0, 0, width, height);
  const colorMap = document.createElement('canvas');
  colorMap.width = map.naturalWidth;
  colorMap.height = map.naturalHeight;
  const mapContext = colorMap.getContext('2d', { willReadFrequently: true });
  mapContext.drawImage(map, 0, 0);
  const pixels = mapContext.getImageData(0, 0, colorMap.width, colorMap.height);
  for (let offset = 0; offset < pixels.data.length; offset += 4) {
    const signed = (pixels.data[offset] / 127.5) - 1;
    const strength = Math.abs(signed) < 0.025 ? 0 : Math.pow(Math.abs(signed), 0.72);
    const color = signed >= 0 ? [238, 82, 74] : [54, 126, 194];
    pixels.data[offset] = color[0];
    pixels.data[offset + 1] = color[1];
    pixels.data[offset + 2] = color[2];
    pixels.data[offset + 3] = Math.round(255 * 0.68 * strength * intensity);
  }
  mapContext.putImageData(pixels, 0, 0);
  context.imageSmoothingEnabled = true;
  context.drawImage(colorMap, 0, 0, width, height);
}

function renderClassificationXaiViewer(container, manifestItem, manifest) {
  const images = manifest.images || [];
  if (!images.length || !manifestItem.url) {
    container.innerHTML = '<div class="empty">No locally previewable ConvNeXt xAI manifest is available yet.</div>';
    return;
  }
  const runRoot = manifestItem.relativePath.replace(/\/xai\/manifest\.json$/, '');
  container.innerHTML = `<section class="card classification-xai-live">
    <div class="model-registry-head"><div><span class="control-kicker">Live signed activation</span><h3>ConvNeXt xAI viewer</h3>
      <p class="muted">Red supports the selected class; blue opposes it. This uses the retained signed map, not a synthetic browser heatmap.</p></div>
      <a class="btn small" id="classification-xai-gallery" target="_blank" rel="noopener">Open gallery ↗</a></div>
    <div class="classification-xai-controls">
      <label class="field"><span>Image</span><select id="classification-xai-image">${images.map((item, index) => `<option value="${index}">${esc(item.token)} · ${esc(item.predicted)} · ${(Number(item.confidence) * 100).toFixed(1)}%</option>`).join('')}</select></label>
      <label class="field"><span>Activation intensity <strong id="classification-xai-intensity-value">72%</strong></span><input id="classification-xai-intensity" type="range" min="0" max="100" value="72" /></label>
    </div>
    <div class="classification-xai-stage">
      <figure class="classification-xai-map"><canvas id="classification-xai-canvas"></canvas><figcaption id="classification-xai-caption"></figcaption></figure>
      <figure><img id="classification-xai-activations" alt="ConvNeXt feature activations" /><figcaption>Highest-energy feature channels at every captured ConvNeXt stage</figcaption></figure>
    </div><div id="classification-xai-scores" class="classification-xai-scores"></div>
  </section>`;
  const select = $('#classification-xai-image', container);
  const slider = $('#classification-xai-intensity', container);
  const canvas = $('#classification-xai-canvas', container);
  const activationImage = $('#classification-xai-activations', container);
  const artifact = (relative) => evidenceFileUrl(`${runRoot}/${relative}`);
  let renderToken = 0;
  const draw = async () => {
    const item = images[Number(select.value) || 0];
    const token = ++renderToken;
    $('#classification-xai-intensity-value', container).textContent = `${slider.value}%`;
    $('#classification-xai-caption', container).textContent = `${item.target} · target ${(Number(item.target_score) * 100).toFixed(1)}% · ${item.layer}`;
    $('#classification-xai-scores', container).innerHTML = (item.topk || []).map((score) => `<span><strong>${esc(score.class)}</strong> ${(Number(score.confidence) * 100).toFixed(1)}%</span>`).join('');
    activationImage.src = artifact(item.activations);
    $('#classification-xai-gallery', container).href = evidenceFileUrl(`${runRoot}/reports/xai.html`);
    canvas.closest('figure').classList.add('loading');
    try {
      await drawClassificationGradcam(canvas, artifact(item.input), artifact(item.signed_map), Number(slider.value) / 100);
      if (token === renderToken) canvas.closest('figure').classList.remove('loading', 'failed');
    } catch {
      if (token === renderToken) canvas.closest('figure').classList.add('failed');
    }
  };
  select.addEventListener('change', draw);
  slider.addEventListener('input', draw);
  draw();
}

async function loadClassificationXaiViewer(container) {
  if (!container) return;
  container.innerHTML = '<div class="empty">Loading ConvNeXt xAI manifest…</div>';
  try {
    const catalog = await api('/evidence?family=classification&kind=document&q=xai%2Fmanifest.json&limit=20');
    const manifestItem = (catalog.items || []).find((item) => item.url && item.relativePath.endsWith('/xai/manifest.json'));
    if (!manifestItem) {
      container.innerHTML = '<div class="empty">No ConvNeXt xAI run yet. Open New Run → xAI to generate signed maps and activations.</div>';
      return;
    }
    const manifest = await api(manifestItem.url.replace(/^\/api/, ''));
    renderClassificationXaiViewer(container, manifestItem, manifest);
  } catch (error) {
    container.innerHTML = `<div class="warn-banner">${esc(error.message || 'Unable to load ConvNeXt xAI manifest')}</div>`;
  }
}

async function renderEvidence(main) {
  const family = currentFamily();
  main.innerHTML = `<div class="model-registry-head"><div><span class="control-kicker">${esc(familyLabel())}</span><h2 class="page-title">xAI and evidence</h2>
      <p class="page-sub">Browse reports, prediction visuals, augmentation previews, saliency maps, and mask/background diagnostics produced by Trainer runs.</p></div>
      <div class="row"><a class="btn" href="#/models">Model registry</a><button class="btn primary" id="evidence-refresh">Refresh</button></div></div>
    <section class="evidence-tools card"><label class="field"><span>Find evidence</span><input id="evidence-query" placeholder="run, checkpoint, class, mask…" /></label>
      <label class="field"><span>Artifact type</span><select id="evidence-kind"><option value="all">All evidence</option><option value="xai">xAI / saliency</option><option value="mask-diagnostic">Mask/background diagnostics</option><option value="augmentation">Augmentation previews</option><option value="visual">Prediction visuals</option><option value="report">Reports</option><option value="metrics">Metrics</option><option value="predictions">Predictions</option><option value="model">Models</option><option value="document">Other documents</option></select></label></section>
    ${family === 'classification' ? '<div id="classification-xai-viewer"></div>' : ''}
    <div id="evidence-status" class="evidence-status"><span class="muted">Loading run evidence…</span></div>
    <section id="evidence-grid" class="evidence-grid"></section>`;

  let timer = null;
  const load = async () => {
    $('#evidence-status').innerHTML = '<span class="muted">Loading run evidence…</span>';
    try {
      const query = $('#evidence-query').value.trim();
      const kind = $('#evidence-kind').value;
      const value = await api(`/evidence?family=${encodeURIComponent(family)}&kind=${encodeURIComponent(kind)}&q=${encodeURIComponent(query)}&limit=300`);
      const items = value.items || [];
      const kinds = new Set(items.map((item) => item.kind));
      $('#evidence-status').innerHTML = `<div><strong>${items.length}</strong> artifact${items.length === 1 ? '' : 's'} · scanned ${Number(value.scanned || 0).toLocaleString()} files</div><span class="muted">${kinds.size} evidence type${kinds.size === 1 ? '' : 's'} · ${esc(value.root)}</span>`;
      $('#evidence-grid').innerHTML = items.map((item) => `<article class="evidence-card ${esc(item.kind)}">
        ${item.url && ['visual', 'xai', 'mask-diagnostic', 'augmentation'].includes(item.kind) ? `<a class="evidence-preview" href="${esc(item.url)}" target="_blank" rel="noopener"><img loading="lazy" data-source="${esc(item.url)}" src="${esc(item.url)}" alt="${esc(item.name)}" /><span>Preview unavailable · retry or open artifact</span></a>` : ''}
        <header><span class="model-family">${esc(item.kind.replaceAll('-', ' '))}</span><h3>${esc(item.name)}</h3></header>
        <code class="evidence-path">${esc(item.relativePath)}</code>
        <dl class="model-specs"><div><dt>Machine</dt><dd>${esc(item.agentId || 'control-plane')}${item.remote ? ' · remote' : ''}</dd></div><div><dt>Size</dt><dd>${fmtBytes(item.size)}</dd></div></dl>
        <footer>${item.url ? `<a class="btn small primary" href="${esc(item.url)}" target="_blank" rel="noopener">${item.kind === 'report' ? 'Open report' : 'Open artifact'} ↗</a>` : '<span class="model-no-evidence">Indexed on remote agent; open from that machine.</span>'}</footer>
      </article>`).join('') || '<div class="empty">No evidence artifacts match these filters yet. Run predict, evaluate/report, or augmentation preview first.</div>';
      bindEvidenceImageRetry();
    } catch (error) {
      $('#evidence-status').innerHTML = `<div class="warn-banner">${esc(error.message || 'Unable to load evidence')} <button type="button" class="btn small" id="evidence-retry">Retry</button></div>`;
      $('#evidence-retry')?.addEventListener('click', load);
    }
  };
  $('#evidence-refresh').addEventListener('click', load);
  $('#evidence-kind').addEventListener('change', load);
  $('#evidence-query').addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(load, 250); });
  if (family === 'classification') await loadClassificationXaiViewer($('#classification-xai-viewer'));
  await load();
}

async function renderHistory(main) {
  const { jobs } = await api('/jobs?limit=500');
  const familyJobs = jobs.filter(familyMatches);
  main.innerHTML = `<h2 class="page-title">History</h2><p class="page-sub">${familyLabel()} jobs queued by all operators</p>
    <div class="row filter-bar"><label class="field"><span>Status</span><select id="history-status"><option value="">All</option><option>queued</option><option>running</option><option>completed</option><option>failed</option><option>cancelled</option></select></label><span class="count" id="history-count"></span></div>
    <div class="card" id="history-table"></div>`;
  const draw = () => {
    const status = $('#history-status').value;
    const filtered = familyJobs.filter((job) => !status || job.status === status);
    $('#history-table').innerHTML = jobTable(filtered);
    $('#history-count').textContent = `${filtered.length} job${filtered.length === 1 ? '' : 's'}`;
    bindJobRows();
  };
  $('#history-status').addEventListener('change', draw);
  draw();
}

async function renderJob(main, id) {
  const { job } = await api(`/jobs/${id}`);
  const terminal = ['completed', 'failed', 'cancelled'].includes(job.status);
  main.innerHTML = `<div class="row spread job-header"><div><h2 class="page-title">${esc(job.name)}</h2><p class="page-sub">${esc(job.type)} · ${esc(job.id)}</p></div>
    <div class="row job-actions">${badge(job.status)}${!terminal ? '<button class="btn danger small" id="cancel-job">Cancel</button>' : '<button class="btn small" id="retry-job">Retry</button>'}</div></div>
    ${job.error ? `<div class="warn-banner">${esc(job.error)}</div>` : ''}
    <div class="grid2"><div class="card"><h3>Run</h3><dl class="kv"><dt>Family</dt><dd>${esc(jobFamily(job))}</dd><dt>Created by</dt><dd>${esc(job.createdBy || '—')}</dd><dt>Target</dt><dd>${esc(job.targetAgentId || 'any')}</dd><dt>Claimed by</dt><dd>${esc(job.agentId || '—')}</dd><dt>Device</dt><dd>${esc(job.params?.device || '—')}</dd><dt>Created</dt><dd>${fmtTime(job.createdAt)}</dd><dt>Duration</dt><dd>${fmtDuration(job.startedAt, job.endedAt)}</dd></dl></div>
      <div class="card"><h3>Artifacts</h3>${job.artifacts?.length ? `<div class="kv">${job.artifacts.map((artifact, index) => `<dt>Artifact ${index + 1}</dt><dd>${esc(artifact)}</dd>`).join('')}</div>` : '<p class="muted">No artifacts published yet.</p>'}</div></div>
    <div class="card"><h3>Steps</h3><div class="steps">${job.steps.map((step) => `<div class="step"><span class="title">${esc(step.title)}</span>${badge(step.status)}</div>`).join('')}</div>
      ${job.commands.map((command) => `<div class="cmd-title">${esc(command.title)}</div><div class="cmd-box">${esc(commandText(command))}</div>`).join('')}</div>
    <div class="card"><h3>Live log</h3><div class="log-box" id="job-log">${esc((job.logTail || []).join('\n') || 'Waiting for agent…')}</div></div>`;
  const log = $('#job-log');
  if (log) log.scrollTop = log.scrollHeight;
  $('#cancel-job')?.addEventListener('click', async () => {
    if (!confirm(`Cancel ${job.name}?`)) return;
    await api(`/jobs/${job.id}/cancel`, { method: 'POST' }); toast('Cancellation requested'); renderJob(main, id);
  });
  $('#retry-job')?.addEventListener('click', async () => {
    const result = await api(`/jobs/${job.id}/retry`, { method: 'POST' }); toast('Retry queued'); location.hash = `#/job/${result.job.id}`;
  });
  if (!terminal && location.hash === `#/job/${id}`) jobTimer = setTimeout(() => renderJob(main, id).catch(() => {}), 2500);
}

function showLogin() {
  $('#login').classList.remove('hidden');
  $('#shell').classList.add('hidden');
}

async function startApp() {
  CONFIG = await api('/config');
  $('#shell').classList.remove('hidden');
  $('#login').classList.add('hidden');
  $('#signed-in').textContent = CURRENT_USER ? `Signed in as ${CURRENT_USER}` : '';
  applyFamilyChrome();
  window.addEventListener('hashchange', route);
  $('#family-switch').addEventListener('click', (event) => {
    const button = event.target.closest('button[data-family]');
    if (button) setFamily(button.dataset.family);
  });
  route();
  pollVm();
  setInterval(pollVm, 20_000);
}

async function pollVm() {
  try {
    const [{ driver }, { agents }] = await Promise.all([api('/vm'), api('/agents')]);
    const machines = machineList(agents);
    const online = machines.filter((machine) => machine.online).length;
    const dot = $('#vm-pill .dot');
    if (driver.configured) {
      dot.className = `dot ${driver.state || 'unknown'}`;
      $('#vm-pill-text').textContent = `VM: ${driver.state} · ${online}/${machines.length} agents`;
    } else {
      dot.className = `dot ${online ? 'ok' : 'error'}`;
      $('#vm-pill-text').textContent = machines.length ? `${online}/${machines.length} machines online` : 'No machines yet';
    }
  } catch { /* best effort */ }
}

async function route() {
  clearTimeout(jobTimer);
  jobTimer = null;
  const hash = location.hash || '#/dashboard';
  const [, view, argument] = hash.split('/');
  $$('.sidebar nav a').forEach((link) => link.classList.toggle('active', link.dataset.nav === view));
  const main = $('#main');
  try {
    if (view === 'new') return await renderNew(main);
    if (view === 'export') return await renderExport(main);
    if (view === 'models') return await renderModels(main);
    if (view === 'evidence') return await renderEvidence(main);
    if (view === 'data') return await renderData(main);
    if (view === 'history') return await renderHistory(main);
    if (view === 'job') return await renderJob(main, argument);
    return await renderDashboard(main);
  } catch (error) {
    main.innerHTML = `<h2 class="page-title">Could not load</h2><div class="warn-banner">${esc(error.message)}</div>`;
  }
}

$('#login-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    const result = await api('/login', { method: 'POST', body: {
      username: $('#login-username').value.trim(), password: $('#login-password').value,
    } });
    CURRENT_USER = result.username;
    $('#login-error').textContent = '';
    await startApp();
  } catch {
    $('#login-error').textContent = 'Wrong username or password';
  }
});

$('#logout').addEventListener('click', async () => {
  await api('/logout', { method: 'POST' });
  location.reload();
});

(async () => {
  const me = await api('/me').catch(() => ({ authed: false }));
  if (!me.authed) return showLogin();
  CURRENT_USER = me.username;
  await startApp();
})();
