/* Trainer Automation control panel. */

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

let CONFIG = null;
let CURRENT_USER = null;
let jobTimer = null;

const FAMILY_KEY = 'trainer.family';
const TARGET_KEY = 'trainer.targetAgent';
const XAI_SAMPLE_JOB_KEY = 'trainer.classification.xaiSampleJob';
const FAMILIES = {
  detection: { label: 'Detection', task: 'detection', prefix: 'detect_', data: 'YOLO labels + dataset TOML', export: 'Detector ONNX / TensorRT' },
  classification: { label: 'Image Classification', task: 'classification', prefix: 'classifier_', data: 'COCO multi-label or class-folder datasets', export: 'Image classifier ONNX / TensorRT + target contract' },
  video: { label: 'Video / Action Classification', task: 'video-classification', prefix: 'video_', data: 'Class-folder video datasets', export: 'Video classifier ONNX / TensorRT + clip contract' },
  vlm: { label: 'VLM Tuning', task: 'vlm', prefix: 'vlm_', data: 'Image-text JSONL datasets', export: 'VLM ONNX / TensorRT forward graph, portable LoRA adapter, or merged model' },
};

const DATASET_RECOMMENDATIONS = {
  detection: [
    ['COCO 2017', 'Current baseline', 'Use val2017 for end-to-end smoke tests and train2017 for real training. Keeps boxes, classes, and COCO evaluation.'],
    ['Custom COCO / YOLO', 'Production data', 'Profile the exported annotations first, then train through the existing detector framework.'],
  ],
  classification: [
    ['COCO 2017 Instances', 'Current baseline', 'Treat every category present in one scene as a simultaneous positive image label.'],
    ['Open Images', 'Scale-up option', 'Use when broader image-level labels and substantially more classes are needed after the COCO flow is stable.'],
  ],
  video: [
    ['UCF101', 'First smoke dataset', 'Small enough to validate decoding, clip sampling, training, evaluation, prediction, and export.'],
    ['Kinetics-400', 'Scale-up option', 'Use after UCF101 when the video pipeline and VM storage/throughput are proven.'],
  ],
  vlm: [
    ['COCO Captions', 'First grounding dataset', 'Good for validating image-text ingestion, batching, LoRA tuning, and held-out evaluation.'],
    ['LLaVA-Instruct-150K', 'Instruction tuning', 'Use for conversational visual instruction tuning after the base smoke flow works.'],
  ],
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
  const view = (location.hash || '#/dashboard').split('/')[1];
  if (view === 'data') route();
  else if ((location.hash || '#/dashboard') !== '#/dashboard') location.hash = '#/dashboard';
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
    train: ['Train classifier', 'Train single-label class folders or multi-label COCO images with the selected architecture plugin.'],
    finetune: ['Finetune classifier', 'Transfer a checkpoint to the current classes and optionally freeze feature stages.'],
    evaluate: ['Evaluate classifier', 'Compute full metrics, tune per-class thresholds, and build the evaluation report.'],
    predict: ['Classify images', 'Write top-k predictions and apply optional per-class confidence thresholds.'],
    xai: ['Explain classifier', 'Generate signed Grad-CAM overlays and ConvNeXt stage activations for selected test samples.'],
    tune: ['Tune classifier', 'Search learning rate, regularization, dropout, and label smoothing.'],
  };
  let body = '';
  if (mode === 'train') body = `<div class="card"><h3>Classification dataset</h3>
    ${field('data', 'Dataset root or COCO annotations', { placeholder: '/data/coco', required: true })}
    ${architectureField('arch', 'Architecture plugin', { value: 'convnext-tiny' })}
    <div class="pipeline"><code>annotations/instances_*.json + train2017/</code><span>or</span><code>train/class_name/*.jpg</code></div>
    <p class="muted">COCO is detected automatically and trained as multi-label category presence. Class folders retain single-label softmax training.</p></div>
    <div class="card"><h3>Training parameters</h3><div class="adv-grid">
      ${field('epochs', 'Epochs', { value: '30', type: 'number', min: 1 })}
      ${field('batch', 'Batch size', { value: '8', type: 'number', min: 1 })}
      ${field('imgsz', 'Image size', { value: '160', type: 'number', min: 32 })}
      ${field('patience', 'Early-stop patience', { value: '20', type: 'number', min: 1 })}
      ${field('accumulate', 'Gradient accumulation', { value: '1', type: 'number', min: 1 })}
    </div>${field('resume', 'Resume checkpoint (optional)', { placeholder: '/runs/classifier/weights/last.pt' })}</div>`;
  if (mode === 'finetune') body = `<div class="card"><h3>Checkpoint and dataset</h3>
    ${field('checkpoint', 'Source checkpoint', { placeholder: '/runs/classifier/weights/best.pt', required: true })}
    ${field('data', 'Dataset root or COCO annotations', { placeholder: '/data/coco', required: true })}
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
    ${field('data', 'Labelled dataset root or COCO annotations', { placeholder: '/data/coco', required: true })}
    ${field('batch', 'Batch size', { value: '32', type: 'number', min: 1 })}
    <p class="muted">Produces metrics JSON, per-image CSV, optimized class thresholds, an HTML report, and a confusion matrix for single-label runs.</p></div>`;
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
    </div><p class="muted">Red supports the selected class; blue opposes it. The result keeps the prediction and highest-energy feature channels together for each selected sample.</p></div>`;
  if (mode === 'tune') body = `<div class="card"><h3>Tuning inputs</h3>
    ${field('data', 'Dataset root or COCO annotations', { placeholder: '/data/coco', required: true })}
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

function vlmForm(mode) {
  const stamp = new Date().toISOString().slice(0, 10).replaceAll('-', '_');
  if (mode === 'evaluate') {
    return `<h2 class="page-title">Evaluate vision-language model</h2><p class="page-sub">Measure teacher-forced loss, perplexity, and answer-token accuracy on one explicit image-text dataset.</p>
      <form id="run-form" data-type="vlm_evaluate">
        ${commonRunFields(`vlm_eval_${stamp}`)}
        <div class="card"><h3>Evaluation inputs</h3>
          ${field('model', 'Base model', { placeholder: 'organization/model-name', required: true })}
          ${field('adapter', 'LoRA adapter directory (optional)', { placeholder: '/runs/vlm_lora/adapter' })}
          ${field('data', 'Evaluation JSONL', { placeholder: '/data/vlm/validation.jsonl', required: true })}
          <div class="adv-grid">
            ${field('batch', 'Batch size', { value: '1', type: 'number', min: 1 })}
            ${field('maxLength', 'Max tokens', { value: '1024', type: 'number', min: 32 })}
            ${field('maxSamples', 'Maximum samples', { value: '1000', type: 'number', min: 1 })}
          </div>
          <label class="field"><span>Compute dtype</span><select name="dtype"><option value="auto">Auto</option><option value="bfloat16">bfloat16</option><option value="float16">float16</option><option value="float32">float32</option></select></label>
          <label class="inline-check"><input type="checkbox" name="trustRemoteCode" /> Allow model repository custom code</label>
          <p class="muted">Evaluation does not augment the validation set. Results are comparable only when the JSONL and requested split are identical.</p>
        </div>${advancedFields()}${runActions('VLM evaluation')}
      </form>`;
  }
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

function exportActions({ merge = false } = {}) {
  return `<div class="row export-actions"><button type="button" class="btn" id="preview-btn">Preview ONNX command</button>
    <button type="submit" class="btn primary" data-export-format="onnx">Export ONNX</button>
    <button type="submit" class="btn primary" data-export-format="tensorrt">Export TensorRT</button>
    ${merge ? '<button type="submit" class="btn" data-job-type="vlm_merge">Merge adapter only</button>' : ''}</div>
    <div id="preview-out" style="margin-top:18px"></div>`;
}

function formPayload(form, submitter = null) {
  const params = Object.fromEntries(new FormData(form));
  if (submitter?.dataset.exportFormat) params.format = submitter.dataset.exportFormat;
  const dryRun = params.dryRun === 'on';
  delete params.dryRun;
  for (const key of Object.keys(params)) if (params[key] === '') delete params[key];
  const targetAgentId = selectedTarget();
  if (targetAgentId) params.targetAgentId = targetAgentId;
  return { type: submitter?.dataset.jobType || form.dataset.type, params, dryRun };
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
      const { job } = await api('/jobs', { method: 'POST', body: formPayload(form, event.submitter) });
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
      <div class="card"><h3>xAI</h3><p class="muted">Inspect a test sample together with its prediction, confidence, explanation, and activations.</p><a class="btn" href="#/evidence">Open xAI</a></div>
    </div>
    <div class="card"><h3>Evaluations</h3><p class="muted">Compare checkpoints only when they were evaluated on the same dataset and split.</p><a class="btn" href="#/evaluations">Open evaluations</a></div>
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

function discoveredRunTable(runs) {
  if (!runs.length) return '<div class="empty">No filesystem or smoke-matrix runs discovered for this filter.</div>';
  return `<table><thead><tr><th>Run</th><th>Family / action</th><th>Status</th><th>Machine</th><th>Updated</th><th>Artifacts</th></tr></thead><tbody>
    ${runs.map((run) => `<tr><td><strong>${esc(run.name)}</strong><br><code>${esc(run.path)}</code></td>
      <td>${esc(FAMILIES[run.family]?.label || run.family || 'Unknown')} · ${esc(run.action || 'run')}${run.architecture ? `<br><span class="muted">${esc(run.architecture)}</span>` : ''}</td>
      <td>${badge(run.status || 'completed')}${run.error ? `<br><span class="error-text">${esc(run.error)}</span>` : ''}</td>
      <td>${esc(run.agentId || 'control-plane')}${run.remote ? ' · remote' : ''}${run.device ? `<br><span class="muted">${esc(run.device)}</span>` : ''}</td>
      <td>${fmtTime(run.updatedAt)}${run.seconds ? `<br><span class="muted">${Number(run.seconds).toFixed(1)}s</span>` : ''}</td>
      <td>${Number(run.files || 0).toLocaleString()} files · ${fmtBytes(run.bytes || 0)}${run.hasEvidence ? `<br><button type="button" class="btn small discovered-evidence" data-family="${esc(run.family)}">Open xAI</button>` : ''}</td></tr>`).join('')}
  </tbody></table>`;
}

function bindDiscoveredRuns() {
  $$('.discovered-evidence').forEach((button) => button.addEventListener('click', () => {
    if (Object.hasOwn(FAMILIES, button.dataset.family)) localStorage.setItem(FAMILY_KEY, button.dataset.family);
    applyFamilyChrome();
    location.hash = '#/evidence';
  }));
}

async function renderNew(main, initialMode = '') {
  const { agents } = await api('/agents');
  const family = currentFamily();
  const modes = family === 'detection'
    ? [['train', 'Train'], ['finetune', 'Finetune'], ['evaluate', 'Evaluate'], ['predict', 'Predict'], ['tune', 'Tune']]
    : family === 'classification'
      ? [['train', 'Train'], ['finetune', 'Finetune'], ['evaluate', 'Evaluate'], ['predict', 'Predict'], ['xai', 'xAI'], ['tune', 'Tune']]
    : family === 'vlm' ? [['tune', 'LoRA Tune'], ['evaluate', 'Evaluate']] : [['train', 'Train'], ['evaluate', 'Evaluate'], ['predict', 'Predict']];
  const selectedMode = modes.some(([value]) => value === initialMode) ? initialMode : modes[0][0];
  main.innerHTML = `<h2 class="page-title">New ${familyLabel()} Run</h2><p class="page-sub">Choose the operation, target machine and device, then preview or queue it.</p>
    <div id="new-target">${machinePickerHtml(agents)}</div>
    <div class="seg" id="mode-seg">${modes.map(([value, label]) => `<button data-mode="${value}" class="${value === selectedMode ? 'active' : ''}">${label}</button>`).join('')}</div>
    <div id="mode-form"></div>`;
  bindMachinePicker(agents);
  const show = (mode) => {
    $('#mode-form').innerHTML = family === 'detection' ? detectionForm(mode)
      : family === 'classification' ? classificationForm(mode)
        : family === 'video' ? videoForm(mode) : vlmForm(mode);
    bindRunForm();
  };
  show(selectedMode);
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
    main.innerHTML = `<h2 class="page-title">Export VLM</h2><p class="page-sub">Export a multimodal forward graph to ONNX or TensorRT. A LoRA adapter can be merged before tracing.</p>
      ${machinePickerHtml(agents)}
      <form id="run-form" data-type="vlm_export">
        <input type="hidden" name="format" value="onnx" />
        ${commonRunFields(`vlm_export_${new Date().toISOString().slice(0, 10).replaceAll('-', '_')}`)}
        <div class="card"><h3>Export source</h3>
          ${field('model', 'Base model', { placeholder: 'organization/model-name', required: true })}
          ${field('adapter', 'LoRA adapter directory (optional)', { placeholder: '/runs/vlm_lora/adapter' })}
          ${field('sampleImage', 'Trace image (optional)', { placeholder: '/data/export/sample.jpg' })}
          ${field('prompt', 'Trace prompt', { value: 'Describe the image.' })}
          <label class="field"><span>Compute dtype</span><select name="dtype"><option value="auto">Auto</option><option value="bfloat16">bfloat16</option><option value="float16">float16</option><option value="float32">float32</option></select></label>
          <div class="adv-grid">
            ${field('imageSize', 'Fallback image size', { value: '384', type: 'number', min: 32 })}
            ${field('maxLength', 'Maximum processed tokens', { value: '4096', type: 'number', min: 32 })}
            ${field('opset', 'ONNX opset', { value: '18', type: 'number', min: 18 })}
            ${field('workspace', 'TensorRT workspace (GiB)', { value: '8', type: 'number', min: 1 })}
          </div>
          <label class="inline-check"><input type="checkbox" name="half" /> Build TensorRT engine with FP16 kernels (TensorRT 8–10)</label>
          <label class="inline-check"><input type="checkbox" name="trustRemoteCode" /> Allow model repository custom code</label>
          <p class="muted">TensorRT requires a CUDA target. VLM export traces fixed input shapes from this image and prompt, returns logits, and leaves autoregressive decoding to the serving runtime.</p>
        </div>${advancedFields()}${exportActions({ merge: true })}
      </form>`;
    bindMachinePicker(agents); bindRunForm(); return;
  }
  const image = family === 'classification';
  const prefix = detection ? 'detect' : image ? 'classifier' : 'video';
  main.innerHTML = `<h2 class="page-title">Export</h2><p class="page-sub">Build ONNX or TensorRT artifacts with the preprocessing and tensor contract required for deployment.</p>
    ${machinePickerHtml(agents)}
    <form id="run-form" data-type="${prefix}_export">
      <input type="hidden" name="format" value="onnx" />
      ${commonRunFields(`${prefix}_export_${new Date().toISOString().slice(0, 10).replaceAll('-', '_')}`)}
      <div class="card"><h3>Export source</h3>
        ${field('checkpoint', 'Checkpoint', { placeholder: detection ? '/runs/experiment/weights/best.pt' : image ? '/runs/classifier/best.pt' : '/runs/video_action/best.pt', required: true })}
        <div class="adv-grid">
          ${detection ? field('imgsz', 'Image size', { value: '640', type: 'number', min: 32 }) : ''}
          ${field('opset', 'ONNX opset', { value: '17', type: 'number', min: 11 })}
          ${field('workspace', 'TensorRT workspace (GiB)', { value: '4', type: 'number', min: 1 })}
        </div>
        <label class="inline-check"><input type="checkbox" name="half" /> Build TensorRT engine with FP16 kernels (TensorRT 8–10)</label>
        <p class="muted">ONNX supports dynamic batch input <code>${detection || image ? 'images' : 'clips'}</code>. TensorRT export requires a CUDA target and also preserves the ONNX intermediate.</p>
      </div>${advancedFields()}${exportActions()}
    </form>`;
  bindMachinePicker(agents);
  bindRunForm();
}

function datasetNumber(value) {
  const numeric = Number(value || 0);
  return Number.isFinite(numeric) ? numeric.toLocaleString() : '0';
}

const DATASET_SPLIT_COLORS = ['#64d2ff', '#5e5ce6', '#30d158', '#ff9f0a', '#bf5af2', '#ff375f'];

function datasetClassRowsHtml(profile, selectedSplit = 'all') {
  const splitNames = (profile.splits || []).map((split) => split.name);
  const classes = [...(profile.classes || [])]
    .sort((left, right) => Number(right.total || 0) - Number(left.total || 0))
    .slice(0, 120);
  const counts = classes.map((item) => selectedSplit === 'all'
    ? Number(item.total || 0) : Number(item.splits?.[selectedSplit] || 0));
  const maximum = Math.max(1, ...counts);
  if (!classes.length) return `<div class="empty">${profile.family === 'vlm'
    ? 'VLM JSONL has no class labels; prompt and answer composition is shown below.'
    : 'No labeled classes were found in this dataset.'}</div>`;
  const unit = profile.format === 'coco-multilabel' ? 'positive image label' : 'annotation';
  return `<div class="golden-head-table">${classes.map((item) => {
    const total = selectedSplit === 'all' ? Number(item.total || 0) : Number(item.splits?.[selectedSplit] || 0);
    const visible = selectedSplit === 'all'
      ? splitNames.map((split, index) => ({ split, index, count: Number(item.splits?.[split] || 0) })).filter((entry) => entry.count > 0)
      : [{ split: selectedSplit, index: Math.max(0, splitNames.indexOf(selectedSplit)), count: total }].filter((entry) => entry.count > 0);
    const segments = selectedSplit === 'all'
      ? visible.map((entry) => `<span style="width:${Math.max(.4, 100 * entry.count / maximum)}%;background:${DATASET_SPLIT_COLORS[entry.index % DATASET_SPLIT_COLORS.length]}" title="${esc(entry.split)}: ${datasetNumber(entry.count)}"></span>`).join('')
      : (total ? `<span style="width:${Math.max(.4, 100 * total / maximum)}%;background:${DATASET_SPLIT_COLORS[visible[0]?.index % DATASET_SPLIT_COLORS.length]}" title="${esc(selectedSplit)}: ${datasetNumber(total)}"></span>` : '');
    return `<div class="golden-head-row"><div class="golden-head-name"><strong>${esc(item.name)}</strong><span>${datasetNumber(total)} ${unit}${total === 1 ? '' : 's'}${selectedSplit === 'all' ? ' across all splits' : ` in ${esc(selectedSplit)}`}</span></div>
      <div class="golden-level-bar ${total ? '' : 'distribution-empty'}">${segments}</div>
      <div class="golden-level-counts">${visible.length ? visible.map((entry) => `<span style="color:${DATASET_SPLIT_COLORS[entry.index % DATASET_SPLIT_COLORS.length]}"><b>${esc(entry.split)}</b> ${datasetNumber(entry.count)}</span>`).join('') : '<span class="level-missing-text"><b>No annotations</b> 0</span>'}</div></div>`;
  }).join('')}</div>`;
}

function datasetProfileDetail(profile) {
  const summary = profile.summary || {};
  const splitNames = (profile.splits || []).map((split) => split.name);
  const classes = profile.classes || [];
  const annotationLabel = profile.format === 'coco-multilabel' ? 'Positive labels' : 'Annotations';
  const splitRows = (profile.splits || []).map((split) => `<tr><td><strong>${esc(split.name)}</strong></td><td>${datasetNumber(split.samples)}</td><td>${datasetNumber(split.images)}</td><td>${datasetNumber(split.annotations)}</td><td>${datasetNumber(split.classes)}</td><td>${datasetNumber(split.unlabeled)}</td><td>${datasetNumber(split.invalid)}</td></tr>`).join('');
  const qualityIssues = Number(summary.unlabeled || 0) + Number(summary.invalid || 0);
  const boxes = profile.boxSizes || {};
  const boxTotal = Number(boxes.small || 0) + Number(boxes.medium || 0) + Number(boxes.large || 0);
  const boxColors = { small: '#64d2ff', medium: '#ffcc00', large: '#ff9f0a' };
  const boxHtml = profile.family === 'detection' && boxTotal ? `<section class="card golden-distribution"><div class="row spread golden-distribution-title"><div><span class="control-kicker">Object scale</span><h3>Bounding-box sizes</h3></div><span class="golden-badge frozen">COCO area thresholds</span></div>
    <div class="golden-legend">${Object.entries(boxColors).map(([key, color]) => `<span><i style="background:${color}"></i>${key}</span>`).join('')}</div>
    <div class="golden-head-table"><div class="golden-head-row"><div class="golden-head-name"><strong>Object annotations</strong><span>${datasetNumber(boxTotal)} valid boxes</span></div><div class="golden-level-bar">${Object.entries(boxColors).map(([key, color]) => Number(boxes[key] || 0) ? `<span style="width:${Math.max(.4, 100 * Number(boxes[key]) / boxTotal)}%;background:${color}" title="${key}: ${datasetNumber(boxes[key])}"></span>` : '').join('')}</div><div class="golden-level-counts">${Object.entries(boxColors).map(([key, color]) => `<span style="color:${color}"><b>${key}</b> ${datasetNumber(boxes[key])}</span>`).join('')}</div></div></div></section>` : '';
  const textHtml = profile.family === 'vlm' ? `<section class="card golden-distribution"><div class="row spread golden-distribution-title"><div><span class="control-kicker">Prompt / answer contract</span><h3>Text composition</h3></div><span class="golden-badge live">JSONL</span></div><div class="golden-kpis"><div><strong>${datasetNumber(Math.round(profile.text?.averagePromptCharacters || 0))}</strong><span>average prompt characters</span></div><div><strong>${datasetNumber(Math.round(profile.text?.averageAnswerCharacters || 0))}</strong><span>average answer characters</span></div><div><strong>${datasetNumber(summary.samples)}</strong><span>paired samples</span></div><div><strong>${datasetNumber(summary.unlabeled)}</strong><span>missing images</span></div></div></section>` : '';
  return `<div class="golden-source-grid"><article class="card golden-source-card live"><div class="row spread"><div><span class="control-kicker">Dataset snapshot</span><h3>${esc(profile.name)}</h3></div><span class="golden-badge live">${esc(profile.format)}</span></div>
      <div class="golden-kpis"><div><strong>${datasetNumber(summary.samples)}</strong><span>${profile.family === 'video' ? 'video clips' : 'samples'}</span></div><div><strong>${datasetNumber(summary.annotations)}</strong><span>${profile.format === 'coco-multilabel' ? 'positive image labels' : 'annotations'}</span></div><div><strong>${datasetNumber(summary.classes)}</strong><span>classes</span></div><div><strong>${datasetNumber(profile.splits?.length)}</strong><span>splits</span></div></div><p><code>${esc(profile.source)}</code></p></article>
    <article class="card golden-source-card ${qualityIssues ? 'frozen' : 'live'}"><div class="row spread"><div><span class="control-kicker">Data quality</span><h3>${qualityIssues ? 'Review required' : 'Profile clean'}</h3></div><span class="golden-badge ${qualityIssues ? 'frozen' : 'live'}">${qualityIssues ? `${datasetNumber(qualityIssues)} issues` : 'Ready'}</span></div>
      <div class="golden-kpis"><div><strong>${datasetNumber(summary.labeled)}</strong><span>labeled samples</span></div><div><strong>${datasetNumber(summary.unlabeled)}</strong><span>unlabeled / missing</span></div><div><strong>${datasetNumber(summary.invalid)}</strong><span>invalid records</span></div><div><strong>${datasetNumber(summary.crowd)}</strong><span>crowd annotations</span></div></div><p>Profiled on <strong>${esc(profile.agentId || 'unknown machine')}</strong> · ${esc(fmtTime(profile.generatedAt))}</p></article></div>
    <section class="card golden-splits"><div><span class="control-kicker">Dataset manifests</span><h3>Split composition</h3><p class="muted">Counts belong to this exact COCO or custom dataset profile.</p></div><div class="table-scroll"><table><thead><tr><th>Split</th><th>Samples</th><th>Images</th><th>${annotationLabel}</th><th>Classes</th><th>Unlabeled</th><th>Invalid</th></tr></thead><tbody>${splitRows || '<tr><td colspan="7">No split data</td></tr>'}</tbody></table></div></section>
    <section class="card golden-distribution"><div class="row spread golden-distribution-title"><div><span class="control-kicker">Ground-truth distribution</span><h3>Class distribution</h3></div><div class="golden-view-controls"><select id="dataset-distribution-split"><option value="all">All splits</option>${splitNames.map((split) => `<option value="${esc(split)}">${esc(split)}</option>`).join('')}</select></div></div>
      <div class="golden-legend">${splitNames.map((split, index) => `<span><i style="background:${DATASET_SPLIT_COLORS[index % DATASET_SPLIT_COLORS.length]}"></i>${esc(split)}</span>`).join('')}</div><div id="dataset-class-distribution">${datasetClassRowsHtml(profile)}</div></section>
    ${boxHtml}${textHtml}<details class="card golden-policy"><summary>Definitions and traceability</summary><p><strong>Format:</strong> ${esc(profile.format)}</p><p><strong>Source:</strong> <code>${esc(profile.source)}</code></p><p><strong>Profile job:</strong> ${esc(profile.jobName || profile.jobId || 'not recorded')}</p><p><strong>Machine:</strong> ${esc(profile.agentId || 'not recorded')}</p></details>`;
}

async function renderData(main) {
  const family = currentFamily();
  const profileType = { detection: 'detect_dataset_profile', classification: 'classifier_dataset_profile', video: 'video_dataset_profile', vlm: 'vlm_dataset_profile' }[family];
  const inputHelp = {
    detection: ['COCO annotation JSON, COCO root, or custom YOLO TOML', '/data/coco/annotations/instances_val2017.json'],
    classification: ['COCO annotation JSON/root or class-folder dataset', '/data/coco/annotations/instances_val2017.json'],
    video: ['Class-folder video dataset root', '/data/actions'],
    vlm: ['Image-text JSONL', '/data/vlm/validation.jsonl'],
  }[family];
  main.innerHTML = `<div class="golden-header"><div><span class="control-kicker">${esc(familyLabel())} · COCO / custom</span><h2 class="page-title">Dataset Explorer</h2><p class="page-sub">Dataset composition rendered with the same visual language as AI Automation.</p></div></div><div class="model-card-loading"><div class="evidence-loading"></div><span>Loading ${esc(familyLabel())} dataset statistics…</span></div>`;
  const [{ profiles }, { agents }] = await Promise.all([
    api(`/dataset-profiles?family=${encodeURIComponent(family)}&limit=100`), api('/agents'),
  ]);
  const stamp = new Date().toISOString().slice(0, 10).replaceAll('-', '_');
  const latest = profiles[0];
  const recommendations = DATASET_RECOMMENDATIONS[family].map(([name, role, detail]) => `<article class="card dataset-recommendation"><span class="control-kicker">${esc(role)}</span><h3>${esc(name)}</h3><p>${esc(detail)}</p></article>`).join('');
  main.innerHTML = `<div class="golden-header"><div><span class="control-kicker">${esc(familyLabel())} · COCO / custom</span><h2 class="page-title">Dataset Explorer</h2><p class="page-sub">Dataset composition rendered with the same source cards, manifest table, and distribution rows as AI Automation.</p></div>
    <div class="golden-refresh"><span class="golden-live-dot ${latest ? '' : 'frozen'}"></span><span><strong>${latest ? 'Latest profile' : 'No profile'}</strong><small>${latest ? esc(fmtTime(latest.generatedAt)) : 'profile a dataset below'}</small></span><button class="btn primary small" id="dataset-profile-new">Profile dataset</button></div></div>
    ${profiles.length ? `<label class="field dataset-profile-select"><span>Dataset snapshot</span><select id="dataset-profile-select">${profiles.map((profile, index) => `<option value="${index}">${esc(profile.name)} · ${esc(profile.format)} · ${esc(fmtTime(profile.generatedAt))}</option>`).join('')}</select></label><div id="dataset-profile-root"></div>` : `<section class="card golden-empty"><div class="evidence-loading"></div><div><h3>First dataset profile is pending</h3><p>Profile COCO or a custom dataset on one of the configured Trainer machines.</p></div></section>`}
    <section class="dataset-recommendations"><div class="dataset-recommendation-head"><span class="control-kicker">Recommended progression</span><h3>Datasets for ${esc(familyLabel())}</h3><p class="muted">Start with the smoke dataset, then move to the scale-up option after the full lifecycle passes.</p></div><div class="dataset-recommendation-grid">${recommendations}</div></section>
    <details class="card golden-policy dataset-profile-form" id="dataset-profile-form"><summary>Profile COCO or a custom dataset</summary><p>COCO annotations drive detection and multi-label image classification. Custom YOLO TOML, class folders, video folders, and VLM JSONL use the same visual workspace.</p>
      ${machinePickerHtml(agents)}<form id="run-form" data-type="${profileType}">${commonRunFields(`dataset_profile_${stamp}`)}<div class="card"><h3>Dataset source</h3>${field('data', inputHelp[0], { placeholder: inputHelp[1], required: true })}<p class="muted">Raw dataset files stay on the selected machine.</p></div>${advancedFields()}${runActions('dataset profile')}</form></details>`;
  const draw = () => {
    const index = Number($('#dataset-profile-select')?.value || 0);
    const profile = profiles[index];
    if ($('#dataset-profile-root') && profile) {
      $('#dataset-profile-root').innerHTML = datasetProfileDetail(profile);
      $('#dataset-distribution-split')?.addEventListener('change', (event) => {
        $('#dataset-class-distribution').innerHTML = datasetClassRowsHtml(profile, event.target.value);
      });
    }
  };
  $('#dataset-profile-select')?.addEventListener('change', draw);
  $('#dataset-profile-new')?.addEventListener('click', () => {
    const details = $('#dataset-profile-form'); details.open = true; details.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
  draw(); bindMachinePicker(agents); bindRunForm();
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
      <div class="row"><a class="btn primary" href="#/new">New run</a><a class="btn" href="#/evaluations">Evaluations</a><a class="btn" href="#/evidence">Open xAI</a></div></div>
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
      <footer><button type="button" class="btn small primary use-architecture" data-architecture="${esc(model.name)}">Use in New Run</button><button type="button" class="btn small inspect-architecture" data-architecture="${esc(model.name)}">Inspect flow</button><a class="btn small" href="#/evaluations">Evaluations</a><a class="btn small" href="#/evidence">xAI</a></footer>
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

const classificationXaiImageCache = new Map();

function loadXaiCanvasImage(url, retry = 0) {
  const key = `${url}:${retry}`;
  if (!classificationXaiImageCache.has(key)) classificationXaiImageCache.set(key, new Promise((resolve, reject) => {
    const source = new Image();
    source.decoding = 'async';
    source.onload = () => resolve(source);
    source.onerror = () => {
      if (retry < 1) loadXaiCanvasImage(`${url}${url.includes('?') ? '&' : '?'}retry=${Date.now()}`, retry + 1).then(resolve, reject);
      else reject(new Error(`Unable to load ${url}`));
    };
    source.src = url;
  }));
  return classificationXaiImageCache.get(key);
}

async function drawClassificationGradcam(canvas, originalUrl, mapUrl, intensity) {
  const [original, map] = await Promise.all([
    loadXaiCanvasImage(originalUrl), loadXaiCanvasImage(mapUrl),
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

function renderClassificationXaiViewer(container, run) {
  const images = run.samples || [];
  if (!images.length) {
    container.innerHTML = '<div class="empty">No sample-level ConvNeXt xAI result is available yet.</div>';
    return;
  }
  container.innerHTML = `<section class="card classification-xai-live">
    <div class="model-registry-head"><div><span class="control-kicker">Test sample result</span><h3>Prediction + ConvNeXt xAI</h3>
      <p class="muted">${esc(run.architecture || 'classification model')} · red supports the selected class; blue opposes it.</p></div></div>
    <div class="classification-xai-controls">
      <label class="field"><span>Test sample</span><select id="classification-xai-image">${images.map((item, index) => `<option value="${index}">${esc(item.label)} · ${esc(item.predicted)} · ${(Number(item.confidence) * 100).toFixed(1)}%</option>`).join('')}</select></label>
      <label class="field"><span>Activation intensity <strong id="classification-xai-intensity-value">72%</strong></span><input id="classification-xai-intensity" type="range" min="0" max="100" value="72" /></label>
    </div>
    <div id="classification-xai-prediction" class="classification-xai-prediction"></div>
    <div class="classification-xai-stage">
      <figure class="classification-xai-map"><canvas id="classification-xai-canvas"></canvas><figcaption id="classification-xai-caption"></figcaption></figure>
      <figure><img id="classification-xai-activations" alt="ConvNeXt feature activations" /><figcaption>Highest-energy feature channels at every captured ConvNeXt stage</figcaption></figure>
    </div><div id="classification-xai-scores" class="classification-xai-scores"></div>
  </section>`;
  const select = $('#classification-xai-image', container);
  const slider = $('#classification-xai-intensity', container);
  const canvas = $('#classification-xai-canvas', container);
  const activationImage = $('#classification-xai-activations', container);
  let renderToken = 0;
  const draw = async () => {
    const item = images[Number(select.value) || 0];
    const token = ++renderToken;
    $('#classification-xai-intensity-value', container).textContent = `${slider.value}%`;
    $('#classification-xai-prediction', container).innerHTML = `<span>Prediction</span><strong>${esc(item.predicted)}</strong><b>${(Number(item.confidence) * 100).toFixed(1)}%</b>`;
    $('#classification-xai-caption', container).textContent = `Test image + signed Grad-CAM · target ${item.target} ${(Number(item.targetScore) * 100).toFixed(1)}% · ${item.layer}`;
    $('#classification-xai-scores', container).innerHTML = (item.topK || []).map((score) => `<span><strong>${esc(score.class)}</strong> ${(Number(score.confidence) * 100).toFixed(1)}%</span>`).join('');
    activationImage.src = item.activationsUrl;
    canvas.closest('figure').classList.add('loading');
    try {
      await drawClassificationGradcam(canvas, item.inputUrl, item.signedMapUrl, Number(slider.value) / 100);
      if (token === renderToken) canvas.closest('figure').classList.remove('loading', 'failed');
    } catch {
      if (token === renderToken) canvas.closest('figure').classList.add('failed');
    }
  };
  select.addEventListener('change', draw);
  slider.addEventListener('input', draw);
  draw();
}

async function loadClassificationXaiViewer(container, runName = '') {
  if (!container) return;
  container.innerHTML = '<div class="empty">Loading ConvNeXt xAI results…</div>';
  try {
    const payload = await api(`/xai/classification?limit=100${runName ? `&run=${encodeURIComponent(runName)}` : ''}`);
    const runs = payload.runs || [];
    if (!runs.length) {
      container.innerHTML = '<div class="empty">No test-sample xAI result yet. Upload one image above to generate its prediction, signed map, and activations.</div>';
      return;
    }
    if (runName || runs.length === 1) {
      renderClassificationXaiViewer(container, runs[0]);
      return;
    }
    container.innerHTML = `<section class="card classification-xai-catalog">
      <div><span class="control-kicker">xAI results</span><h3>Completed xAI run</h3>
        <p class="muted">Choose a run, then inspect each sample’s input, prediction, signed Grad-CAM, and ConvNeXt feature activations together.</p></div>
      <label class="field"><span>Run</span><select id="classification-xai-run-select">${runs.map((run, index) => `<option value="${index}">${esc(run.name)}</option>`).join('')}</select></label></section><div id="classification-xai-active"></div>`;
    const select = $('#classification-xai-run-select', container);
    const active = $('#classification-xai-active', container);
    let drawToken = 0;
    const draw = async () => {
      const token = ++drawToken;
      const run = runs[Number(select.value) || 0];
      if (token === drawToken) renderClassificationXaiViewer(active, run);
    };
    select.addEventListener('change', () => draw().catch((error) => {
      active.innerHTML = `<div class="warn-banner">${esc(error.message || 'Unable to load selected xAI run')}</div>`;
    }));
    await draw();
  } catch (error) {
    container.innerHTML = `<div class="warn-banner">${esc(error.message || 'Unable to load ConvNeXt xAI manifest')}</div>`;
  }
}

function fileDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error || new Error('Unable to read test sample'));
    reader.readAsDataURL(file);
  });
}

function classificationCheckpointRows(inventory) {
  return Object.entries(inventory || {}).flatMap(([agentId, value]) => (value.checkpoints || [])
    .filter((checkpoint) => !String(checkpoint).toLowerCase().includes('/video') && !String(checkpoint).toLowerCase().endsWith('/adapter'))
    .map((checkpoint) => ({ agentId, checkpoint })));
}

async function watchClassificationXaiSample(status, viewer, jobId) {
  if (!status?.isConnected || currentFamily() !== 'classification') return;
  try {
    const { job } = await api(`/jobs/${jobId}`);
    status.innerHTML = `<div><strong>${esc(job.name)}</strong> ${badge(job.status)}</div><span class="muted">${esc(job.steps.find((step) => step.status === 'running')?.title || (job.status === 'queued' ? 'Waiting for the selected machine' : 'One-image prediction and xAI'))} · <a href="#/job/${job.id}">open job</a></span>`;
    if (job.status === 'completed') {
      await loadClassificationXaiViewer(viewer, job.name);
      return;
    }
    if (['failed', 'cancelled'].includes(job.status)) return;
    jobTimer = setTimeout(() => watchClassificationXaiSample(status, viewer, jobId), 1800);
  } catch (error) {
    status.innerHTML = `<div class="warn-banner">${esc(error.message || 'Unable to read sample job')}</div>`;
  }
}

function renderClassificationXaiTest(container, viewer, agents, inventory) {
  const machines = machineList(agents);
  const savedTarget = localStorage.getItem(TARGET_KEY);
  const selected = machines.find((machine) => machine.agentId === savedTarget) || machines.find((machine) => machine.online) || machines[0];
  const checkpoints = classificationCheckpointRows(inventory);
  container.innerHTML = `<section class="card classification-xai-test">
    <div class="model-registry-head"><div><span class="control-kicker">Selected-sample xAI</span><h3>Test one image</h3>
      <p class="muted">Choose a reserved or uploaded test image and a trained checkpoint, then inspect the prediction and explanation together.</p></div></div>
    <form id="classification-xai-test-form">
      <div class="classification-xai-test-grid">
        <label class="xai-sample-picker" for="classification-xai-file"><input id="classification-xai-file" type="file" accept="image/jpeg,image/png,image/webp" required />
          <span id="classification-xai-file-copy"><strong>Choose test sample</strong><small>JPEG, PNG, or WebP · one image · up to 12 MiB</small></span>
          <img id="classification-xai-file-preview" class="hidden" alt="Selected test sample" /></label>
        <div>
          <label class="field"><span>Checkpoint on target machine</span><input id="classification-xai-checkpoint" list="classification-xai-checkpoints" placeholder="/runs/classifier/weights/best.pt" required /></label>
          <datalist id="classification-xai-checkpoints"></datalist>
          <div class="grid2">
            <label class="field"><span>Target class</span><input id="classification-xai-target" value="predicted" placeholder="predicted or class name" /></label>
            <label class="field"><span>Device</span><select id="classification-xai-device">${deviceOptions(selected?.gpus?.length ? 'cuda:0' : 'cpu')}</select></label>
          </div>
        </div>
      </div>
      <div class="grid2 xai-sample-run-controls">
        <label class="field"><span>Machine</span><select id="classification-xai-machine">${machines.map((machine) => `<option value="${esc(machine.agentId)}" ${machine.agentId === selected?.agentId ? 'selected' : ''}>${esc(machineOptionLabel(machine))}</option>`).join('')}</select></label>
        <div class="xai-sample-submit"><button type="submit" class="btn primary" id="classification-xai-run">Run test sample + xAI</button><span class="muted">Prediction, confidence, signed Grad-CAM, and stage activations</span></div>
      </div>
    </form>
    <div id="classification-xai-test-status" class="evidence-status"><span class="muted">Choose a sample and a checkpoint to begin.</span></div>
  </section>`;

  const form = $('#classification-xai-test-form', container);
  const fileInput = $('#classification-xai-file', container);
  const preview = $('#classification-xai-file-preview', container);
  const copy = $('#classification-xai-file-copy', container);
  const machine = $('#classification-xai-machine', container);
  const checkpoint = $('#classification-xai-checkpoint', container);
  const options = $('#classification-xai-checkpoints', container);
  const device = $('#classification-xai-device', container);
  const status = $('#classification-xai-test-status', container);
  const button = $('#classification-xai-run', container);
  let previewUrl = '';

  const updateCheckpoints = () => {
    const rows = checkpoints.filter((item) => !machine.value || item.agentId === machine.value);
    options.innerHTML = rows.map((item) => `<option value="${esc(item.checkpoint)}">${esc(item.agentId)}</option>`).join('');
    const preferred = rows.find((item) => /classif|convnext|image.?class/i.test(item.checkpoint) && item.checkpoint.endsWith('/best.pt'));
    if (!checkpoint.value && (preferred || rows.length === 1)) checkpoint.value = preferred?.checkpoint || rows[0].checkpoint;
  };
  machine.addEventListener('change', () => {
    localStorage.setItem(TARGET_KEY, machine.value);
    checkpoint.value = '';
    const selectedMachine = machines.find((item) => item.agentId === machine.value);
    device.value = selectedMachine?.gpus?.length ? 'cuda:0' : 'cpu';
    updateCheckpoints();
  });
  updateCheckpoints();

  fileInput.addEventListener('change', () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    const file = fileInput.files?.[0];
    if (!file) {
      preview.classList.add('hidden');
      copy.classList.remove('hidden');
      return;
    }
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    preview.classList.remove('hidden');
    copy.classList.add('hidden');
    copy.innerHTML = `<strong>${esc(file.name)}</strong><small>${fmtBytes(file.size)} · ready to test</small>`;
  });

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const file = fileInput.files?.[0];
    if (!file) return toast('Choose one test image', true);
    if (file.size > 12 * 1024 * 1024) return toast('Test sample must be 12 MiB or smaller', true);
    button.disabled = true;
    button.textContent = 'Uploading sample…';
    try {
      const uploaded = await api('/samples', { method: 'POST', body: {
        name: file.name, type: file.type, data: await fileDataUrl(file),
      } });
      const stamp = new Date().toISOString().replace(/[-:TZ.]/g, '').slice(0, 14);
      const params = {
        name: `classifier_xai_sample_${stamp}`,
        source: `sample://${uploaded.sample.id}`,
        checkpoint: checkpoint.value.trim(),
        target: $('#classification-xai-target', container).value.trim() || 'predicted',
        maxImages: 1, activationChannels: 4, intensity: 0.72,
        device: device.value, project: CONFIG.defaults.runsDir,
        targetAgentId: machine.value,
      };
      if (!params.checkpoint) throw new Error('Checkpoint is required');
      button.textContent = 'Queueing xAI…';
      const { job } = await api('/jobs', { method: 'POST', body: { type: 'classifier_xai', params } });
      localStorage.setItem(XAI_SAMPLE_JOB_KEY, job.id);
      toast(`Queued test sample ${job.name}`);
      await watchClassificationXaiSample(status, viewer, job.id);
    } catch (error) {
      status.innerHTML = `<div class="warn-banner">${esc(error.message || 'Unable to run test sample')}</div>`;
      toast(error.message, true);
    } finally {
      button.disabled = false;
      button.textContent = 'Run test sample + xAI';
    }
  });

  const activeJobId = localStorage.getItem(XAI_SAMPLE_JOB_KEY);
  return { status, activeJobId };
}

async function renderEvidence(main) {
  const family = currentFamily();
  const context = family === 'classification'
    ? await Promise.all([api('/agents'), api('/inventory')]).catch(() => [{ agents: {} }, { inventory: {} }])
    : [{ agents: {} }, { inventory: {} }];
  const agents = context[0].agents || {};
  const inventory = context[1].inventory || {};
  main.innerHTML = `<div class="model-registry-head"><div><span class="control-kicker">${esc(familyLabel())}</span><h2 class="page-title">xAI</h2>
      <p class="page-sub">Inspect each test sample together with its prediction, confidence, and explanation.</p></div>
      <div class="row"><a class="btn" href="#/models">Model registry</a><button class="btn primary" id="evidence-refresh">Refresh</button></div></div>
    ${family === 'classification'
      ? '<div id="classification-xai-test"></div><div id="classification-xai-viewer"></div>'
      : '<div class="empty">No sample-level xAI results are available for this model family yet.</div>'}`;

  $('#evidence-refresh').addEventListener('click', () => renderEvidence(main));
  if (family === 'classification') {
    const viewer = $('#classification-xai-viewer');
    const sampleTest = renderClassificationXaiTest($('#classification-xai-test'), viewer, agents, inventory);
    await loadClassificationXaiViewer(viewer);
    if (sampleTest.activeJobId) watchClassificationXaiSample(sampleTest.status, viewer, sampleTest.activeJobId);
  }
}

function evaluationNumber(value, { percent = true, digits = 3 } = {}) {
  if (value == null || !Number.isFinite(Number(value))) return '—';
  return percent ? `${(Number(value) * 100).toFixed(1)}%` : Number(value).toFixed(digits);
}

function evaluationSpec(family) {
  if (family === 'detection') return {
    headline: 'Detector evaluations',
    subtitle: 'Checkpoints re-evaluated together on one explicit dataset and validation split.',
    columns: [
      ['map5095', 'mAP50–95'], ['map50', 'mAP50'], ['precision', 'Precision'],
      ['recall', 'Recall'], ['lowestRecall', 'Lowest recall'],
    ],
    cards: [['map5095', 'mAP50–95'], ['map50', 'mAP50'], ['precision', 'Precision'], ['recall', 'Recall']],
  };
  if (family === 'classification') return {
    headline: 'Image classification evaluations',
    subtitle: 'Single-label and COCO multi-label classifiers re-evaluated on one explicit dataset and split.',
    columns: [
      ['map', 'mAP'], ['microF1', 'Micro F1'], ['macroF1', 'Macro F1'], ['accuracy', 'Exact / top-1 accuracy'],
      ['lowestRecall', 'Lowest recall'],
    ],
    cards: [['map', 'mAP'], ['microF1', 'Micro F1'], ['macroF1', 'Macro F1'], ['accuracy', 'Exact / top-1 accuracy']],
  };
  if (family === 'video') return {
    headline: 'Video / action evaluations',
    subtitle: 'Action checkpoints re-evaluated together on one explicit video dataset and split.',
    columns: [['accuracy', 'Accuracy'], ['loss', 'Loss', false], ['lowestAccuracy', 'Lowest class accuracy']],
    cards: [['accuracy', 'Accuracy'], ['loss', 'Loss', false], ['lowestAccuracy', 'Lowest class accuracy'], ['samples', 'Clips', false]],
  };
  return {
    headline: 'VLM evaluations',
    subtitle: 'Base models and adapters re-evaluated together on one explicit image-text dataset and split.',
    columns: [['tokenAccuracy', 'Token accuracy'], ['loss', 'Loss', false], ['perplexity', 'Perplexity', false], ['evaluatedTokens', 'Evaluated tokens', false]],
    cards: [['tokenAccuracy', 'Token accuracy'], ['loss', 'Loss', false], ['perplexity', 'Perplexity', false], ['evaluatedTokens', 'Evaluated tokens', false]],
  };
}

function evaluationValue(record, key) {
  if (key === 'samples') return record.samples;
  if (key === 'lowestRecall') {
    const values = (record.classes || []).map((item) => item.recall).filter((item) => item != null);
    return values.length ? Math.min(...values) : null;
  }
  if (key === 'lowestAccuracy') {
    const values = (record.classes || []).map((item) => item.accuracy).filter((item) => item != null);
    return values.length ? Math.min(...values) : null;
  }
  return record.metrics?.[key] ?? null;
}

function evaluationMetric(value, percent = true) {
  if (value == null || !Number.isFinite(Number(value))) return '—';
  if (!percent && Number.isInteger(Number(value))) return Number(value).toLocaleString();
  return evaluationNumber(value, { percent, digits: 3 });
}

function evaluationClassTable(record) {
  const rows = record.classes || [];
  if (!rows.length) return '';
  const detection = record.family === 'detection';
  const video = record.family === 'video';
  const multiLabel = record.family === 'classification' && rows.some((item) => item.ap != null);
  const headers = detection
    ? ['Class', 'Precision', 'Recall', 'AP50', 'AP50–95']
    : video ? ['Class', 'Accuracy', 'Support']
      : multiLabel ? ['Class', 'Precision', 'Recall', 'F1', 'AP', 'Support']
        : ['Class', 'Precision', 'Recall', 'F1', 'Support'];
  const cells = (item) => detection
    ? [item.name, evaluationMetric(item.precision), evaluationMetric(item.recall), evaluationMetric(item.ap50), evaluationMetric(item.map5095)]
    : video ? [item.name, evaluationMetric(item.accuracy), evaluationMetric(item.support, false)]
      : multiLabel
        ? [item.name, evaluationMetric(item.precision), evaluationMetric(item.recall), evaluationMetric(item.f1), evaluationMetric(item.ap), evaluationMetric(item.support, false)]
        : [item.name, evaluationMetric(item.precision), evaluationMetric(item.recall), evaluationMetric(item.f1), evaluationMetric(item.support, false)];
  return `<section class="card evaluation-section"><div class="evaluation-section-head"><div><span class="control-kicker">Class diagnostics</span><h3>Per-class metrics</h3></div><span>${rows.length} classes</span></div>
    <table class="evaluation-class-table"><thead><tr>${headers.map((item) => `<th>${esc(item)}</th>`).join('')}</tr></thead><tbody>
      ${rows.map((item) => `<tr>${cells(item).map((value, index) => `<td>${index === 0 ? `<strong>${esc(value)}</strong>` : esc(value)}</td>`).join('')}</tr>`).join('')}
    </tbody></table></section>`;
}

function evaluationConfusionMatrix(record) {
  const matrix = record.confusionMatrix || [];
  if (!matrix.length || matrix.length > 30) return '';
  const names = record.classes?.map((item) => item.name) || matrix.map((_, index) => String(index));
  const maximum = Math.max(1, ...matrix.flat().map(Number));
  return `<section class="card evaluation-section"><div class="evaluation-section-head"><div><span class="control-kicker">Error distribution</span><h3>Confusion matrix</h3></div><span>Rows: actual · columns: predicted</span></div>
    <div class="evaluation-matrix-wrap"><table class="evaluation-matrix"><thead><tr><th></th>${names.map((name) => `<th title="${esc(name)}">${esc(name)}</th>`).join('')}</tr></thead><tbody>
      ${matrix.map((row, rowIndex) => `<tr><th title="${esc(names[rowIndex] || rowIndex)}">${esc(names[rowIndex] || rowIndex)}</th>${row.map((value) => `<td style="--matrix-alpha:${Math.max(.05, Number(value) / maximum)}">${esc(value)}</td>`).join('')}</tr>`).join('')}
    </tbody></table></div></section>`;
}

function evaluationDetail(record, spec) {
  const cards = spec.cards.map(([key, label, percent = true]) => {
    const value = evaluationValue(record, key);
    return `<div><span>${esc(label)}</span><strong>${esc(evaluationMetric(value, percent))}</strong></div>`;
  }).join('');
  return `<section class="evaluation-detail" id="evaluation-detail">
    <div class="evaluation-scope"><div><span class="control-kicker">Comparison scope locked</span><strong>Same dataset · same split</strong><small>${esc(record.dataset?.label || 'dataset')} · ${esc(record.dataset?.split || 'validation')}</small></div>
      <div><span>Evaluated samples</span><strong>${esc(evaluationMetric(record.samples, false))}</strong><small>${esc(record.agentId || 'machine not recorded')}</small></div></div>
    <div class="evaluation-model-head"><div><span class="control-kicker">Selected checkpoint</span><h3>${esc(record.checkpoint?.label || record.jobName)}</h3><p>${esc(record.architecture || familyLabel())}</p></div><a class="btn" href="#/job/${encodeURIComponent(record.jobId)}">Open evaluation job</a></div>
    <div class="evaluation-kpis">${cards}</div>
    ${evaluationClassTable(record)}${evaluationConfusionMatrix(record)}
    <section class="evaluation-provenance">
      <div><span>Dataset</span><strong>${esc(record.dataset?.label || '—')}</strong><small>${esc(record.dataset?.source || '—')}</small></div>
      <div><span>Split</span><strong>${esc(record.dataset?.split || 'validation')}</strong><small>held constant in this ranking</small></div>
      <div><span>Checkpoint</span><strong>${esc(record.checkpoint?.label || '—')}</strong><small>${esc(record.checkpoint?.source || '—')}</small></div>
      <div><span>Completed</span><strong>${esc(fmtTime(record.endedAt))}</strong><small>${esc(record.agentId || 'machine not recorded')}</small></div>
    </section>
    <div class="evaluation-rule"><strong>Comparison rule</strong><span>Only checkpoints evaluated in the same evaluation scope are ranked together. Dataset and split must match exactly.</span></div>
  </section>`;
}

async function renderEvaluations(main) {
  const family = currentFamily();
  const spec = evaluationSpec(family);
  const actionLabel = family === 'vlm' ? 'VLM' : familyLabel();
  main.innerHTML = `<div class="evaluation-head"><div><span class="control-kicker">${esc(familyLabel())}</span><h2 class="page-title">${esc(spec.headline)}</h2><p class="page-sub">${esc(spec.subtitle)}</p></div><a class="btn primary" href="#/new/evaluate">Run ${esc(actionLabel)} evaluation</a></div>
    <div class="card"><div class="muted">Loading evaluation jobs…</div></div>`;
  const { evaluations } = await api(`/evaluations?family=${encodeURIComponent(family)}&limit=200`);
  if (!evaluations.length) {
    main.innerHTML = `<div class="evaluation-head"><div><span class="control-kicker">${esc(familyLabel())}</span><h2 class="page-title">${esc(spec.headline)}</h2><p class="page-sub">${esc(spec.subtitle)}</p></div><a class="btn primary" href="#/new/evaluate">Run ${esc(actionLabel)} evaluation</a></div>
      <div class="card evaluation-empty"><h3>No completed evaluations yet</h3><p>Run an evaluation for this model family. The result will appear here with its metrics and model-level diagnostics.</p><a class="btn" href="#/new/evaluate">Configure evaluation</a></div>`;
    return;
  }
  const scopes = [...new Map(evaluations.map((record) => [record.dataset.key, record.dataset])).entries()];
  let activeScope = scopes[0][0];
  let selectedId = null;
  main.innerHTML = `<div class="evaluation-head"><div><span class="control-kicker">${esc(familyLabel())}</span><h2 class="page-title">${esc(spec.headline)}</h2><p class="page-sub">${esc(spec.subtitle)}</p></div><a class="btn primary" href="#/new/evaluate">Run ${esc(actionLabel)} evaluation</a></div>
    <section class="card evaluation-filter"><div><span class="control-kicker">Evaluation scope</span><h3>Evaluation run + dataset + split</h3></div><label class="field"><span>Dataset and split</span><select id="evaluation-scope">${scopes.map(([key, dataset]) => `<option value="${esc(key)}">${esc(dataset.label)} · ${esc(dataset.split)}</option>`).join('')}</select></label></section>
    <div class="warn-banner">Training splits may differ. Only checkpoints evaluated against the same dataset and split are ranked together.</div>
    <section class="card evaluation-leaderboard"><div class="evaluation-section-head"><div><span class="control-kicker">Comparable checkpoints</span><h3>Checkpoint leaderboard</h3></div><span id="evaluation-count"></span></div><div id="evaluation-table"></div></section>
    <div id="evaluation-detail-root"></div>`;
  const draw = () => {
    const scoped = evaluations.filter((record) => record.dataset.key === activeScope).sort((left, right) => {
      const leftValue = Number(left.primary?.value);
      const rightValue = Number(right.primary?.value);
      if (!Number.isFinite(leftValue)) return 1;
      if (!Number.isFinite(rightValue)) return -1;
      return left.primary?.direction === 'lower' ? leftValue - rightValue : rightValue - leftValue;
    });
    if (!scoped.some((record) => record.id === selectedId)) selectedId = scoped[0]?.id;
    $('#evaluation-count').textContent = `${scoped.length} comparable checkpoint${scoped.length === 1 ? '' : 's'}`;
    $('#evaluation-table').innerHTML = `<table><thead><tr><th>Rank</th><th>Checkpoint</th><th>Split</th><th>Samples</th>${spec.columns.map(([, label]) => `<th>${esc(label)}</th>`).join('')}<th>Completed</th></tr></thead><tbody>
      ${scoped.map((record, index) => `<tr class="clickable evaluation-row ${record.id === selectedId ? 'selected' : ''}" data-id="${esc(record.id)}"><td><span class="evaluation-rank">${index + 1}</span></td><td><strong>${esc(record.checkpoint?.label || record.jobName)}</strong><br><span class="muted">${esc(record.architecture || record.jobName)}</span></td><td>${esc(record.dataset?.split || 'validation')}</td><td>${esc(evaluationMetric(record.samples, false))}</td>${spec.columns.map(([key, , percent = true]) => `<td>${esc(evaluationMetric(evaluationValue(record, key), percent))}</td>`).join('')}<td>${esc(fmtTime(record.endedAt))}</td></tr>`).join('')}
      </tbody></table>`;
    const selected = scoped.find((record) => record.id === selectedId) || scoped[0];
    $('#evaluation-detail-root').innerHTML = selected ? evaluationDetail(selected, spec) : '';
    $$('.evaluation-row').forEach((row) => row.addEventListener('click', () => {
      selectedId = row.dataset.id; draw(); $('#evaluation-detail')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }));
  };
  $('#evaluation-scope').addEventListener('change', (event) => { activeScope = event.target.value; selectedId = null; draw(); });
  draw();
}

async function renderHistory(main) {
  const [{ jobs }, { runs }] = await Promise.all([api('/jobs?limit=500'), api('/runs?limit=1000')]);
  const artifactPaths = new Set(jobs.flatMap((job) => job.artifacts || []).map(String));
  const discovered = (runs || []).filter((run) => !artifactPaths.has(String(run.path)));
  main.innerHTML = `<h2 class="page-title">History</h2><p class="page-sub">Web-queued jobs and completed runs discovered on every connected Trainer machine.</p>
    <div class="row filter-bar">
      <label class="field"><span>Model family</span><select id="history-family"><option value="all">All model families</option>${Object.entries(FAMILIES).map(([key, value]) => `<option value="${esc(key)}">${esc(value.label)}</option>`).join('')}</select></label>
      <label class="field"><span>Status</span><select id="history-status"><option value="">All statuses</option><option>queued</option><option>running</option><option>completed</option><option>failed</option><option>cancelled</option></select></label>
      <span class="count" id="history-count"></span></div>
    <div class="card"><h3>Control-plane jobs</h3><div id="history-table"></div></div>
    <div class="card"><h3>Discovered runs</h3><p class="muted">Includes runs launched directly on a VM and per-family entries from <code>vm_smoke_matrix/summary.json</code>.</p><div id="discovered-history-table"></div></div>`;
  const draw = () => {
    const status = $('#history-status').value;
    const family = $('#history-family').value;
    const filtered = jobs.filter((job) => (family === 'all' || String(job.type).startsWith(FAMILIES[family]?.prefix || '')) && (!status || job.status === status));
    const filteredRuns = discovered.filter((run) => (family === 'all' || run.family === family) && (!status || run.status === status));
    $('#history-table').innerHTML = jobTable(filtered);
    $('#discovered-history-table').innerHTML = discoveredRunTable(filteredRuns);
    $('#history-count').textContent = `${filtered.length} job${filtered.length === 1 ? '' : 's'} · ${filteredRuns.length} discovered run${filteredRuns.length === 1 ? '' : 's'}`;
    bindJobRows();
    bindDiscoveredRuns();
  };
  $('#history-status').addEventListener('change', draw);
  $('#history-family').addEventListener('change', draw);
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
    if (view === 'new') return await renderNew(main, argument);
    if (view === 'evaluations') return await renderEvaluations(main);
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
