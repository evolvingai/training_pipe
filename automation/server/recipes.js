import path from 'node:path';

const DEVICES = /^(auto|cpu|mps|cuda(?::\d+)?)$/;
const MODEL_NAMES = /^(?:trainer(?:-(?:n|s|m|l|x))?|trainer-(?:csp|fpn)(?:-(?:n|s|m|l|x))?)$/;

function text(value, name, { required = true, max = 500 } = {}) {
  const result = String(value ?? '').trim();
  if (required && !result) throw new Error(`${name} is required`);
  if (result.length > max || result.includes('\0') || /[\r\n]/.test(result)) throw new Error(`${name} is invalid`);
  return result;
}

function name(value) {
  const result = text(value || 'run', 'name', { max: 80 }).replace(/[^A-Za-z0-9_.-]/g, '_');
  if (!result || result === '.' || result === '..') throw new Error('name is invalid');
  return result;
}

function integer(value, field, fallback, min = 1, max = 100000) {
  const parsed = value == null || value === '' ? fallback : Number(value);
  if (!Number.isInteger(parsed) || parsed < min || parsed > max) throw new Error(`${field} must be ${min}..${max}`);
  return parsed;
}

function number(value, field, fallback, min = 0, max = 1) {
  const parsed = value == null || value === '' ? fallback : Number(value);
  if (!Number.isFinite(parsed) || parsed < min || parsed > max) throw new Error(`${field} must be ${min}..${max}`);
  return parsed;
}

function device(value, fallback) {
  const result = text(value || fallback || 'cpu', 'device').toLowerCase();
  if (!DEVICES.test(result)) throw new Error('device must be auto, cpu, mps, cuda, or cuda:N');
  return result === 'auto' ? '' : result;
}

function dtype(value) {
  const result = text(value || 'auto', 'dtype').toLowerCase();
  if (!['auto', 'bfloat16', 'float16', 'float32'].includes(result)) {
    throw new Error('dtype must be auto, bfloat16, float16, or float32');
  }
  return result;
}

function model(value) {
  const result = text(value || 'trainer-n', 'model');
  if (result.endsWith('.pt') || MODEL_NAMES.test(result)) return result;
  throw new Error('model must be a supported detector model name or .pt checkpoint');
}

function command(id, title, cwd, argv) {
  return { id, title, cwd, argv: argv.map(String) };
}

function detector(python, cwd, id, title, action, values) {
  const args = Object.entries(values)
    .filter(([, value]) => value !== '' && value != null)
    .map(([key, value]) => `${key}=${String(value).toLowerCase()}`);
  return command(id, title, cwd, [python, '-m', 'trainer', action, ...args]);
}

function runDir(project, runName, cwd) {
  return path.resolve(cwd, project, runName);
}

function augmentationValues(p) {
  const values = {};
  if (p.augmentationConfig) values.augmentation_config = text(p.augmentationConfig, 'augmentationConfig');
  if (p.augProfile) values.aug_profile = text(p.augProfile, 'augProfile');
  const probabilities = {
    mosaic: 'mosaic', mosaic9: 'mosaic9', mixup: 'mixup', copyPaste: 'copy_paste',
    flipud: 'flipud', fliplr: 'fliplr', brightnessContrast: 'brightness_contrast',
    blur: 'blur', noise: 'noise', jpeg: 'jpeg', erasing: 'erasing',
  };
  for (const [field, key] of Object.entries(probabilities)) {
    if (p[field] !== undefined && p[field] !== '') values[key] = number(p[field], field, 0, 0, 1);
  }
  const magnitudes = {
    mixupAlpha: ['mixup_alpha', 100], degrees: ['degrees', 360], translate: ['translate', 1],
    scale: ['scale', 1], shear: ['shear', 90], perspective: ['perspective', 1],
    hsvH: ['hsv_h', 1], hsvS: ['hsv_s', 1], hsvV: ['hsv_v', 1],
    brightness: ['brightness', 1], contrast: ['contrast', 1], squareCropJitter: ['square_crop_jitter', 1],
  };
  for (const [field, [key, maximum]] of Object.entries(magnitudes)) {
    if (p[field] !== undefined && p[field] !== '') values[key] = number(p[field], field, 0, 0, maximum);
  }
  if (p.copyPasteIou) {
    const pair = text(p.copyPasteIou, 'copyPasteIou').split(',').map((item) => Number(item.trim()));
    if (pair.length !== 2 || pair.some((item) => !Number.isFinite(item) || item < 0 || item > 1)) {
      throw new Error('copyPasteIou must be two comma-separated values between 0 and 1');
    }
    values.copy_paste_iou = pair.join(',');
  }
  if (p.squareCrop === true || p.squareCrop === 'on') values.square_crop = true;
  return values;
}

function appendAugmentation(argv, p) {
  for (const [key, value] of Object.entries(augmentationValues(p))) {
    argv.push(`${key}=${String(value).toLowerCase()}`);
  }
}

export const recipeCatalog = [
  { type: 'detect_train', label: 'Detector · train', fields: ['data', 'model', 'epochs', 'batch', 'imgsz', 'device', 'project'] },
  { type: 'detect_finetune', label: 'Detector · finetune', fields: ['data', 'checkpoint', 'epochs', 'batch', 'imgsz', 'freeze', 'device', 'project'] },
  { type: 'detect_evaluate', label: 'Detector · evaluate/report', fields: ['data', 'checkpoint', 'device', 'project'] },
  { type: 'detect_predict', label: 'Detector · predict', fields: ['source', 'checkpoint', 'conf', 'device', 'project'] },
  { type: 'detect_export', label: 'Detector · export ONNX / TensorRT', fields: ['checkpoint', 'format', 'imgsz', 'device'] },
  { type: 'detect_tune', label: 'Detector · tune', fields: ['data', 'model', 'iterations', 'device', 'project'] },
  { type: 'classifier_train', label: 'Classifier · train', fields: ['data', 'arch', 'epochs', 'batch', 'imgsz', 'device', 'project'] },
  { type: 'classifier_finetune', label: 'Classifier · finetune', fields: ['data', 'checkpoint', 'epochs', 'batch', 'imgsz', 'freeze', 'device', 'project'] },
  { type: 'classifier_evaluate', label: 'Classifier · evaluate/report', fields: ['data', 'checkpoint', 'batch', 'device', 'project'] },
  { type: 'classifier_predict', label: 'Classifier · predict', fields: ['source', 'checkpoint', 'batch', 'device', 'project'] },
  { type: 'classifier_xai', label: 'Classifier · signed Grad-CAM + activations', fields: ['source', 'checkpoint', 'target', 'maxImages', 'device', 'project'] },
  { type: 'classifier_export', label: 'Classifier · export ONNX / TensorRT', fields: ['checkpoint', 'format', 'device', 'project'] },
  { type: 'classifier_tune', label: 'Classifier · tune', fields: ['data', 'arch', 'iterations', 'tuneEpochs', 'device', 'project'] },
  { type: 'classifier_tune-thresh', label: 'Classifier · tune thresholds', fields: ['data', 'checkpoint', 'device', 'project'] },
  { type: 'classifier_report', label: 'Classifier · report', fields: ['data', 'checkpoint', 'thresholds', 'device', 'project'] },
  { type: 'classifier_viz-augment', label: 'Classifier · augmentation preview', fields: ['data', 'imgsz', 'device', 'project'] },
  { type: 'classifier_bench-data', label: 'Classifier · benchmark data', fields: ['data', 'batch', 'imgsz', 'device', 'project'] },
  { type: 'video_train', label: 'Video/action · train', fields: ['data', 'arch', 'epochs', 'batch', 'imgsz', 'clipLen', 'device', 'project'] },
  { type: 'video_evaluate', label: 'Video/action · evaluate', fields: ['data', 'checkpoint', 'batch', 'device', 'project'] },
  { type: 'video_predict', label: 'Video/action · predict', fields: ['source', 'checkpoint', 'batch', 'device', 'project'] },
  { type: 'video_export', label: 'Video/action · export ONNX / TensorRT', fields: ['checkpoint', 'format', 'device', 'project'] },
  { type: 'vlm_tune', label: 'VLM · LoRA tune', fields: ['data', 'arch', 'model', 'epochs', 'batch', 'lr', 'loraR', 'device', 'project'] },
  { type: 'vlm_merge', label: 'VLM · merge adapter', fields: ['model', 'adapter', 'device', 'project'] },
  { type: 'vlm_export', label: 'VLM · export ONNX / TensorRT', fields: ['model', 'adapter', 'format', 'device', 'project'] },
];

export function buildRecipe(type, input = {}, defaults) {
  const p = { ...input };
  const cwd = text(p.codeDir || defaults.codeDir, 'codeDir');
  const python = text(p.pythonBin || defaults.pythonBin, 'pythonBin');
  const jobName = name(p.name);
  const selectedDevice = device(p.device, defaults.device);
  const project = text(p.project || defaults.runsDir, 'project');
  let commands;
  let artifacts = [];

  if (type === 'detect_train' || type === 'detect_finetune') {
    const data = text(p.data, 'data');
    const action = type === 'detect_train' ? 'train' : 'finetune';
    const selectedModel = type === 'detect_train' ? model(p.model) : text(p.checkpoint, 'checkpoint');
    const values = {
      data,
      model: selectedModel,
      epochs: integer(p.epochs, 'epochs', 100, 1, 10000),
      batch: integer(p.batch, 'batch', selectedDevice === 'cpu' ? 2 : 16, 1, 4096),
      imgsz: integer(p.imgsz, 'imgsz', selectedDevice === 'cpu' ? 320 : 640, 32, 4096),
      device: selectedDevice,
      project,
      name: jobName,
      exist_ok: true,
      ...augmentationValues(p),
    };
    if (type === 'detect_finetune') values.freeze = integer(p.freeze, 'freeze', 10, 0, 1000);
    commands = [detector(python, cwd, action, `Detector ${action}`, action, values)];
    artifacts = [runDir(project, jobName, cwd)];
  } else if (type === 'detect_evaluate') {
    const data = text(p.data, 'data');
    const checkpoint = text(p.checkpoint, 'checkpoint');
    const reportDir = runDir(project, jobName, cwd);
    commands = [
      detector(python, cwd, 'validate', 'Validate detector', 'val', { data, model: checkpoint, device: selectedDevice }),
      detector(python, cwd, 'report', 'Build evaluation report', 'report', {
        data, model: checkpoint, device: selectedDevice, run_dir: reportDir,
      }),
    ];
    artifacts = [reportDir];
  } else if (type === 'detect_predict') {
    const output = runDir(project, jobName, cwd);
    commands = [detector(python, cwd, 'predict', 'Run detector inference', 'predict', {
      source: text(p.source, 'source'),
      model: text(p.checkpoint, 'checkpoint'),
      conf: number(p.conf, 'conf', 0.25),
      device: selectedDevice,
      save_dir: output,
    })];
    artifacts = [output];
  } else if (type === 'detect_export') {
    const format = text(p.format || 'onnx', 'format').toLowerCase();
    if (!['onnx', 'tensorrt'].includes(format)) throw new Error('format must be onnx or tensorrt');
    if (format === 'tensorrt' && !selectedDevice.startsWith('cuda')) {
      throw new Error('TensorRT export requires a CUDA device and GPU target');
    }
    const output = path.join(runDir(project, jobName, cwd), 'model');
    commands = [detector(python, cwd, 'export', `Export detector to ${format}`, 'export', {
      model: text(p.checkpoint, 'checkpoint'),
      format,
      imgsz: integer(p.imgsz, 'imgsz', 640, 32, 4096),
      opset: integer(p.opset, 'opset', 17, 11, 21),
      workspace: integer(p.workspace, 'workspace', 4, 1, 256),
      device: selectedDevice,
      half: Boolean(p.half),
      out: output,
    })];
    artifacts = [path.dirname(output)];
  } else if (type === 'detect_tune') {
    commands = [detector(python, cwd, 'tune', 'Tune detector hyperparameters', 'tune', {
      data: text(p.data, 'data'),
      model: model(p.model),
      tune_iterations: integer(p.iterations, 'iterations', 30, 1, 1000),
      device: selectedDevice,
      project,
      name: jobName,
    })];
    artifacts = [runDir(project, jobName, cwd)];
  } else if (type.startsWith('classifier_')) {
    const action = type.slice('classifier_'.length);
    const output = runDir(project, jobName, cwd);
    const base = (operation) => [python, '-m', 'trainer', operation, 'task=classification',
      `project=${project}`, `name=${jobName}`, `device=${selectedDevice || 'auto'}`];
    if (action === 'train' || action === 'finetune') {
      const argv = base(action);
      argv.push(`data=${text(p.data, 'data')}`);
      if (action === 'train' || p.arch) argv.push(`arch=${text(p.arch || 'convnext-tiny', 'arch')}`);
      if (action === 'finetune') argv.push(`checkpoint=${text(p.checkpoint, 'checkpoint')}`);
      argv.push(
        `epochs=${integer(p.epochs, 'epochs', 30, 1, 10000)}`,
        `batch=${integer(p.batch, 'batch', selectedDevice === 'cpu' ? 8 : 32, 1, 4096)}`,
        `imgsz=${integer(p.imgsz, 'imgsz', 160, 32, 2048)}`,
        `patience=${integer(p.patience, 'patience', 20, 1, 10000)}`,
        `accumulate=${integer(p.accumulate, 'accumulate', 1, 1, 4096)}`,
      );
      if (action === 'finetune') argv.push(`freeze=${integer(p.freeze, 'freeze', 4, 0, 1000)}`);
      if (p.resume) argv.push(`resume=${text(p.resume, 'resume')}`);
      appendAugmentation(argv, p);
      commands = [command(action, `Classifier ${action}`, cwd, argv)];
    } else if (action === 'evaluate') {
      const data = text(p.data, 'data');
      const checkpoint = text(p.checkpoint, 'checkpoint');
      const batch = integer(p.batch, 'batch', 32, 1, 4096);
      const thresholds = path.join(output, 'thresholds.json');
      const evaluateArgv = base('evaluate');
      evaluateArgv.push(`data=${data}`, `checkpoint=${checkpoint}`, `batch=${batch}`);
      const thresholdArgv = base('tune-thresh');
      thresholdArgv.push(`data=${data}`, `checkpoint=${checkpoint}`, `batch=${batch}`);
      const reportArgv = base('report');
      reportArgv.push(`data=${data}`, `checkpoint=${checkpoint}`, `batch=${batch}`, `thresholds=${thresholds}`);
      commands = [
        command('evaluate', 'Evaluate classifier', cwd, evaluateArgv),
        command('tune-thresh', 'Tune per-class confidence thresholds', cwd, thresholdArgv),
        command('report', 'Build classification report', cwd, reportArgv),
      ];
    } else if (action === 'predict') {
      const argv = base('predict');
      argv.push(`source=${text(p.source, 'source')}`, `checkpoint=${text(p.checkpoint, 'checkpoint')}`);
      argv.push(`batch=${integer(p.batch, 'batch', 32, 1, 4096)}`);
      if (p.thresholds) argv.push(`thresholds=${text(p.thresholds, 'thresholds')}`);
      argv.push(`conf=${number(p.conf, 'conf', 0, 0, 1)}`, `topk=${integer(p.topk, 'topk', 3, 1, 1000)}`);
      commands = [command(action, 'Classifier predict', cwd, argv)];
    } else if (action === 'xai') {
      const argv = base('xai');
      argv.push(`source=${text(p.source, 'source')}`, `checkpoint=${text(p.checkpoint, 'checkpoint')}`,
        `target=${text(p.target || 'predicted', 'target')}`,
        `max_images=${integer(p.maxImages, 'maxImages', 24, 1, 1000)}`,
        `activation_channels=${integer(p.activationChannels, 'activationChannels', 4, 1, 16)}`,
        `intensity=${number(p.intensity, 'intensity', 0.72, 0.05, 1)}`);
      commands = [command(action, 'Classifier signed Grad-CAM and feature activations', cwd, argv)];
    } else if (action === 'export') {
      const format = text(p.format || 'onnx', 'format').toLowerCase();
      if (!['onnx', 'tensorrt'].includes(format)) throw new Error('format must be onnx or tensorrt');
      if (format === 'tensorrt' && !selectedDevice.startsWith('cuda')) {
        throw new Error('TensorRT export requires a CUDA device and GPU target');
      }
      const argv = base('export');
      argv.push(`checkpoint=${text(p.checkpoint, 'checkpoint')}`, `format=${format}`,
        `opset=${integer(p.opset, 'opset', 17, 11, 21)}`, `workspace=${integer(p.workspace, 'workspace', 4, 1, 256)}`);
      if (p.half) argv.push('half=true');
      commands = [command(action, `Classifier export ${format}`, cwd, argv)];
    } else if (action === 'tune') {
      const argv = base('tune');
      argv.push(`data=${text(p.data, 'data')}`, `arch=${text(p.arch || 'convnext-tiny', 'arch')}`,
        `iterations=${integer(p.iterations, 'iterations', 10, 1, 1000)}`,
        `tune_epochs=${integer(p.tuneEpochs, 'tuneEpochs', 5, 1, 1000)}`,
        `batch=${integer(p.batch, 'batch', selectedDevice === 'cpu' ? 8 : 32, 1, 4096)}`,
        `imgsz=${integer(p.imgsz, 'imgsz', 160, 32, 2048)}`);
      appendAugmentation(argv, p);
      commands = [command(action, 'Tune classifier hyperparameters', cwd, argv)];
    } else if (action === 'tune-thresh') {
      const argv = base(action);
      argv.push(`data=${text(p.data, 'data')}`, `checkpoint=${text(p.checkpoint, 'checkpoint')}`);
      commands = [command(action, 'Tune classifier thresholds', cwd, argv)];
    } else if (action === 'report') {
      const argv = base(action);
      argv.push(`data=${text(p.data, 'data')}`, `checkpoint=${text(p.checkpoint, 'checkpoint')}`);
      if (p.thresholds) argv.push(`thresholds=${text(p.thresholds, 'thresholds')}`);
      commands = [command(action, 'Build classification report', cwd, argv)];
    } else if (action === 'viz-augment') {
      const argv = base(action);
      argv.push(`data=${text(p.data, 'data')}`, `imgsz=${integer(p.imgsz, 'imgsz', 160, 32, 2048)}`,
        `n=${integer(p.n, 'n', 16, 1, 256)}`);
      appendAugmentation(argv, p);
      commands = [command(action, 'Preview classifier augmentation', cwd, argv)];
    } else if (action === 'bench-data') {
      const argv = base(action);
      argv.push(`data=${text(p.data, 'data')}`, `imgsz=${integer(p.imgsz, 'imgsz', 160, 32, 2048)}`,
        `batch=${integer(p.batch, 'batch', 16, 1, 4096)}`, `batches=${integer(p.batches, 'batches', 50, 1, 10000)}`);
      commands = [command(action, 'Benchmark classifier data loading', cwd, argv)];
    } else {
      throw new Error(`unknown classifier action: ${action}`);
    }
    artifacts = [output];
  } else if (type.startsWith('video_')) {
    const action = type.slice('video_'.length);
    const output = runDir(project, jobName, cwd);
    const argv = [python, '-m', 'trainer', action, 'task=video-classification',
      `arch=${text(p.arch || 'r2plus1d-tiny', 'arch')}`, `project=${project}`, `name=${jobName}`,
      `device=${selectedDevice || 'auto'}`];
    if (['train', 'evaluate'].includes(action)) argv.push(`data=${text(p.data, 'data')}`);
    if (action === 'predict') argv.push(`source=${text(p.source, 'source')}`);
    if (action !== 'train') argv.push(`checkpoint=${text(p.checkpoint, 'checkpoint')}`);
    if (action === 'train') {
      argv.push(
        `epochs=${integer(p.epochs, 'epochs', 30, 1, 10000)}`,
        `batch=${integer(p.batch, 'batch', selectedDevice === 'cpu' ? 2 : 8, 1, 512)}`,
        `imgsz=${integer(p.imgsz, 'imgsz', selectedDevice === 'cpu' ? 112 : 160, 32, 2048)}`,
        `clip_len=${integer(p.clipLen, 'clipLen', selectedDevice === 'cpu' ? 8 : 16, 2, 256)}`,
      );
      appendAugmentation(argv, p);
    } else if (action === 'evaluate') {
      argv.push(`batch=${integer(p.batch, 'batch', 4, 1, 512)}`);
    } else if (action === 'predict') {
      argv.push(`batch=${integer(p.batch, 'batch', 4, 1, 512)}`);
    } else if (action === 'export') {
      const format = text(p.format || 'onnx', 'format').toLowerCase();
      if (!['onnx', 'tensorrt'].includes(format)) throw new Error('format must be onnx or tensorrt');
      if (format === 'tensorrt' && !selectedDevice.startsWith('cuda')) {
        throw new Error('TensorRT export requires a CUDA device and GPU target');
      }
      argv.push(`format=${format}`, `opset=${integer(p.opset, 'opset', 17, 11, 21)}`,
        `workspace=${integer(p.workspace, 'workspace', 4, 1, 256)}`);
      if (p.half) argv.push('half=true');
    } else {
      throw new Error(`unknown video action: ${action}`);
    }
    commands = [command(action, `Video/action classifier ${action}`, cwd, argv)];
    artifacts = [output];
  } else if (type === 'vlm_tune') {
    const output = runDir(project, jobName, cwd);
    const argv = [
      python, '-m', 'trainer', 'tune', 'task=vlm', `arch=${text(p.arch || 'hf-lora', 'arch')}`,
      `data=${text(p.data, 'data')}`, `model=${text(p.model, 'model')}`, `project=${project}`, `name=${jobName}`,
      `device=${selectedDevice || 'auto'}`, `dtype=${dtype(p.dtype)}`,
      `epochs=${integer(p.epochs, 'epochs', 1, 1, 1000)}`, `batch=${integer(p.batch, 'batch', 1, 1, 128)}`,
      `lr=${number(p.lr, 'lr', 0.00002, 0.000000001, 1)}`,
      `max_length=${integer(p.maxLength, 'maxLength', 1024, 32, 32768)}`,
      `lora_r=${integer(p.loraR, 'loraR', 16, 1, 1024)}`,
      `lora_alpha=${integer(p.loraAlpha, 'loraAlpha', 32, 1, 8192)}`,
      `gradient_accumulation=${integer(p.gradientAccumulation, 'gradientAccumulation', 8, 1, 4096)}`,
      `target_modules=${text(p.targetModules || 'q_proj,k_proj,v_proj,o_proj', 'targetModules')}`,
    ];
    appendAugmentation(argv, p);
    if (p.trustRemoteCode) argv.push('trust_remote_code=true');
    commands = [command('tune', 'Tune VLM with LoRA', cwd, argv)];
    artifacts = [output, path.join(output, 'adapter')];
  } else if (type === 'vlm_merge') {
    const output = runDir(project, jobName, cwd);
    const argv = [
      python, '-m', 'trainer', 'merge', 'task=vlm', 'arch=hf-lora', `model=${text(p.model, 'model')}`,
      `adapter=${text(p.adapter, 'adapter')}`, `project=${project}`, `name=${jobName}`,
      `device=${selectedDevice || 'auto'}`, `dtype=${dtype(p.dtype)}`,
    ];
    if (p.trustRemoteCode) argv.push('trust_remote_code=true');
    commands = [command('merge', 'Merge VLM LoRA adapter', cwd, argv)];
    artifacts = [output];
  } else if (type === 'vlm_export') {
    const output = runDir(project, jobName, cwd);
    const format = text(p.format || 'onnx', 'format').toLowerCase();
    if (!['onnx', 'tensorrt'].includes(format)) throw new Error('format must be onnx or tensorrt');
    if (format === 'tensorrt' && !selectedDevice.startsWith('cuda')) {
      throw new Error('TensorRT export requires a CUDA device and GPU target');
    }
    const argv = [
      python, '-m', 'trainer', 'export', 'task=vlm', `model=${text(p.model, 'model')}`,
      `format=${format}`, `project=${project}`, `name=${jobName}`, `device=${selectedDevice || 'auto'}`,
      `dtype=${dtype(p.dtype)}`, `prompt=${text(p.prompt || 'Describe the image.', 'prompt')}`,
      `image_size=${integer(p.imageSize, 'imageSize', 384, 32, 4096)}`,
      `max_length=${integer(p.maxLength, 'maxLength', 4096, 32, 32768)}`,
      `opset=${integer(p.opset, 'opset', 18, 18, 21)}`,
      `workspace=${integer(p.workspace, 'workspace', 8, 1, 256)}`,
    ];
    if (p.adapter) argv.push(`adapter=${text(p.adapter, 'adapter')}`);
    if (p.sampleImage) argv.push(`sample_image=${text(p.sampleImage, 'sampleImage')}`);
    if (p.half) argv.push('half=true');
    if (p.trustRemoteCode) argv.push('trust_remote_code=true');
    commands = [command('export', `VLM export ${format}`, cwd, argv)];
    artifacts = [output];
  } else {
    throw new Error(`unknown job type: ${type}`);
  }

  return {
    name: jobName,
    params: { ...p, name: jobName, device: selectedDevice || 'auto', project, codeDir: cwd },
    commands,
    artifacts,
  };
}
