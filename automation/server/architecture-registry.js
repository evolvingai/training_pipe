import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);
let cached = null;

const TASK_LABELS = {
  detection: 'Detection',
  classification: 'Image Classification',
  'video-classification': 'Video / Action Classification',
  vlm: 'VLM Tuning',
};

export function normalizeArchitectureCatalog(value = {}) {
  const tasks = Array.isArray(value.tasks) ? value.tasks : [];
  const architectures = Array.isArray(value.architectures) ? value.architectures : [];
  return {
    generatedAt: Date.now(),
    tasks: tasks.map((task) => ({
      name: String(task.name || ''),
      label: TASK_LABELS[task.name] || String(task.name || 'Task'),
      actions: Array.isArray(task.actions) ? task.actions.map(String) : [],
      defaultArchitecture: String(task.default_architecture || ''),
      description: String(task.description || ''),
    })).filter((task) => task.name),
    architectures: architectures.map((architecture) => ({
      task: String(architecture.task || ''),
      name: String(architecture.name || ''),
      description: String(architecture.description || ''),
      aliases: Array.isArray(architecture.aliases) ? architecture.aliases.map(String) : [],
      family: String(architecture.family || 'External plugin'),
      inputContract: String(architecture.input_contract || 'Defined by the plugin.'),
      outputContract: String(architecture.output_contract || 'Defined by the plugin.'),
      trainingStrategy: String(architecture.training_strategy || 'Defined by the plugin.'),
      stages: Array.isArray(architecture.stages) && architecture.stages.length
        ? architecture.stages.map(String)
        : ['input', 'architecture plugin', 'outputs'],
    })).filter((architecture) => architecture.task && architecture.name),
  };
}

export async function loadArchitectureCatalog(config, { refresh = false } = {}) {
  if (!refresh && cached && Date.now() - cached.at < 30_000) return cached.value;
  const { stdout } = await execFileAsync(
    config.vmDefaults.pythonBin,
    ['-m', 'trainer', 'list'],
    {
      cwd: config.vmDefaults.codeDir,
      env: process.env,
      timeout: 15_000,
      maxBuffer: 2 * 1024 * 1024,
    },
  );
  const value = normalizeArchitectureCatalog(JSON.parse(stdout));
  cached = { at: Date.now(), value };
  return value;
}

