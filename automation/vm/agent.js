import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { discoverRuns } from '../server/runs.js';

const env = process.env;
const controlUrl = String(env.CONTROL_PLANE_URL || 'http://127.0.0.1:8080').replace(/\/+$/, '');
const token = env.AGENT_TOKEN || 'development-agent-token';
const agentId = env.AGENT_ID || os.hostname();
const pollMs = Number.parseInt(env.AGENT_POLL_MS || '5000', 10);
const heartbeatMs = Number.parseInt(env.AGENT_HEARTBEAT_MS || '15000', 10);
const runsRoot = path.resolve(env.VM_RUNS_DIR || '/runs');
const acceptedTypes = String(env.AGENT_TYPES || '').split(',').map((value) => value.trim()).filter(Boolean);
const gpuDisabled = String(env.AGENT_DISABLE_GPU || '').toLowerCase() === 'true';
const cancelled = new Set();
const sampleCache = new Map();
let activeChild = null;
let activeJobId = null;
let stopping = false;

const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

async function api(route, { method = 'GET', value, tries = 4 } = {}) {
  let last;
  for (let attempt = 0; attempt < tries; attempt += 1) {
    try {
      const response = await fetch(`${controlUrl}/api/agent${route}`, {
        method,
        headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
        body: value == null ? undefined : JSON.stringify(value),
      });
      if (!response.ok) throw new Error(`${route}: HTTP ${response.status} ${await response.text()}`);
      return response.json();
    } catch (error) {
      last = error;
      if (attempt + 1 < tries) await delay(Math.min(8000, 500 * 2 ** attempt));
    }
  }
  throw last;
}

function gpuInventory() {
  if (gpuDisabled) return [];
  const result = spawnSync('nvidia-smi', [
    '--query-gpu=index,name,memory.total,memory.free,compute_cap', '--format=csv,noheader,nounits',
  ], { encoding: 'utf8', timeout: 5000 });
  if (result.status !== 0) return [];
  return result.stdout.trim().split('\n').filter(Boolean).map((line) => {
    const [index, name, memoryTotalMb, memoryFreeMb, computeCapability] = line.split(',').map((value) => value.trim());
    return { index: Number(index), name, memoryTotalMb: Number(memoryTotalMb), memoryFreeMb: Number(memoryFreeMb), computeCapability };
  });
}

function diskInventory() {
  const result = spawnSync('df', ['-Pk', runsRoot], { encoding: 'utf8', timeout: 5000 });
  const line = result.status === 0 ? result.stdout.trim().split('\n').at(-1) : '';
  const fields = line.trim().split(/\s+/);
  return fields.length >= 6 ? { totalKb: Number(fields[1]), usedKb: Number(fields[2]), freeKb: Number(fields[3]), mount: fields[5] } : null;
}

function capabilities() {
  return {
    gpus: gpuInventory(), cpus: os.cpus().length,
    memoryTotalMb: Math.round(os.totalmem() / 1024 / 1024),
    memoryFreeMb: Math.round(os.freemem() / 1024 / 1024),
    platform: `${os.platform()}-${os.arch()}`,
  };
}

async function heartbeat() {
  const current = capabilities();
  const response = await api('/heartbeat', {
    method: 'POST', value: {
      agentId,
      label: env.AGENT_LABEL || agentId,
      provider: env.AGENT_PROVIDER || '',
      machineType: env.AGENT_MACHINE_TYPE || '',
      host: os.hostname(),
      ...current,
      disk: diskInventory(),
      activeJobId,
    },
  });
  for (const id of response.cancels || []) cancelled.add(id);
  if (activeJobId && cancelled.has(activeJobId) && activeChild) {
    process.kill(-activeChild.pid, 'SIGTERM');
    setTimeout(() => {
      if (activeChild && activeJobId && cancelled.has(activeJobId)) {
        try { process.kill(-activeChild.pid, 'SIGKILL'); } catch { /* already exited */ }
      }
    }, 10_000).unref();
  }
  return current;
}

function checkpoints(root, depth = 0) {
  if (depth > 4 || !fs.existsSync(root)) return [];
  const output = [];
  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    const full = path.join(root, entry.name);
    if (entry.isDirectory()) output.push(...checkpoints(full, depth + 1));
    else if (['best.pt', 'last.pt'].includes(entry.name)) output.push(full);
    else if (entry.name === 'adapter_config.json') output.push(root);
  }
  return [...new Set(output)].slice(0, 1000);
}

async function inventory() {
  await api('/inventory', {
    method: 'POST', value: {
      agentId, runsRoot, checkpoints: checkpoints(runsRoot),
      runs: discoverRuns(runsRoot, { limit: 500 }), scannedAt: Date.now(),
    },
  });
}

async function pushLog(jobId, lines) {
  if (!lines.length) return;
  try { await api(`/jobs/${jobId}/log`, { method: 'POST', value: { lines: lines.slice(-200) } }); }
  catch (error) { console.error(`[agent] log upload failed: ${error.message}`); }
}

function printable(command) {
  return command.argv.map((value) => /[\s"']/.test(value) ? JSON.stringify(value) : value).join(' ');
}

async function downloadSample(id) {
  if (sampleCache.has(id) && fs.existsSync(sampleCache.get(id))) return sampleCache.get(id);
  const response = await fetch(`${controlUrl}/api/agent/samples/${encodeURIComponent(id)}`, {
    headers: { authorization: `Bearer ${token}` },
  });
  if (!response.ok) throw new Error(`sample ${id}: HTTP ${response.status} ${await response.text()}`);
  const types = { 'image/jpeg': '.jpg', 'image/png': '.png', 'image/webp': '.webp' };
  const contentType = String(response.headers.get('content-type') || '').split(';')[0].toLowerCase();
  const extension = types[contentType];
  if (!extension) throw new Error(`sample ${id}: unsupported content type ${contentType || '(missing)'}`);
  const value = Buffer.from(await response.arrayBuffer());
  if (!value.length || value.length > 12 * 1024 * 1024) throw new Error(`sample ${id}: invalid size`);
  const directory = path.join(runsRoot, '.samples');
  fs.mkdirSync(directory, { recursive: true });
  const destination = path.join(directory, `${id}${extension}`);
  const temporary = `${destination}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, value);
  fs.renameSync(temporary, destination);
  sampleCache.set(id, destination);
  return destination;
}

async function materializeSamples(command) {
  const argv = [];
  for (const value of command.argv) {
    const match = String(value).match(/^(source=)?sample:\/\/([A-Za-z0-9_-]{12,80})$/);
    if (!match) argv.push(value);
    else argv.push(`${match[1] || ''}${await downloadSample(match[2])}`);
  }
  return { ...command, argv };
}

async function syncClassifierXai(job) {
  if (job.type !== 'classifier_xai') return 0;
  const files = new Set();
  for (const artifact of job.artifacts || []) {
    const runRoot = path.resolve(artifact);
    const manifestPath = path.join(runRoot, 'xai', 'manifest.json');
    if (!fs.existsSync(manifestPath)) continue;
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    for (const item of manifest.images || []) {
      for (const key of ['input', 'signed_map', 'activations']) {
        if (!item[key]) continue;
        const file = path.resolve(runRoot, item[key]);
        if (file.startsWith(`${runRoot}${path.sep}`) && fs.existsSync(file)) files.add(file);
      }
    }
    files.add(manifestPath);
  }
  let uploaded = 0;
  const ordered = [...files].sort((left, right) => Number(left.endsWith('manifest.json')) - Number(right.endsWith('manifest.json')));
  for (const file of ordered) {
    const relativePath = path.relative(runsRoot, file).split(path.sep).join('/');
    if (!relativePath || relativePath.startsWith('../')) continue;
    const stat = fs.statSync(file);
    if (!stat.isFile() || stat.size > 12 * 1024 * 1024) continue;
    await api('/xai/upload', {
      method: 'POST', value: { relativePath, data: fs.readFileSync(file).toString('base64') },
    });
    uploaded += 1;
  }
  return uploaded;
}

async function runCommand(job, command) {
  if (!Array.isArray(command.argv) || !command.argv.length) throw new Error(`step ${command.id} has no argv`);
  if (job.dryRun) {
    await pushLog(job.id, [`[dry-run] cwd=${command.cwd}`, `[dry-run] ${printable(command)}`]);
    return 0;
  }
  if (!fs.existsSync(command.cwd) || !fs.statSync(command.cwd).isDirectory()) {
    throw new Error(`working directory does not exist: ${command.cwd}`);
  }
  const runnable = await materializeSamples(command);
  await pushLog(job.id, [`[agent] cwd=${runnable.cwd}`, `[agent] ${printable(runnable)}`]);
  return new Promise((resolve, reject) => {
    const child = spawn(runnable.argv[0], runnable.argv.slice(1), {
      cwd: runnable.cwd, env: { ...process.env, ...(runnable.env || {}) },
      shell: false, detached: true, stdio: ['ignore', 'pipe', 'pipe'],
    });
    activeChild = child;
    let buffered = [];
    let partial = '';
    let uploading = Promise.resolve();
    const flush = () => {
      if (!buffered.length) return;
      const lines = buffered; buffered = [];
      uploading = uploading.then(() => pushLog(job.id, lines));
    };
    const accept = (chunk, prefix = '') => {
      const pieces = `${partial}${chunk.toString('utf8')}`.split(/\r?\n/);
      partial = pieces.pop() || '';
      buffered.push(...pieces.map((line) => `${prefix}${line}`).filter((line) => line.trim()));
      if (buffered.length >= 20) flush();
    };
    const timer = setInterval(flush, 1000);
    child.stdout.on('data', (chunk) => accept(chunk));
    child.stderr.on('data', (chunk) => accept(chunk, '[stderr] '));
    child.on('error', (error) => { clearInterval(timer); activeChild = null; reject(error); });
    child.on('close', async (code, signal) => {
      clearInterval(timer);
      if (partial) buffered.push(partial);
      flush(); await uploading; activeChild = null;
      if (signal) await pushLog(job.id, [`[agent] process ended by ${signal}`]);
      resolve(code ?? (signal ? 143 : 1));
    });
  });
}

async function update(jobId, value) {
  return api(`/jobs/${jobId}/update`, { method: 'POST', value });
}

async function runJob(job) {
  activeJobId = job.id;
  await pushLog(job.id, [`[agent] ${agentId} claimed ${job.type} at ${new Date().toISOString()}`]);
  try {
    for (const command of job.commands) {
      const existing = job.steps.find((step) => step.id === command.id);
      if (existing?.status === 'completed') continue;
      if (cancelled.has(job.id)) throw new Error('cancelled by operator');
      const startedAt = Date.now();
      await update(job.id, { step: { id: command.id, status: 'running', startedAt } });
      let exitCode;
      try { exitCode = await runCommand(job, command); }
      catch (error) {
        await update(job.id, { step: { id: command.id, status: 'failed', startedAt, endedAt: Date.now(), exitCode: 1 } });
        throw error;
      }
      const wasCancelled = cancelled.has(job.id);
      const status = wasCancelled ? 'cancelled' : exitCode === 0 ? 'completed' : 'failed';
      await update(job.id, { step: { id: command.id, status, startedAt, endedAt: Date.now(), exitCode } });
      if (wasCancelled) throw new Error('cancelled by operator');
      if (exitCode !== 0) throw new Error(`step ${command.id} exited with code ${exitCode}`);
    }
    const artifacts = (job.artifacts || []).filter((artifact) => fs.existsSync(artifact));
    try {
      const synced = await syncClassifierXai({ ...job, artifacts });
      if (synced) await pushLog(job.id, [`[agent] synced ${synced} xAI evidence file(s) to the control plane`]);
    } catch (error) {
      await pushLog(job.id, [`[agent] xAI evidence sync warning: ${error.message || error}`]);
    }
    await pushLog(job.id, ['[agent] job completed']);
    // Terminal status may trigger VM deallocation, so upload the final log first.
    await update(job.id, { status: 'completed', artifacts });
  } catch (error) {
    const status = cancelled.has(job.id) ? 'cancelled' : 'failed';
    await pushLog(job.id, [`[agent] ${status}: ${error.message || error}`]);
    await update(job.id, { status, error: String(error.message || error) });
  } finally {
    cancelled.delete(job.id); activeJobId = null; activeChild = null;
    inventory().catch((error) => console.error(`[agent] inventory failed: ${error.message}`));
  }
}

async function main() {
  console.log(`[agent] ${agentId} polling ${controlUrl}`);
  let current = await heartbeat();
  inventory().catch(() => {});
  const heartbeatTimer = setInterval(() => heartbeat().then((value) => { current = value; }).catch((error) => {
    console.error(`[agent] heartbeat failed: ${error.message}`);
  }), heartbeatMs);
  const inventoryTimer = setInterval(() => inventory().catch(() => {}), 60_000);
  while (!stopping) {
    try {
      const { job } = await api('/claim', {
        method: 'POST', value: { agentId, types: acceptedTypes.length ? acceptedTypes : undefined, capabilities: current },
      });
      if (job) await runJob(job); else await delay(pollMs);
    } catch (error) {
      console.error(`[agent] poll failed: ${error.message}`); await delay(Math.max(pollMs, 5000));
    }
  }
  clearInterval(heartbeatTimer); clearInterval(inventoryTimer);
}

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    stopping = true;
    if (activeChild) { try { process.kill(-activeChild.pid, 'SIGTERM'); } catch { /* already exited */ } }
  });
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
