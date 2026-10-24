import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { clearSessionCookie, makeSession, sessionCookie, sessionUser, validAgent, validLogin } from './auth.js';
import { loadArchitectureCatalog } from './architecture-registry.js';
import { config } from './config.js';
import { evidenceSyncLimits, saveSyncedEvidence } from './evidence.js';
import { makePowerDriver } from './power.js';
import { buildRecipe, recipeCatalog } from './recipes.js';
import { discoverRuns } from './runs.js';
import { resolveSample, sampleLimits, saveSample } from './samples.js';
import { FileStore, newId } from './store.js';
import { listClassificationXaiResults, resolveClassificationXaiAsset } from './xai-results.js';

const moduleDir = path.dirname(fileURLToPath(import.meta.url));
const publicDir = path.resolve(moduleDir, '..', 'public');
const store = new FileStore(config.dataDir);
const power = makePowerDriver(config);

function visibleAgents(liveAgents = {}) {
  const output = {};
  for (const [agentId, target] of Object.entries(config.managedTargets || {})) {
    output[agentId] = { ...target };
  }
  for (const [agentId, agent] of Object.entries(liveAgents || {})) {
    output[agentId] = { ...(output[agentId] || {}), ...agent, agentId };
  }
  return output;
}

function send(res, status, body, headers = {}) {
  const payload = body == null ? '' : JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store', ...headers });
  res.end(payload);
}

function error(res, status, value) {
  send(res, status, { error: String(value?.message || value) });
}

function architectureCatalog(refresh = false) {
  return loadArchitectureCatalog(config, { refresh }).catch((reason) => ({
    generatedAt: Date.now(), tasks: [], architectures: [], warning: String(reason?.message || reason),
  }));
}

async function body(req, maxBytes = 1_000_000) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > maxBytes) throw new Error('request body is too large');
    chunks.push(chunk);
  }
  if (!chunks.length) return {};
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}

function sendSample(res, sample) {
  res.writeHead(200, {
    'content-type': sample.type,
    'content-length': sample.stat.size,
    'cache-control': 'private, max-age=3600',
    'x-content-type-options': 'nosniff',
    'x-trainer-sample-name': encodeURIComponent(sample.name),
  });
  fs.createReadStream(sample.path).pipe(res);
  return true;
}

function jobId(pathname, suffix = '') {
  const expression = suffix
    ? new RegExp(`^/api/jobs/([A-Za-z0-9_-]+)/${suffix}$`)
    : /^\/api\/jobs\/([A-Za-z0-9_-]+)$/;
  return pathname.match(expression)?.[1] || null;
}

function agentJobId(pathname, action) {
  return pathname.match(new RegExp(`^/api/agent/jobs/([A-Za-z0-9_-]+)/${action}$`))?.[1] || null;
}

function safeStep(step) {
  const allowed = ['pending', 'running', 'completed', 'failed', 'cancelled', 'skipped'];
  if (!step?.id || !allowed.includes(step.status)) throw new Error('invalid step update');
  return {
    id: String(step.id),
    status: step.status,
    ...(step.startedAt ? { startedAt: Number(step.startedAt) } : {}),
    ...(step.endedAt ? { endedAt: Number(step.endedAt) } : {}),
    ...(Number.isInteger(step.exitCode) ? { exitCode: step.exitCode } : {}),
  };
}

async function createJob(type, params, createdBy, dryRun = false) {
  const recipe = buildRecipe(type, params, config.vmDefaults);
  const now = Date.now();
  const job = {
    id: newId(), type, name: recipe.name, params: recipe.params,
    targetAgentId: String(params.targetAgentId || '').trim() || null,
    createdBy, dryRun: Boolean(dryRun), status: 'queued', cancelRequested: false,
    commands: recipe.commands,
    steps: recipe.commands.map(({ id, title }) => ({ id, title, status: 'pending' })),
    artifacts: recipe.artifacts, logTail: [], error: null, agentId: null,
    createdAt: now, updatedAt: now, startedAt: null, endedAt: null,
  };
  await store.createJob(job);
  if (power.configured) {
    power.start().catch((reason) => store.appendLog(job.id, [`[control] VM start warning: ${reason.message || reason}`]));
  }
  return job;
}

async function maybeAutoShutdown(status) {
  if (!config.autoShutdown || !power.configured || !['completed', 'failed', 'cancelled'].includes(status)) return false;
  const jobs = await store.listJobs(500);
  if (jobs.some((job) => ['queued', 'running'].includes(job.status))) return false;
  await power.stop();
  return true;
}

async function api(req, res, url) {
  const pathname = url.pathname;

  if (req.method === 'POST' && pathname === '/api/login') {
    const input = await body(req);
    if (!validLogin(input.username, input.password, config.adminUser, config.adminPassword)) {
      return error(res, 401, 'wrong username or password');
    }
    return send(res, 200, { ok: true, username: config.adminUser }, {
      'set-cookie': sessionCookie(makeSession(config.adminUser, config.sessionSecret), config.secureCookies),
    });
  }
  if (req.method === 'POST' && pathname === '/api/logout') {
    return send(res, 200, { ok: true }, { 'set-cookie': clearSessionCookie(config.secureCookies) });
  }
  if (req.method === 'GET' && pathname === '/api/me') {
    const username = sessionUser(req, config.sessionSecret);
    return send(res, 200, { authed: Boolean(username), username });
  }

  if (pathname.startsWith('/api/agent/')) return agentApi(req, res, url);

  const username = sessionUser(req, config.sessionSecret);
  if (!username) return error(res, 401, 'authentication required');

  if (req.method === 'GET' && pathname === '/api/config') {
    const registry = await architectureCatalog();
    return send(res, 200, {
      recipes: recipeCatalog, defaults: config.vmDefaults, autoShutdown: config.autoShutdown,
      power: { name: power.name, configured: power.configured },
      registry,
    });
  }
  if (req.method === 'GET' && pathname === '/api/architectures') {
    return send(res, 200, await architectureCatalog(url.searchParams.get('refresh') === 'true'));
  }
  if (req.method === 'POST' && pathname === '/api/samples') {
    try {
      const sample = saveSample(config.dataDir, await body(req, sampleLimits.requestBytes));
      return send(res, 201, { sample: { ...sample, url: `/api/samples/${sample.id}` } });
    } catch (reason) {
      return error(res, 400, reason);
    }
  }
  const sampleId = pathname.match(/^\/api\/samples\/([A-Za-z0-9_-]+)$/)?.[1];
  if (req.method === 'GET' && sampleId) {
    try { return sendSample(res, resolveSample(config.dataDir, sampleId)); }
    catch (reason) { return error(res, 404, reason); }
  }
  if (req.method === 'GET' && pathname === '/api/jobs') {
    const limit = Math.min(Number.parseInt(url.searchParams.get('limit') || '100', 10), 500);
    return send(res, 200, { jobs: await store.listJobs(limit) });
  }
  if (req.method === 'GET' && pathname === '/api/runs') {
    const limit = Math.min(Number.parseInt(url.searchParams.get('limit') || '500', 10), 1_000);
    const local = discoverRuns(config.vmDefaults.runsDir, { limit });
    const localPaths = new Set(local.map((run) => path.resolve(run.path)));
    const inventory = (await store.getState('inventory')) || {};
    const remote = Object.entries(inventory).flatMap(([agentId, value]) => (value.runs || [])
      .filter((run) => !localPaths.has(path.resolve(String(run.path || ''))))
      .map((run) => ({ ...run, id: `${agentId}:${run.id}`, agentId, remote: true })));
    const runs = [
      ...local.map((run) => ({ ...run, agentId: 'control-plane', remote: false })),
      ...remote,
    ].sort((left, right) => Number(right.updatedAt || 0) - Number(left.updatedAt || 0)).slice(0, limit);
    return send(res, 200, { root: config.vmDefaults.runsDir, runs });
  }
  if (req.method === 'POST' && pathname === '/api/preview') {
    const input = await body(req);
    try {
      const recipe = buildRecipe(input.type, input.params || {}, config.vmDefaults);
      return send(res, 200, { commands: recipe.commands, artifacts: recipe.artifacts });
    } catch (reason) {
      return error(res, 400, reason);
    }
  }
  if (req.method === 'POST' && pathname === '/api/jobs') {
    const input = await body(req);
    try {
      return send(res, 201, { job: await createJob(input.type, input.params || {}, username, input.dryRun) });
    } catch (reason) {
      return error(res, 400, reason);
    }
  }
  const getId = jobId(pathname);
  if (req.method === 'GET' && getId) {
    const job = await store.getJob(getId);
    return job ? send(res, 200, { job }) : error(res, 404, 'job not found');
  }
  const cancelId = jobId(pathname, 'cancel');
  if (req.method === 'POST' && cancelId) {
    const job = await store.getJob(cancelId);
    if (!job) return error(res, 404, 'job not found');
    const patch = job.status === 'queued'
      ? { status: 'cancelled', endedAt: Date.now(), cancelRequested: true }
      : { cancelRequested: true };
    return send(res, 200, { job: await store.updateJob(cancelId, patch) });
  }
  const retryId = jobId(pathname, 'retry');
  if (req.method === 'POST' && retryId) {
    const source = await store.getJob(retryId);
    if (!source) return error(res, 404, 'job not found');
    try {
      return send(res, 201, { job: await createJob(source.type, { ...source.params, retryOf: source.id }, username, source.dryRun) });
    } catch (reason) {
      return error(res, 400, reason);
    }
  }
  if (req.method === 'GET' && pathname === '/api/agents') {
    return send(res, 200, { agents: visibleAgents((await store.getState('agents')) || {}) });
  }
  if (req.method === 'GET' && pathname === '/api/inventory') {
    return send(res, 200, { inventory: (await store.getState('inventory')) || {} });
  }
  if (req.method === 'GET' && pathname === '/api/xai/classification') {
    return send(res, 200, {
      runs: listClassificationXaiResults(config.vmDefaults.runsDir, {
        runId: String(url.searchParams.get('run') || ''),
        limit: Math.min(Number.parseInt(url.searchParams.get('limit') || '100', 10), 200),
      }),
    });
  }
  const xaiAsset = pathname.match(/^\/api\/xai\/classification\/([^/]+)\/samples\/(\d+)\/(input|signed-map|activations)$/);
  if (req.method === 'GET' && xaiAsset) {
    try {
      const file = resolveClassificationXaiAsset(
        config.vmDefaults.runsDir, decodeURIComponent(xaiAsset[1]), Number(xaiAsset[2]), xaiAsset[3],
      );
      res.writeHead(200, {
        'content-type': file.contentType,
        'content-length': file.stat.size,
        'cache-control': 'private, max-age=60',
        'x-content-type-options': 'nosniff',
      });
      fs.createReadStream(file.path).pipe(res);
      return true;
    } catch (reason) {
      return error(res, 404, reason);
    }
  }
  if (req.method === 'GET' && pathname === '/api/vm') {
    return send(res, 200, { driver: await power.status() });
  }
  if (req.method === 'POST' && pathname === '/api/vm/start') {
    await power.start();
    return send(res, 200, { ok: true });
  }
  if (req.method === 'POST' && pathname === '/api/vm/stop') {
    const input = await body(req);
    const active = (await store.listJobs(500)).find((job) => job.status === 'running');
    if (active && !input.force) return error(res, 409, `job ${active.name} is running`);
    await power.stop();
    return send(res, 200, { ok: true });
  }
  return false;
}

async function agentApi(req, res, url) {
  if (!validAgent(req, config.agentToken)) return error(res, 401, 'invalid agent token');
  const pathname = url.pathname;

  const sampleId = pathname.match(/^\/api\/agent\/samples\/([A-Za-z0-9_-]+)$/)?.[1];
  if (req.method === 'GET' && sampleId) {
    try { return sendSample(res, resolveSample(config.dataDir, sampleId)); }
    catch (reason) { return error(res, 404, reason); }
  }
  if (req.method === 'POST' && pathname === '/api/agent/xai/upload') {
    try {
      const saved = saveSyncedEvidence(config.vmDefaults.runsDir, await body(req, evidenceSyncLimits.requestBytes));
      return send(res, 201, { evidence: saved });
    } catch (reason) {
      return error(res, 400, reason);
    }
  }

  if (req.method === 'POST' && pathname === '/api/agent/heartbeat') {
    const input = await body(req);
    const agentId = String(input.agentId || 'agent');
    const agents = (await store.getState('agents')) || {};
    agents[agentId] = { ...input, agentId, at: Date.now() };
    await store.setState('agents', agents);
    const cancels = (await store.listJobs(500))
      .filter((job) => job.status === 'running' && job.cancelRequested && job.agentId === agentId)
      .map((job) => job.id);
    return send(res, 200, { ok: true, cancels });
  }
  if (req.method === 'POST' && pathname === '/api/agent/claim') {
    const input = await body(req);
    const job = await store.claimJob(String(input.agentId || 'agent'), {
      types: Array.isArray(input.types) ? input.types : undefined,
      capabilities: input.capabilities || {},
    });
    return send(res, 200, { job });
  }
  const fetchId = pathname.match(/^\/api\/agent\/jobs\/([A-Za-z0-9_-]+)$/)?.[1];
  if (req.method === 'GET' && fetchId) {
    const job = await store.getJob(fetchId);
    return job ? send(res, 200, { job }) : error(res, 404, 'job not found');
  }
  const logId = agentJobId(pathname, 'log');
  if (req.method === 'POST' && logId) {
    const input = await body(req);
    await store.appendLog(logId, Array.isArray(input.lines) ? input.lines.slice(0, 200) : []);
    return send(res, 200, { ok: true });
  }
  const updateId = agentJobId(pathname, 'update');
  if (req.method === 'POST' && updateId) {
    const input = await body(req);
    const job = await store.getJob(updateId);
    if (!job) return error(res, 404, 'job not found');
    const patch = {};
    if (input.step) {
      const next = safeStep(input.step);
      patch.steps = job.steps.map((step) => step.id === next.id ? { ...step, ...next } : step);
    }
    if (input.status) {
      if (!['running', 'completed', 'failed', 'cancelled'].includes(input.status)) throw new Error('invalid job status');
      patch.status = input.status;
      if (input.status !== 'running') patch.endedAt = Date.now();
    }
    if (input.error != null) patch.error = String(input.error).slice(0, 4000);
    if (Array.isArray(input.artifacts)) patch.artifacts = input.artifacts.map(String).slice(0, 100);
    const updated = await store.updateJob(updateId, patch);
    let shutdown = false;
    try { shutdown = await maybeAutoShutdown(patch.status); } catch (reason) {
      await store.appendLog(updateId, [`[control] auto-shutdown warning: ${reason.message || reason}`]);
    }
    return send(res, 200, { job: updated, shutdown });
  }
  if (req.method === 'POST' && pathname === '/api/agent/inventory') {
    const input = await body(req);
    const inventory = (await store.getState('inventory')) || {};
    inventory[String(input.agentId || 'agent')] = { ...input, at: Date.now() };
    await store.setState('inventory', inventory);
    return send(res, 200, { ok: true });
  }
  return false;
}

function staticFile(req, res, pathname) {
  if (req.method !== 'GET') return false;
  const relative = pathname === '/' ? 'index.html' : pathname.replace(/^\/+/, '');
  const file = path.resolve(publicDir, relative);
  if (!file.startsWith(`${publicDir}${path.sep}`) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) return false;
  const contentType = file.endsWith('.js') ? 'text/javascript' : file.endsWith('.css') ? 'text/css' : 'text/html';
  res.writeHead(200, { 'content-type': `${contentType}; charset=utf-8`, 'cache-control': 'no-store' });
  fs.createReadStream(file).pipe(res);
  return true;
}

export function createServer() {
  return http.createServer(async (req, res) => {
    const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
    try {
      if (url.pathname.startsWith('/api/')) {
        const handled = await api(req, res, url);
        if (handled === false && !res.writableEnded) error(res, 404, 'not found');
        return;
      }
      if (!staticFile(req, res, url.pathname)) error(res, 404, 'not found');
    } catch (reason) {
      if (!res.writableEnded) error(res, reason instanceof SyntaxError ? 400 : 500, reason);
    }
  });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  createServer().listen(config.port, config.host, () => {
    console.log(`Trainer automation listening on ${config.host}:${config.port} (power=${power.name}, data=${config.dataDir})`);
  });
}
