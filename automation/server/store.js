import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const MAX_LOG_LINES = 800;

export function newId() {
  return `${Date.now().toString(36)}-${crypto.randomBytes(4).toString('hex')}`;
}

function cleanId(value) {
  return String(value || '').replace(/[^A-Za-z0-9_-]/g, '');
}

function matchesTarget(agentId, target) {
  if (!target) return true;
  const normalize = (value) => String(value || '').replace(/-eval$/i, '').toLowerCase();
  return normalize(agentId) === normalize(target);
}

function deviceCompatible(job, capabilities = {}) {
  const device = String(job.params?.device || '').toLowerCase();
  if (!device.startsWith('cuda')) return true;
  return Array.isArray(capabilities.gpus) && capabilities.gpus.length > 0;
}

export class FileStore {
  constructor(root) {
    this.root = root;
    this.jobsDir = path.join(root, 'jobs');
    this.stateDir = path.join(root, 'state');
    fs.mkdirSync(this.jobsDir, { recursive: true });
    fs.mkdirSync(this.stateDir, { recursive: true });
  }

  jobPath(id) {
    return path.join(this.jobsDir, `${cleanId(id)}.json`);
  }

  atomicWrite(file, value) {
    const temporary = `${file}.${process.pid}.tmp`;
    fs.writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`, { mode: 0o600 });
    fs.renameSync(temporary, file);
  }

  async createJob(job) {
    this.atomicWrite(this.jobPath(job.id), job);
    return job;
  }

  async getJob(id) {
    try {
      return JSON.parse(fs.readFileSync(this.jobPath(id), 'utf8'));
    } catch {
      return null;
    }
  }

  async updateJob(id, patch) {
    const job = await this.getJob(id);
    if (!job) return null;
    const updated = { ...job, ...patch, updatedAt: Date.now() };
    this.atomicWrite(this.jobPath(id), updated);
    return updated;
  }

  async appendLog(id, lines) {
    const job = await this.getJob(id);
    if (!job) return null;
    const clean = (lines || []).map(String).filter((line) => line.trim());
    return this.updateJob(id, { logTail: [...(job.logTail || []), ...clean].slice(-MAX_LOG_LINES) });
  }

  async listJobs(limit = 100) {
    const jobs = [];
    for (const file of fs.readdirSync(this.jobsDir)) {
      if (!file.endsWith('.json')) continue;
      try {
        jobs.push(JSON.parse(fs.readFileSync(path.join(this.jobsDir, file), 'utf8')));
      } catch {
        // A corrupt job must not make the queue unavailable.
      }
    }
    return jobs.sort((a, b) => b.createdAt - a.createdAt).slice(0, limit);
  }

  async claimJob(agentId, { types, capabilities } = {}) {
    const jobs = (await this.listJobs(1000))
      .filter((job) => job.status === 'queued')
      .filter((job) => !types?.length || types.includes(job.type))
      .filter((job) => matchesTarget(agentId, job.targetAgentId))
      .filter((job) => deviceCompatible(job, capabilities))
      .sort((a, b) => a.createdAt - b.createdAt);
    if (!jobs.length) return null;
    const job = jobs[0];
    return this.updateJob(job.id, {
      status: 'running',
      agentId,
      startedAt: job.startedAt || Date.now(),
    });
  }

  async getState(key) {
    try {
      return JSON.parse(fs.readFileSync(path.join(this.stateDir, `${cleanId(key)}.json`), 'utf8'));
    } catch {
      return null;
    }
  }

  async setState(key, value) {
    this.atomicWrite(path.join(this.stateDir, `${cleanId(key)}.json`), value);
    return value;
  }
}
