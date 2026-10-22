import path from 'node:path';

const env = process.env;

function bool(name, fallback) {
  const value = env[name];
  return value == null ? fallback : value.toLowerCase() === 'true';
}

export function parseManagedTargets(value = '') {
  if (!String(value).trim()) return {};
  const parsed = JSON.parse(value);
  if (!Array.isArray(parsed)) throw new Error('MANAGED_TARGETS_JSON must be a JSON array');
  const targets = {};
  for (const entry of parsed) {
    const agentId = String(entry?.agentId || '').trim();
    if (!/^[A-Za-z0-9_-]+$/.test(agentId)) {
      throw new Error(`managed target has invalid agentId: ${agentId || '(empty)'}`);
    }
    targets[agentId] = {
      ...entry,
      agentId,
      configured: true,
      gpus: Array.isArray(entry.gpus) ? entry.gpus : [],
      at: 0,
    };
  }
  return targets;
}

export const config = {
  host: env.HOST || '0.0.0.0',
  port: Number.parseInt(env.PORT || '8080', 10),
  dataDir: path.resolve(env.AUTOMATION_DATA_DIR || '.automation-data'),
  adminUser: env.ADMIN_USER || 'admin',
  adminPassword: env.ADMIN_PASSWORD || 'change-me',
  sessionSecret: env.SESSION_SECRET || 'development-secret-change-me',
  secureCookies: bool('SECURE_COOKIES', Boolean(env.K_SERVICE || env.CONTAINER_APP_NAME)),
  agentToken: env.AGENT_TOKEN || 'development-agent-token',
  managedTargets: parseManagedTargets(env.MANAGED_TARGETS_JSON),
  autoShutdown: bool('AUTO_SHUTDOWN', false),
  powerDriver: env.POWER_DRIVER || 'noop',
  azure: {
    tenantId: env.AZURE_TENANT_ID || '',
    clientId: env.AZURE_CLIENT_ID || '',
    clientSecret: env.AZURE_CLIENT_SECRET || '',
    subscriptionId: env.AZURE_SUBSCRIPTION_ID || '',
    resourceGroup: env.AZURE_RESOURCE_GROUP || '',
    vmName: env.AZURE_VM_NAME || '',
  },
  gce: {
    project: env.GCE_PROJECT || '',
    zone: env.GCE_ZONE || '',
    instance: env.GCE_INSTANCE || '',
    accessToken: env.GCE_ACCESS_TOKEN || '',
  },
  vmDefaults: {
    codeDir: env.VM_CODE_DIR || '/opt/trainer',
    pythonBin: env.VM_PYTHON || 'python3',
    dataDir: env.VM_DATA_DIR || '/data',
    runsDir: env.VM_RUNS_DIR || '/runs',
    device: env.VM_DEVICE || 'cpu',
  },
};
