function configured(values) {
  return values.every(Boolean);
}

function azureDriver(azure) {
  const ready = configured([
    azure.tenantId, azure.clientId, azure.clientSecret, azure.subscriptionId, azure.resourceGroup, azure.vmName,
  ]);
  const base = `https://management.azure.com/subscriptions/${azure.subscriptionId}/resourceGroups/${azure.resourceGroup}/providers/Microsoft.Compute/virtualMachines/${azure.vmName}`;
  let cachedToken;

  async function token() {
    if (cachedToken?.expiresAt > Date.now() + 60_000) return cachedToken.value;
    const body = new URLSearchParams({
      grant_type: 'client_credentials', client_id: azure.clientId,
      client_secret: azure.clientSecret, scope: 'https://management.azure.com/.default',
    });
    const response = await fetch(`https://login.microsoftonline.com/${azure.tenantId}/oauth2/v2.0/token`, {
      method: 'POST', headers: { 'content-type': 'application/x-www-form-urlencoded' }, body,
    });
    if (!response.ok) throw new Error(`Azure authentication failed (${response.status})`);
    const value = await response.json();
    cachedToken = { value: value.access_token, expiresAt: Date.now() + value.expires_in * 1000 };
    return cachedToken.value;
  }

  async function request(suffix, method = 'GET') {
    const response = await fetch(`${base}${suffix}?api-version=2024-03-01`, {
      method, headers: { authorization: `Bearer ${await token()}` },
    });
    if (!response.ok && response.status !== 202) throw new Error(`Azure ${method} failed (${response.status})`);
    return response.status === 202 ? { accepted: true } : response.json();
  }

  return {
    name: 'azure', configured: ready,
    async status() {
      if (!ready) return { configured: false, state: 'unknown', detail: 'Azure environment variables are incomplete' };
      const view = await request('/instanceView');
      const power = (view.statuses || []).find((entry) => String(entry.code).startsWith('PowerState/'));
      return { configured: true, state: power?.code?.replace('PowerState/', '') || 'unknown', detail: power?.displayStatus || '' };
    },
    async start() { if (!ready) throw new Error('Azure driver is not configured'); return request('/start', 'POST'); },
    async stop() { if (!ready) throw new Error('Azure driver is not configured'); return request('/deallocate', 'POST'); },
  };
}

function gceDriver(gce) {
  const ready = configured([gce.project, gce.zone, gce.instance]);
  const base = `https://compute.googleapis.com/compute/v1/projects/${gce.project}/zones/${gce.zone}/instances/${gce.instance}`;
  let cachedToken;

  async function token() {
    if (gce.accessToken) return gce.accessToken;
    if (cachedToken?.expiresAt > Date.now() + 60_000) return cachedToken.value;
    const response = await fetch('http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token', {
      headers: { 'Metadata-Flavor': 'Google' },
    });
    if (!response.ok) throw new Error(`GCE metadata authentication failed (${response.status})`);
    const value = await response.json();
    cachedToken = { value: value.access_token, expiresAt: Date.now() + value.expires_in * 1000 };
    return cachedToken.value;
  }

  async function request(suffix = '', method = 'GET') {
    const response = await fetch(`${base}${suffix}`, { method, headers: { authorization: `Bearer ${await token()}` } });
    if (!response.ok) throw new Error(`GCE ${method} failed (${response.status}): ${await response.text()}`);
    return response.json();
  }

  return {
    name: 'gce', configured: ready,
    async status() {
      if (!ready) return { configured: false, state: 'unknown', detail: 'GCE environment variables are incomplete' };
      const instance = await request();
      return { configured: true, state: String(instance.status || 'unknown').toLowerCase(), detail: instance.machineType?.split('/').pop() || '' };
    },
    async start() { if (!ready) throw new Error('GCE driver is not configured'); return request('/start', 'POST'); },
    async stop() { if (!ready) throw new Error('GCE driver is not configured'); return request('/stop', 'POST'); },
  };
}

const noop = {
  name: 'noop', configured: false,
  async status() { return { configured: false, state: 'unknown', detail: 'No power driver configured' }; },
  async start() { throw new Error('No power driver configured'); },
  async stop() { throw new Error('No power driver configured'); },
};

export function makePowerDriver(config) {
  if (config.powerDriver === 'azure') return azureDriver(config.azure);
  if (config.powerDriver === 'gce') return gceDriver(config.gce);
  return noop;
}
