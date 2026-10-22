import assert from 'node:assert/strict';
import test from 'node:test';

import { normalizeArchitectureCatalog } from '../server/architecture-registry.js';

test('architecture catalog preserves plugin contracts and flow stages', () => {
  const value = normalizeArchitectureCatalog({
    tasks: [{ name: 'classification', actions: ['train'], default_architecture: 'convnext-tiny' }],
    architectures: [{
      task: 'classification', name: 'convnext-tiny', aliases: ['convnext'], family: 'ConvNeXt',
      input_contract: 'images', output_contract: 'logits', training_strategy: 'supervised',
      stages: ['image', 'backbone', 'logits'],
    }],
  });
  assert.equal(value.tasks[0].label, 'Image Classification');
  assert.equal(value.architectures[0].family, 'ConvNeXt');
  assert.deepEqual(value.architectures[0].stages, ['image', 'backbone', 'logits']);
});

test('external plugins receive safe display fallbacks', () => {
  const value = normalizeArchitectureCatalog({ architectures: [{ task: 'classification', name: 'custom' }] });
  assert.equal(value.architectures[0].family, 'External plugin');
  assert.deepEqual(value.architectures[0].stages, ['input', 'architecture plugin', 'outputs']);
});

