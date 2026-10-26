# Trainer Automation

Trainer applies the queue-and-agent flow from `gender_automation` to four
first-class workflows: detection, image classification, video/action
classification, and VLM LoRA tuning. The shared UI, queue, target-machine
selection, logs, retry flow, and artifact inventory apply to every workflow.
Every queued command is dispatched through `python -m trainer` with explicit
`task=` and `arch=` values; the web wrapper and direct CLI therefore exercise
the same plugin registry.

## What is included

- Password-protected web control plane with a persistent file-backed job queue.
- Outbound-only VM agent: heartbeat, claim, step status, live log tail,
  cancellation, artifact inventory, and target-machine pinning.
- Recipes for detector train, finetune, evaluate/report, predict, export, and
  hyperparameter tuning.
- Detector-parity image-classifier recipes: train/resume, finetune, evaluate,
  per-class threshold tuning, HTML report, top-k prediction, hyperparameter
  tuning, augmentation preview, data benchmark, and ONNX/TensorRT export.
- Recipes for video/action train, evaluate, predict, and ONNX/TensorRT export. The
  baseline uses a frame encoder plus GRU and can run on CPU.
- VLM multimodal JSONL validation, PEFT LoRA tuning and evaluation, adapter artifacts,
  adapter-to-base-model merge, and fixed-shape ONNX/TensorRT forward export.
- CPU, CUDA, Apple MPS, and auto device selection. A CPU-only agent will not
  claim a job that explicitly requests CUDA.
- Azure ARM and Google Compute Engine REST power drivers. Azure `stop`
  deallocates the VM; GCE `stop` stops it.
- A dependency-free Node 20 control plane/agent and systemd deployment units.
- A combined History ledger: persisted control-plane jobs plus completed runs
  discovered under every agent's `VM_RUNS_DIR`. Direct CLI runs and the four
  per-family entries in `vm_smoke_matrix/summary.json` therefore remain visible
  even when they were not launched from the web UI.
- One external `trainer.augmentation` subsystem and UI panel shared by detector,
  image-classifier, video/action, and VLM training. Profiles can be stored as TOML;
  video replays identical geometry across a clip and VLM disables multi-image transforms.
- A live model registry generated from installed Python architecture plugins,
  including input/output contracts, training strategy, aliases, and an interactive
  per-family model-flow diagram. External `trainer.architectures` entry points
  appear without a web-code change.
- A sample-level xAI view that exposes inputs, predictions, confidence,
  explanations, and activations as one result. It does not expose manifests,
  metrics files, filesystem paths, or generic run artifacts.
- ConvNeXt classification xAI that writes signed Grad-CAM overlays (red supports
  the selected class; blue opposes it), per-stage top activation channels,
  combined dossiers, a searchable HTML gallery, and an auditable JSON manifest.
  The xAI page also has a selected-sample workflow: upload one test image, choose
  its checkpoint and CPU/GPU target, and keep the prediction, confidence, signed
  map, and feature activations together in the page.
- A family-aware Evaluations workspace for detection, image classification,
  video/action classification, and VLM checkpoints. Leaderboards compare only
  jobs with the exact same dataset and split, then show the metrics appropriate
  to that family, per-class diagnostics, and confusion matrices where applicable.
- A family-aware Dataset Explorer for COCO and custom datasets. COCO annotation
  JSON/root paths and custom YOLO TOML, image folders, video folders, or VLM
  JSONL are profiled on the selected VM and shown through the same sample/split,
  class-distribution, missing-label, invalid-record, and quality visualizations.
  Raw dataset files are not copied into or exposed by the control plane.
- New Run forms expose runnable datasets through an **Available dataset or
  new** selector. Selecting a registered/profiled dataset fills and locks its
  worker-local path; selecting **New dataset…** enables a new TOML, root, or
  JSONL path appropriate to that model family.

The control plane creates fixed argument arrays. The agent uses `spawn()` with
`shell: false`; job parameters are never evaluated by a shell.

## Model registry and xAI

`trainer list` is the source of truth for the Models page. Architecture plugins
publish their family, input and output contracts, training strategy, and ordered
flow stages through `ArchitecturePlugin`. The control plane caches that catalog
for 30 seconds and exposes it through `GET /api/architectures`.

The browser reads structured sample results through `GET /api/xai/classification`.
Three opaque, authenticated image endpoints render the input, signed Grad-CAM,
and stage activations. No generic evidence-list or arbitrary-file endpoint is
available. For remote CPU/GPU runs, the agent syncs only the minimum internal
record and those three images needed to render the same sample result.

Selected test images are stored behind authentication on the control plane. The
chosen agent downloads the opaque `sample://…` reference over its existing
authenticated outbound connection before inference, so operators do not need to
copy an ad-hoc image to the VM manually. JPEG, PNG, and WebP inputs up to 12 MiB
are accepted.

## Architecture

```text
Browser ──HTTP(S)──> control plane ──persistent queue
                          │
                          ├── Azure ARM or GCE start/stop (optional)
                          │
Training VM agent ─outbound HTTP(S)──┘
       └── python -m trainer (task and architecture plugin registry)
```

For the first CPU trial, run the control plane and agent on one VM and leave
`POWER_DRIVER=noop` and `AUTO_SHUTDOWN=false`. For production, keep the control
plane on an always-on small VM/container and put only the agent on the training
VM; then enable a power driver and auto-shutdown.

## Local CPU smoke run

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .

ADMIN_PASSWORD=local-password \
SESSION_SECRET=replace-with-random-value \
AGENT_TOKEN=replace-with-another-random-value \
VM_CODE_DIR="$PWD" VM_PYTHON="$PWD/.venv/bin/python" \
VM_DATA_DIR="$PWD/data" VM_RUNS_DIR="$PWD/runs" VM_DEVICE=cpu \
node automation/server/index.js
```

In a second terminal:

```bash
CONTROL_PLANE_URL=http://127.0.0.1:8080 \
AGENT_TOKEN=replace-with-another-random-value \
AGENT_ID=local-cpu VM_RUNS_DIR="$PWD/runs" \
node automation/vm/agent.js
```

Open `http://127.0.0.1:8080`, sign in as `admin`, and queue a job with device
`CPU`. The CPU form defaults (`trainer-n`, batch 2, image size 320) are intended
as a functional first run, not a performance target.

The control plane can also run as a container while the agent stays on the
host (the agent must launch Python against the host filesystem):

```bash
docker build -f automation/Dockerfile -t trainer-automation .
docker run --rm -p 8080:8080 -v trainer-automation-data:/data \
  -e ADMIN_PASSWORD -e SESSION_SECRET -e AGENT_TOKEN \
  -e VM_CODE_DIR=/opt/trainer -e VM_PYTHON=/opt/trainer-venv/bin/python \
  -e VM_RUNS_DIR=/runs trainer-automation
```

## Classifier datasets

The bundled classifier supports both target contracts. For multi-label scene
classification, point `data` at a COCO instances JSON or at a COCO root:

```text
coco/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/*.jpg
└── val2017/*.jpg
```

Every unique category present in an image becomes one positive target. Multiple
instances of the same category still count once. Training uses sigmoid with
binary cross entropy; evaluation reports mAP, micro/macro F1, exact-match
accuracy, per-class AP, and tunable per-class thresholds.

For single-label classification, explicit class-folder splits remain supported:

```text
dataset/
├── train/
│   ├── healthy/*.jpg
│   └── affected/*.jpg
└── val/
    ├── healthy/*.jpg
    └── affected/*.jpg
```

or a compact layout, which is deterministically split per class:

```text
dataset/
├── healthy/*.jpg
└── affected/*.jpg
```

A classifier run writes EMA-backed checkpoints under `weights/`, tracker and
history CSVs under `logs/`, curves/confusion matrices under `plots/`, and an
HTML report under `reports/`. Checkpoints include raw and EMA weights,
optimizer/scheduler/scaler state, class names, input size, architecture, metrics,
and the single-label or multi-label activation contract so a run can resume
exactly. Evaluation also writes per-image metrics
and optimized class thresholds; prediction supports top-k output and threshold
rejection. Export writes a dynamic-batch ONNX graph and JSON preprocessing,
class, and tensor contract. TensorRT export retains the ONNX intermediate and
writes an engine plus the same JSON contract.

## Video/action datasets

Video classification accepts the same split or compact class-folder structure,
using video files instead of images:

```text
dataset/
├── train/
│   ├── walking/*.mp4
│   └── running/*.mp4
└── val/
    ├── walking/*.mp4
    └── running/*.mp4
```

Each checkpoint records the class names, frame size, number of sampled frames,
and temporal model shape. Evaluate writes accuracy and a confusion matrix;
predict writes CSV; export writes an ONNX graph with `clips` shaped
`batch × time × 3 × height × width` plus a JSON contract. The same Export page
can compile that graph into a TensorRT engine on a CUDA worker.

## VLM tuning datasets

Install optional dependencies on VLM workers with:

```bash
pip install -e ".[vlm]"
```

The simplest JSONL format is one local image, prompt, and answer per row:

```json
{"image":"images/0001.jpg","prompt":"What is the worker doing?","answer":"The worker is lifting a box."}
```

Trainer also accepts multimodal `messages` rows. Images must be staged on the
worker; remote image URLs are rejected so runs remain reproducible. A tune run
writes `adapter/`, `run.json`, and `metrics.json`. The Export page merges that
adapter with its base model when a standalone model directory is needed. VLM
ONNX/TensorRT export traces the selected sample image and prompt as a fixed-shape
forward-logits contract; autoregressive decoding remains a serving-runtime concern.
The Evaluate action scores a base model or LoRA adapter against an explicit JSONL
set and reports loss, perplexity, answer-token accuracy, sample count, and evaluated
token count in the Evaluations workspace.

## Dataset progression

The Data page keeps a family-specific shortlist next to the profiler. Use COCO
2017 val/train for detection and multi-label image classification, UCF101 for
the first video/action smoke test before scaling to Kinetics-400, and COCO
Captions for the first VLM ingestion/tuning smoke before conversational tuning
with LLaVA-Instruct-150K. These are recommendations, not automatic downloads;
the worker profiles the exact local path chosen for a run.

## VM deployment

### Common bootstrap

The supported target is Ubuntu. Clone the repository and run the bootstrap as
root with secrets supplied through a root-only environment or secret manager:

```bash
sudo env \
  TRAINER_REPO_URL=git@YOUR_HOST:YOUR_ORG/trainer.git \
  TRAINER_REPO_REF=main \
  CONTROL_PLANE_URL=http://127.0.0.1:8080 \
  AGENT_TOKEN='replace-me' \
  AGENT_ID=cpu-01 \
  VM_DEVICE=cpu \
  INSTALL_VLM=true \
  INSTALL_CONTROL=true \
  ADMIN_PASSWORD='replace-me' \
  SESSION_SECRET='replace-with-32-random-bytes' \
  bash automation/deploy/bootstrap.sh
```

This installs Node 20, creates a `trainer` service account, creates a Python
virtual environment, installs CPU PyTorch and Trainer, and starts both systemd
services. Data and outputs live at `/data` and `/runs`. For a later GPU image,
set `TORCH_INDEX_URL` to the PyTorch wheel index that matches the VM's installed
CUDA driver and set `VM_DEVICE=cuda:0`.

Do not keep production secrets in `cloud-init.yaml`. The template is for a
throwaway trial; use Azure Key Vault, Google Secret Manager, or root-owned
`/etc/trainer-automation/*.env` files (mode `0600`) for a persistent machine.

Useful checks:

```bash
systemctl status trainer-control trainer-agent
journalctl -u trainer-agent -f
curl http://127.0.0.1:8080/api/me
```

### Azure CPU VM

Create a general-purpose CPU VM; an L4 is not required for the first pass:

```bash
az group create --name trainer-rg --location westeurope
az vm create \
  --resource-group trainer-rg \
  --name trainer-cpu-01 \
  --image Ubuntu2204 \
  --size Standard_D8s_v5 \
  --admin-username azureuser \
  --generate-ssh-keys
```

SSH to the VM, clone this repository, and run the common bootstrap. For a
first private trial, keep port 8080 closed and use an SSH tunnel:

```bash
ssh -L 8080:127.0.0.1:8080 azureuser@VM_PUBLIC_IP
```

Then open `http://127.0.0.1:8080` locally. For a production UI, terminate TLS
in an Azure Application Gateway/reverse proxy and do not expose port 8080
directly.

For a split control/training deployment, configure the control plane with:

```text
POWER_DRIVER=azure
AZURE_TENANT_ID=...
AZURE_CLIENT_ID=...
AZURE_CLIENT_SECRET=...
AZURE_SUBSCRIPTION_ID=...
AZURE_RESOURCE_GROUP=trainer-rg
AZURE_VM_NAME=trainer-cpu-01
AUTO_SHUTDOWN=true
```

The service principal needs VM start/read/deallocate permissions scoped to the
training VM.

### Google Compute Engine CPU VM

```bash
gcloud compute instances create trainer-cpu-01 \
  --project YOUR_PROJECT \
  --zone europe-west4-a \
  --machine-type n2-standard-8 \
  --image-family ubuntu-2204-lts \
  --image-project ubuntu-os-cloud \
  --boot-disk-size 100GB
```

SSH, clone, and run the common bootstrap. A startup script can instead be
passed with `--metadata-from-file=startup-script=automation/deploy/bootstrap.sh`
after arranging its required environment through instance metadata/secrets.
Keep the UI private for the first trial and tunnel it locally:

```bash
gcloud compute ssh trainer-cpu-01 --zone europe-west4-a -- -L 8080:127.0.0.1:8080
```

For a split deployment, give the control plane service account
`roles/compute.instanceAdmin.v1` on the training project and configure:

```text
POWER_DRIVER=gce
GCE_PROJECT=YOUR_PROJECT
GCE_ZONE=europe-west4-a
GCE_INSTANCE=trainer-cpu-01
AUTO_SHUTDOWN=true
```

When hosted on Google infrastructure, the driver obtains an access token from
the metadata service. `GCE_ACCESS_TOKEN` exists only for short-lived local
testing and should not be stored as configuration.

Azure CLI can pass cloud-init through `az vm create --custom-data`; Compute
Engine supports `--metadata-from-file=startup-script=...`. See the current
[Azure custom-data documentation](https://learn.microsoft.com/azure/virtual-machines/custom-data),
[Compute Engine startup-script documentation](https://cloud.google.com/compute/docs/instances/startup-scripts/linux),
and [Compute Engine start/stop API](https://cloud.google.com/compute/docs/instances/stop-start-instance).

## Configuration

Control plane:

| Variable | Default | Meaning |
| --- | --- | --- |
| `HOST` | `0.0.0.0` | HTTP bind address; use `127.0.0.1` for SSH-tunnel-only access |
| `PORT` | `8080` | HTTP listen port |
| `AUTOMATION_DATA_DIR` | `.automation-data` | Persistent queue/state directory |
| `ADMIN_USER` / `ADMIN_PASSWORD` | `admin` / `change-me` | UI login |
| `SESSION_SECRET` | development value | HMAC session secret |
| `AGENT_TOKEN` | development value | Agent bearer token |
| `MANAGED_TARGETS_JSON` | empty | JSON array of CPU/GPU targets kept visible in the UI while offline |
| `VM_CODE_DIR` | `/opt/trainer` | Working directory sent in commands |
| `VM_PYTHON` | `python3` | Python executable sent in commands |
| `VM_DATA_DIR` | `/data` | UI dataset-path default |
| `VM_RUNS_DIR` | `/runs` | Artifact root |
| `VM_DEVICE` | `cpu` | Default device |
| `POWER_DRIVER` | `noop` | `noop`, `azure`, or `gce` |
| `AUTO_SHUTDOWN` | `false` | Stop/deallocate when queue empties |

Agent:

| Variable | Default | Meaning |
| --- | --- | --- |
| `CONTROL_PLANE_URL` | `http://127.0.0.1:8080` | Control plane base URL |
| `AGENT_TOKEN` | development value | Must match the control plane |
| `AGENT_ID` | hostname | Queue target identity |
| `AGENT_TYPES` | all | Optional comma-separated recipe allow-list |
| `AGENT_LABEL` | agent id | Human-readable compute target name shown in the UI |
| `AGENT_PROVIDER` | empty | Provider label such as `azure` or `gcp` |
| `AGENT_MACHINE_TYPE` | empty | Cloud SKU/machine type shown in the UI |
| `AGENT_DISABLE_GPU` | `false` | Hide GPUs and accept CPU work only (for an intentional CPU execution profile) |
| `VM_RUNS_DIR` | `/runs` | Inventory scan root |
| `AGENT_POLL_MS` | `5000` | Idle claim interval |

The file store is intentionally single-control-plane. Put
`AUTOMATION_DATA_DIR` on a persistent disk and run one control instance. A
multi-instance hosted control plane should replace `FileStore` with a
transactional database adapter before scaling horizontally.

## Tests

```bash
cd automation && npm test
cd .. && pytest tests/test_classifier_automation.py tests/test_video_classifier_automation.py tests/test_vlm_tune_automation.py -q
```
