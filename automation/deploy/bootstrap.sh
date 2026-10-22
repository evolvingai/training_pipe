#!/usr/bin/env bash
set -euo pipefail

# Run as root on a fresh Ubuntu VM. Required environment:
#   TRAINER_REPO_URL, CONTROL_PLANE_URL, AGENT_TOKEN
# Optional: TRAINER_REPO_REF, AGENT_ID, VM_DEVICE, INSTALL_CONTROL,
#   INSTALL_VLM, ADMIN_USER, ADMIN_PASSWORD, SESSION_SECRET, TORCH_INDEX_URL.
TRAINER_REPO_URL="${TRAINER_REPO_URL:-}"
TRAINER_REPO_REF="${TRAINER_REPO_REF:-main}"
: "${TRAINER_REPO_URL:?set TRAINER_REPO_URL}"
: "${CONTROL_PLANE_URL:?set CONTROL_PLANE_URL}"
: "${AGENT_TOKEN:?set AGENT_TOKEN}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "bootstrap.sh must run as root" >&2
  exit 2
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ca-certificates curl git python3 python3-venv build-essential libgl1 libglib2.0-0
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs

id trainer >/dev/null 2>&1 || useradd --system --create-home --shell /bin/bash trainer
install -d -o trainer -g trainer /opt/trainer /data /runs /var/lib/trainer-automation /etc/trainer-automation

if [[ ! -d /opt/trainer/.git ]]; then
  rmdir /opt/trainer
  git clone --branch "${TRAINER_REPO_REF}" "${TRAINER_REPO_URL}" /opt/trainer
fi
chown -R trainer:trainer /opt/trainer /data /runs /var/lib/trainer-automation

python3 -m venv /opt/trainer-venv
if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  /opt/trainer-venv/bin/pip install torch --index-url "${TORCH_INDEX_URL}"
else
  /opt/trainer-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
if [[ "${INSTALL_VLM:-false}" == "true" ]]; then
  /opt/trainer-venv/bin/pip install -e '/opt/trainer[vlm]'
else
  /opt/trainer-venv/bin/pip install -e /opt/trainer
fi

install -m 0644 /opt/trainer/automation/deploy/trainer-agent.service /etc/systemd/system/trainer-agent.service
cat >/etc/trainer-automation/agent.env <<EOF
CONTROL_PLANE_URL=${CONTROL_PLANE_URL}
AGENT_TOKEN=${AGENT_TOKEN}
AGENT_ID=${AGENT_ID:-$(hostname)}
VM_RUNS_DIR=/runs
VM_DEVICE=${VM_DEVICE:-cpu}
EOF
chmod 0600 /etc/trainer-automation/agent.env

if [[ "${INSTALL_CONTROL:-false}" == "true" ]]; then
  : "${ADMIN_PASSWORD:?set ADMIN_PASSWORD when INSTALL_CONTROL=true}"
  : "${SESSION_SECRET:?set SESSION_SECRET when INSTALL_CONTROL=true}"
  install -m 0644 /opt/trainer/automation/deploy/trainer-control.service /etc/systemd/system/trainer-control.service
  cat >/etc/trainer-automation/control.env <<EOF
PORT=8080
AUTOMATION_DATA_DIR=/var/lib/trainer-automation
ADMIN_USER=${ADMIN_USER:-admin}
ADMIN_PASSWORD=${ADMIN_PASSWORD}
SESSION_SECRET=${SESSION_SECRET}
AGENT_TOKEN=${AGENT_TOKEN}
VM_CODE_DIR=/opt/trainer
VM_PYTHON=/opt/trainer-venv/bin/python
VM_DATA_DIR=/data
VM_RUNS_DIR=/runs
VM_DEVICE=${VM_DEVICE:-cpu}
POWER_DRIVER=noop
AUTO_SHUTDOWN=false
EOF
  chmod 0600 /etc/trainer-automation/control.env
fi

systemctl daemon-reload
systemctl enable --now trainer-agent.service
if [[ "${INSTALL_CONTROL:-false}" == "true" ]]; then
  systemctl enable --now trainer-control.service
fi
