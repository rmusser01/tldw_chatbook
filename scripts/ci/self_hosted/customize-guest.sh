#!/bin/bash
# Run as root inside a trusted image-bake VM, before its final cloud-init clean.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends gh python3.12-venv time
# setup-python's shared-library builds must survive sanitized child environments.
install -d -o runner -g runner /opt/hostedtoolcache
runner_env=/opt/actions-runner/.env
touch "$runner_env"
sed -i '/^AGENT_TOOLSDIRECTORY=/d' "$runner_env"
printf 'AGENT_TOOLSDIRECTORY=/opt/hostedtoolcache\n' >> "$runner_env"
chown runner:runner "$runner_env"
