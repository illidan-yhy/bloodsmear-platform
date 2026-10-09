#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p outputs data logs
export BLOODSMEAR_UID="$(id -u)"
export BLOODSMEAR_GID="$(id -g)"
docker compose up -d --no-build
docker compose ps
