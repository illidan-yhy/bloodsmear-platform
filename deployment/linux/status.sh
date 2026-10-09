#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
docker compose ps
curl --fail --max-time 15 http://127.0.0.1:8000/health/ready
