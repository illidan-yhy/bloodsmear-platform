#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
curl --fail --max-time 15 http://127.0.0.1:8000/health/ready
curl --fail --max-time 120 -F image=@samples/Blood.png http://127.0.0.1:8000/api/v1/infer
