#!/usr/bin/env bash
set -u
printf 'Time UTC: '; date -u
uname -a
if [ -f /etc/os-release ]; then cat /etc/os-release; fi
command -v lscpu >/dev/null && lscpu
command -v free >/dev/null && free -h
df -h .
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
command -v docker >/dev/null && docker version
command -v docker >/dev/null && docker compose version
command -v nvidia-container-cli >/dev/null && nvidia-container-cli --version
