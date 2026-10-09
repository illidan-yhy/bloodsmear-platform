"""Deployment smoke check: validate the actual session, not just installed providers."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from datetime import datetime, timezone

from bloodsmear.config import AppSettings
from bloodsmear.inference import InferenceService
from bloodsmear.reporting.service import ReportService


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-gpu', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('outputs/environment-check.json'))
    args = parser.parse_args()
    evidence = {'time_utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(), 'os': platform.platform(), 'passed': False}
    try:
        import onnxruntime as ort
        evidence['onnxruntime'] = ort.__version__
        evidence['available_providers'] = ort.get_available_providers()
        service = InferenceService.from_settings(AppSettings(require_gpu=args.require_gpu))
        evidence['actual_provider'] = service.adapter.provider
        result = service.infer_bytes(Path('samples/Blood.png').read_bytes(), 'Blood.png', 'deployment-smoke')
        evidence['model_sha256'] = result.model.sha256
        evidence['counts'] = result.summary.cell_counts
        evidence['pipeline_ms'] = result.runtime.total_ms
        reports = ReportService().generate_single(result, service.package.manifest.license_status, Path('outputs/deployment-smoke'))
        evidence['reports'] = {'excel': str(reports.excel), 'pdf': str(reports.pdf)}
        if args.require_gpu and result.runtime.provider != 'CUDAExecutionProvider':
            raise RuntimeError('CUDAExecutionProvider was not selected')
        evidence['passed'] = True
    except Exception as exc:
        evidence['error'] = f'{type(exc).__name__}: {exc}'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0 if evidence['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
