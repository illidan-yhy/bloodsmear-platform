"""Build allowlisted deployment archives; never include local runtime/user data."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build(kind: str, *, offline: bool = False, output: Path | None = None, revision: str | None = None) -> Path:
    destination = output or ROOT / 'releases'
    destination.mkdir(parents=True, exist_ok=True)
    name = 'BloodSmear-Windows-RTX3060-' + ('offline-client' if offline else 'online-client') if kind == 'windows' else 'BloodSmear-Linux-server-source'
    if revision:
        if not re.fullmatch(r'[a-zA-Z0-9-]+', revision):
            raise ValueError('Release revision must contain only letters, digits and hyphens')
        name += '-' + revision
    archive_path = destination / f'{name}.zip'
    files: dict[str, Path] = {}
    for folder in ('src', 'models', 'docs/deployment'):
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.pyo'):
                files[path.relative_to(ROOT).as_posix()] = path
    for filename in ('README.md', 'requirements.lock'):
        files[filename] = ROOT / filename
    files['samples/Blood.png'] = ROOT / 'samples/Blood.png'
    files['deployment_check.py'] = ROOT / 'deployment/deployment_check.py'
    for filename in ('benchmark_single.py',):
        files[f'scripts/{filename}'] = ROOT / 'scripts' / filename
    if kind == 'windows':
        for path in (ROOT / 'deployment/windows').iterdir():
            if path.is_file():
                files[path.name] = path
        excluded_manuals = (
            'docs/deployment/03-Linux服务器与离线部署手册.md',
            'docs/deployment/04-并发能力与容量说明.md',
        )
        for manual in excluded_manuals:
            files.pop(manual, None)
        customer_docs = ROOT / 'deployment/windows/customer'
        files['README.md'] = customer_docs / 'README.md'
        files['docs/deployment/README.md'] = customer_docs / 'docs-index.md'
        files['docs/deployment/05-交付包清单与验证记录.md'] = customer_docs / 'delivery-list.md'
        if offline:
            wheels = list((ROOT / 'releases/wheelhouse-win-py311').glob('*.whl'))
            if not wheels:
                raise ValueError('Offline package requested without wheels')
            for wheel in wheels:
                files[f'wheelhouse/{wheel.name}'] = wheel
    else:
        for filename in ('Dockerfile', 'docker-compose.yml'):
            files[filename] = ROOT / filename
        for path in (ROOT / 'deployment/linux').iterdir():
            if path.is_file():
                files[f'deployment/linux/{path.name}'] = path
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = 'unavailable'
    manifest = {
        'name': name, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit': revision, 'snapshot': 'current allowlisted working files',
        'python': '3.11 x64', 'target': kind,
        'includes_linux_image_tar': False,
        'includes_python_installer': False,
        'includes_nvidia_driver': False,
        'includes_python_wheels': offline,
        'files': {name: sha256(path) for name, path in sorted(files.items())},
    }
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for relative, path in sorted(files.items()):
            archive.write(path, f'{name}/{relative}')
        sections = []
        for relative, manual in sorted(files.items()):
            if relative.startswith('docs/deployment/') and relative.endswith('.md'):
                sections.append('<section><h2>' + html.escape(Path(relative).stem) + '</h2><pre>' + html.escape(manual.read_text(encoding='utf-8')) + '</pre></section>')
        manual_html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>BloodSmear部署手册</title><style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;max-width:1050px;margin:30px auto;padding:0 24px;color:#24343a}section{border-top:1px solid #ccc;padding-top:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-family:inherit;line-height:1.7}h1,h2{color:#0f766e}</style><h1>BloodSmear 部署材料</h1>' + ''.join(sections) + '</html>'
        archive.writestr(f'{name}/部署手册.html', manual_html)
        manifest['files']['部署手册.html'] = hashlib.sha256(manual_html.encode('utf-8')).hexdigest()
        archive.writestr(f'{name}/PACKAGE_MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    checksum_path = archive_path.with_suffix('.zip.sha256')
    checksum_path.write_text(f'{sha256(archive_path)}  {archive_path.name}\n', encoding='ascii')
    print(json.dumps({'archive': str(archive_path), 'bytes': archive_path.stat().st_size, 'sha256_file': str(checksum_path)}, ensure_ascii=False))
    return archive_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=('windows','linux'), required=True)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--revision')
    args = parser.parse_args()
    build(args.kind, offline=args.offline, revision=args.revision)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
