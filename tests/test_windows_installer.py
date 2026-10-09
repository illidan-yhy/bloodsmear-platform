"""Run the real installer preflight through Windows PowerShell's legacy argv path."""
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest


@pytest.mark.skipif(os.name != 'nt' or not shutil.which('powershell.exe'), reason='Windows PowerShell is required')
def test_installer_python_preflight_handles_spaces_with_windows_powershell(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('build_deployment', root / 'scripts/build_deployment.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    archive_path = builder.build('windows', output=tmp_path / 'archives')
    extraction = tmp_path / 'package with spaces'
    with ZipFile(archive_path) as archive:
        archive.extractall(extraction)
    install_script = next(extraction.glob('*/install.ps1'))
    command = [
        'powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
        str(install_script), '-PythonExe', sys.executable, '-CheckPythonOnly',
    ]
    environment = os.environ.copy()
    environment['PSModulePath'] = str(Path(environment.get('SystemRoot', r'C:\Windows')) / 'System32/WindowsPowerShell/v1.0/Modules')
    completed = subprocess.run(command, capture_output=True, timeout=60, env=environment)
    assert completed.returncode == 0, (completed.stdout + completed.stderr).decode('utf-8', errors='replace')
    assert b'Python 3.11 x64 preflight passed' in completed.stdout
    assert not (install_script.parent / '.venv').exists()
