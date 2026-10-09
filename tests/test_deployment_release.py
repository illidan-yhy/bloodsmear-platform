"""Checks customer-facing package selection and archive/hash consistency."""
import hashlib
import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile


def test_windows_customer_release_excludes_internal_manuals_and_information(tmp_path):
    spec = importlib.util.spec_from_file_location('build_deployment', Path(__file__).parents[1] / 'scripts/build_deployment.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package = module.build('windows', output=tmp_path)
    with ZipFile(package) as archive:
        names = archive.namelist()
        assert not any('03-Linux' in name or '04-并发' in name for name in names)
        manifest_name = next(name for name in names if name.endswith('/PACKAGE_MANIFEST.json'))
        manifest = json.loads(archive.read(manifest_name))
        prefix = manifest['name'] + '/'
        for relative, expected in manifest['files'].items():
            assert hashlib.sha256(archive.read(prefix + relative)).hexdigest() == expected
        readme = archive.read(prefix + 'README.md').decode('utf-8')
        assert 'C:\\Program_file' not in readme
        assert 'GTX 1650' not in readme
        assert 'superpowers' not in readme
        manual = archive.read(prefix + '部署手册.html').decode('utf-8')
        assert '03-Linux' not in manual
        assert '04-并发' not in manual
        assert 'GTX 1650' not in manual
        assert '重试未获授权' not in manual
