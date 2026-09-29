"""Build extension packages from an explicit allowlist; no browser data included."""
import io
import json
import zipfile
from pathlib import Path

EXTENSION_FILES = ('core.js', 'workspace.js', 'workspace.html', 'style.css', 'launcher.html')


def extension_archive(base: Path, flavor: str) -> bytes:
    if flavor not in {'chromium', 'firefox'}:
        raise ValueError('Unsupported browser package')
    root = base / 'extensions/browser-maintenance'
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if flavor == 'firefox':
        manifest.pop('minimum_chrome_version', None)
        manifest['browser_specific_settings'] = {'gecko': {
            'id': 'omnitool-maintenance@local.invalid', 'strict_min_version': '145.0',
            'data_collection_permissions': {'required': ['none']}}}
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('manifest.json', json.dumps(manifest, indent=2))
        for filename in EXTENSION_FILES:
            archive.write(root / filename, filename)
    return output.getvalue()

