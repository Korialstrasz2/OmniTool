import io
import json
import zipfile
from pathlib import Path
import pytest
from omnitool_core.browser_package import extension_archive
from omnitool_core.catalog import validate, Catalog

ROOT=Path(__file__).resolve().parents[1]
@pytest.mark.parametrize('flavor',['chromium','firefox'])
def test_package_only_includes_public_extension_sources(flavor):
    with zipfile.ZipFile(io.BytesIO(extension_archive(ROOT,flavor))) as archive:
        assert set(archive.namelist())=={'manifest.json','core.js','workspace.js','workspace.html','style.css','launcher.html'}
        manifest=json.loads(archive.read('manifest.json'))
        assert manifest['permissions']==['storage']
        assert 'bookmarks' not in manifest['optional_permissions']
        assert not manifest.get('content_scripts') and not manifest.get('externally_connectable') and not manifest.get('background')
        assert "connect-src 'none'" in manifest['content_security_policy']['extension_pages']
        if flavor=='firefox':
            assert manifest['browser_specific_settings']['gecko']['strict_min_version']=='145.0'
        else: assert manifest['minimum_chrome_version']=='130'

@pytest.mark.parametrize('page',['/maintenance/browser/history','/maintenance/browser/cookies','/maintenance/lowercase'])
def test_maintenance_pages_are_registered_in_catalog(page):
    spec=validate({'id':'test','name':'Test','kind':'page','page':page})
    assert spec['page']==page

def test_browser_tools_show_setup_not_false_ready():
    assert Catalog.availability({'setup':'browser-extension'})=='needs-setup'

def test_invalid_package_flavor():
    with pytest.raises(ValueError): extension_archive(ROOT,'../../private')
