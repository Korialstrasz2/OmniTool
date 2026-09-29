"""HTTP integration gate: install requirements-dev.txt before running."""
import re
from pathlib import Path
import pytest
pytest.importorskip('flask')
from omnitool_core.web import create_app
from omnitool_core.maintenance_web import register

TOKEN='test-local-access-code-not-a-real-secret'

@pytest.fixture
def app(tmp_path):
    application=create_app(Path(__file__).resolve().parents[1],access_token=TOKEN)
    register(application,state=tmp_path/'state')
    application.config['TESTING']=True
    yield application
    application.extensions['omnitool_shutdown']()

def login(client):
    response=client.get('/login')
    csrf=re.search(r'name="csrf_token" value="([^"]+)"',response.text).group(1)
    assert client.post('/login',data={'csrf_token':csrf,'access_token':TOKEN}).status_code==302
    with client.session_transaction() as session:return {'X-CSRF-Token':session['csrf']}

def test_access_csrf_and_setup(app):
    client=app.test_client()
    assert client.get('/api/maintenance/rename/operations').status_code==401
    headers=login(client)
    assert client.post('/api/maintenance/rename/preview',json={}).status_code==403
    assert client.post('/api/maintenance/rename/preview',json={},headers={**headers,'Origin':'https://untrusted.example'}).status_code==403
    for path in ['/maintenance/browser/history','/maintenance/browser/cookies','/maintenance/lowercase']:
        assert client.get(path).status_code==200
    package=client.get('/maintenance/browser/package/chromium')
    assert package.status_code==200 and package.data[:2]==b'PK'
    assert package.headers['Cache-Control']=='no-store'

def test_preview_apply_undo_and_single_use(app,tmp_path):
    root=tmp_path/'files';root.mkdir();(root/'PHOTO.JPG').write_bytes(b'test')
    client=app.test_client();headers=login(client)
    preview=client.post('/api/maintenance/rename/preview',json={'root':str(root)},headers=headers)
    assert preview.status_code==200 and preview.json['count']==1
    token=preview.json['token']
    assert client.post('/api/maintenance/rename/apply',json={'token':token,'confirmation':'wrong'},headers=headers).status_code==400
    applied=client.post('/api/maintenance/rename/apply',json={'token':token,'confirmation':'RENAME 1'},headers=headers)
    assert applied.status_code==200 and (root/'photo.jpg').read_bytes()==b'test'
    assert client.post('/api/maintenance/rename/apply',json={'token':token,'confirmation':'RENAME 1'},headers=headers).status_code==400
    undo=client.post('/api/maintenance/rename/recover',json={'operation':applied.json['id'],'confirmation':'UNDO'},headers=headers)
    assert undo.status_code==200 and 'PHOTO.JPG' in {p.name for p in root.iterdir()}

def test_preview_session_scope_and_stale_files(app,tmp_path):
    root=tmp_path/'files';root.mkdir();(root/'A.TXT').write_text('one')
    alice,bob=app.test_client(),app.test_client();ha,hb=login(alice),login(bob)
    token=alice.post('/api/maintenance/rename/preview',json={'root':str(root)},headers=ha).json['token']
    assert bob.get('/api/maintenance/rename/preview/'+token).status_code==400
    assert bob.post('/api/maintenance/rename/apply',json={'token':token,'confirmation':'RENAME 1'},headers=hb).status_code==400
    (root/'A.TXT').write_text('changed')
    assert alice.post('/api/maintenance/rename/apply',json={'token':token,'confirmation':'RENAME 1'},headers=ha).status_code==400
    assert 'A.TXT' in {p.name for p in root.iterdir()}
