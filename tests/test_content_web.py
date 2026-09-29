"""HTTP integration: executed in CI with real Flask and disposable audio, not remote services."""
import re
from pathlib import Path
import pytest
pytest.importorskip('flask')
from omnitool_core.web import create_app
from omnitool_core.content_web import register
from test_content_tools import music

ROOT = Path(__file__).resolve().parents[1]
ACCESS = 'test-local-access-not-a-production-secret'

@pytest.fixture
def app():
    application = create_app(ROOT, access_token=ACCESS)
    register(application); application.config['TESTING'] = True
    yield application
    application.extensions['omnitool_shutdown']()

def login(client):
    page = client.get('/login')
    token = re.search(r'name="csrf_token" value="([^"]+)"', page.text)[1]
    assert client.post('/login', data={'csrf_token': token, 'access_token': ACCESS}).status_code == 302
    with client.session_transaction() as session: return {'X-CSRF-Token': session['csrf']}

def finish(app, client, token):
    with client.session_transaction() as session: owner = session['sid']
    task = app.extensions['omnitool_content_tasks'].get(owner, token)
    task['future'].result(timeout=20)
    data = client.get('/api/content/tasks/' + token).json
    assert data['state'] == 'succeeded', data
    return data['result']

def test_auth_csrf_and_templates(app):
    client = app.test_client()
    assert client.get('/api/content/tasks/' + 'a'*32).status_code == 401
    headers = login(client)
    for path in ('/content/prompts', '/content/lyrics', '/csv-editor'):
        response = client.get(path)
        assert response.status_code == 200 and response.headers['Cache-Control'] == 'no-store'
    assert client.post('/api/content/prompts/status', json={}).status_code == 403
    assert client.post('/api/content/prompts/status', json={}, headers={**headers, 'Origin': 'https://evil.invalid'}).status_code == 403
    assert client.post('/api/content/prompts/generate', json={'url': 'https://evil.invalid'}, headers=headers).status_code == 400

def test_reviewed_lyrics_pipeline_and_single_use(app, music, tmp_path):
    client = app.test_client(); headers = login(client)
    scan = client.post('/api/content/lyrics/scan', json={'root': str(music)}, headers=headers).json['task']
    finish(app, client, scan)
    request = {'scan_task': scan, 'out': str(tmp_path/'out'), 'provider': 'sidecar', 'selections': [{'index': 0}]}
    preview = client.post('/api/content/lyrics/prepare', json=request, headers=headers).json['task']
    assert finish(app, client, preview)['count'] == 1
    assert not (tmp_path/'out').exists()
    assert client.post('/api/content/lyrics/apply', json={'preview_task': preview, 'confirmation': 'WRONG'}, headers=headers).status_code == 400
    payload = {'preview_task': preview, 'confirmation': 'WRITE 1', 'out': str(tmp_path/'not-reviewed')}
    response = client.post('/api/content/lyrics/apply', json=payload, headers=headers)
    assert response.status_code == 202
    finish(app, client, response.json['task'])
    assert (tmp_path/'out/Track.mp3').exists() and not (tmp_path/'not-reviewed').exists()
    assert client.post('/api/content/lyrics/apply', json=payload, headers=headers).status_code == 400

def test_provider_consent_and_cross_session_access(app, music, tmp_path):
    alice, bob = app.test_client(), app.test_client()
    a, b = login(alice), login(bob)
    token = alice.post('/api/content/lyrics/scan', json={'root': str(music)}, headers=a).json['task']
    finish(app, alice, token)
    assert bob.get('/api/content/tasks/' + token).status_code == 400
    assert bob.post('/api/content/tasks/' + token + '/stop', json={}, headers=b).status_code == 400
    payload = {'scan_task': token, 'out': str(tmp_path/'out'), 'provider': 'lrclib', 'selections': [{'index': 0}]}
    response = alice.post('/api/content/lyrics/prepare', json=payload, headers=a)
    assert response.status_code == 400 and 'Explicitly allow' in response.json['error']

def test_signout_clears_content_results(app, music):
    client = app.test_client(); headers = login(client)
    token = client.post('/api/content/lyrics/scan', json={'root': str(music)}, headers=headers).json['task']
    finish(app, client, token)
    assert client.post('/logout', json={}, headers=headers).status_code == 302
    assert token not in app.extensions['omnitool_content_tasks'].items
