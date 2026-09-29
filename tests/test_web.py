"""Integration checks require requirements-dev.txt; skip when Flask is absent."""
import re
from pathlib import Path
import pytest
pytest.importorskip("flask")
from omnitool_core.web import create_app

@pytest.fixture
def app():
    app = create_app(Path(__file__).parents[1], access_token="test-access-code-that-is-not-a-secret")
    app.config["TESTING"] = True
    yield app
    app.extensions["omnitool_shutdown"]()

def sign_in(client):
    response = client.get('/login')
    csrf = re.search(r'name="csrf_token" value="([^"]+)"', response.text).group(1)
    response = client.post('/login', data={'csrf_token': csrf, 'access_token': 'test-access-code-that-is-not-a-secret'})
    assert response.status_code == 302
    with client.session_transaction() as session: return session['csrf']

def test_access_control_and_removed_routes(app):
    client = app.test_client()
    assert client.get('/api/jobs').status_code == 401
    assert client.get('/login', headers={'Host': 'evil.example'}).status_code == 400
    csrf = sign_in(client)
    assert client.post('/api/catalog', json={}).status_code == 403
    assert client.post('/api/catalog', json={}, headers={'X-CSRF-Token': csrf, 'Origin': 'https://evil.example'}).status_code == 403
    assert client.get('/hunyuan3d').status_code == 404
    assert client.get('/lyrics-embedder/logs?log_file=/etc/passwd').status_code == 404
    response = client.post('/api/catalog', json={'size': 1}, headers={'X-CSRF-Token': csrf})
    assert response.status_code == 200 and len(response.json['items']) <= 1
    assert response.headers['Cache-Control'] == 'no-store'
    assert client.get('/api/vaults/' + 'a'*32 + '/contents').status_code == 423
