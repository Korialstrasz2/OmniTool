"""Real Flask route/authentication tests; requirements-dev.txt is required."""
from pathlib import Path
import re
import time
import pytest
pytest.importorskip('flask')
from app import create_app

@pytest.fixture
def application(tmp_path):
    app=create_app(Path(__file__).parents[1], access_token='test-code-for-workbench-integration')
    app.config['TESTING']=True
    app.extensions['omnitool_renamer'].state=tmp_path/'journal'
    yield app
    app.extensions['omnitool_file_tasks'].shutdown()
    app.extensions['omnitool_shutdown']()

def login(client):
    response=client.get('/login')
    csrf=re.search(r'name="csrf_token" value="([^"]+)"',response.text).group(1)
    assert client.post('/login',data={'csrf_token':csrf,'access_token':'test-code-for-workbench-integration'}).status_code==302
    with client.session_transaction() as s:return {'X-CSRF-Token':s['csrf']}

def finish(app,client,token):
    with client.session_transaction() as s:owner=s['sid']
    app.extensions['omnitool_file_tasks'].get(owner,token)['future'].result(timeout=30)
    response=client.get('/api/files/tasks/'+token)
    assert response.status_code==200
    assert response.json['state']=='succeeded',response.json
    return response.json['result']

def test_authentication_and_pages(application):
    c=application.test_client()
    assert c.get('/files/dual').status_code==302
    assert c.get('/api/files/tasks/'+'a'*32).status_code==401
    h=login(c)
    assert c.post('/api/files/dual/open',json={}).status_code==403
    for page in ('dual','compare','convert'):
        r=c.get('/files/'+page);assert r.status_code==200 and r.headers['Cache-Control']=='no-store'
    assert c.post('/api/files/dual/open',json={},headers={**h,'Origin':'https://evil.example'}).status_code==403

def test_dual_routes_and_consumed_preview(application,tmp_path):
    a,b=tmp_path/'left',tmp_path/'right';a.mkdir();b.mkdir()
    (a/'Photo.JPG').write_text('reference');(b/'DSC.PNG').write_text('original')
    c=application.test_client();h=login(c)
    r=c.post('/api/files/dual/open',json={'left':str(a),'right':str(b)},headers=h)
    assert r.status_code==200,r.json
    token=r.json['token']
    r=c.get(f'/api/files/dual/{token}/list/right');assert r.json['items'][0]['path']=='DSC.PNG'
    r=c.post('/api/files/dual/preview',json={'token':token,'pairs':[{'left':0,'right':0}]},headers=h)
    assert r.json['rows'][0]['target']=='Photo.PNG';preview=r.json['token']
    payload={'token':preview,'confirmation':'RENAME 1'}
    r=c.post('/api/files/dual/apply',json=payload,headers=h);assert r.status_code==200,r.json
    assert (b/'Photo.PNG').read_text()=='original'
    assert c.post('/api/files/dual/apply',json=payload,headers=h).status_code==400
    r=c.post('/api/maintenance/rename/recover',json={'operation':r.json['id'],'confirmation':'UNDO'},headers=h)
    assert r.status_code==200 and (b/'DSC.PNG').exists()

def test_inventory_owner_scope(application,tmp_path):
    a,b=tmp_path/'a',tmp_path/'b';a.mkdir();b.mkdir()
    c=application.test_client();h=login(c)
    token=c.post('/api/files/dual/open',json={'left':str(a),'right':str(b)},headers=h).json['token']
    other=application.test_client();login(other)
    assert other.get(f'/api/files/dual/{token}/list/left').status_code==400

def test_compare_report_pagination_export(application,tmp_path):
    a,b=tmp_path/'a',tmp_path/'b';a.mkdir();b.mkdir()
    for i in range(55):(a/f'file-{i}').touch()
    (a/'=formula').touch()
    c=application.test_client();h=login(c)
    r=c.post('/api/files/compare',json={'left':str(a),'right':str(b)},headers=h);assert r.status_code==202
    token=r.json['task'];result=finish(application,c,token)
    assert result['total']==56 and len(result['rows'])==50
    assert len(c.get(f'/api/files/tasks/{token}?page=2').json['result']['rows'])==6
    csv=c.get(f'/api/files/tasks/{token}/export/csv')
    assert csv.status_code==200 and "'=formula" in csv.text
    full=c.get(f'/api/files/tasks/{token}/export/json');assert len(full.json['rows'])==56
    other=application.test_client();login(other)
    assert other.get(f'/api/files/tasks/{token}/export/json').status_code==400

def test_conversion_uses_reviewed_options(application,tmp_path):
    from PIL import Image
    source=tmp_path/'image.png';Image.new('RGB',(10,20)).save(source)
    out=tmp_path/'out';c=application.test_client();h=login(c)
    r=c.post('/api/files/convert/preview',json={'input':str(source),'out':str(out),'dpi':72,'max_pages':1},headers=h)
    token=r.json['task'];finish(application,c,token)
    assert not out.exists()
    assert c.post('/api/files/convert/apply',json={'task':token,'confirmation':'wrong'},headers=h).status_code==400
    payload={'task':token,'confirmation':'CONVERT 1','out':str(tmp_path/'not-reviewed')}
    r=c.post('/api/files/convert/apply',json=payload,headers=h);assert r.status_code==202
    finish(application,c,r.json['task']);assert (out/'image.png').exists() and not (tmp_path/'not-reviewed').exists()
    assert c.post('/api/files/convert/apply',json=payload,headers=h).status_code==400

def test_invalid_requests_do_not_start_workers(application):
    c=application.test_client();h=login(c)
    for body in ([],{'left':False,'right':'x'},{'left':'x','right':'y','hash_budget_mib':True}):
        assert c.post('/api/files/compare',json=body,headers=h).status_code==400
    assert not application.extensions['omnitool_file_tasks'].items
