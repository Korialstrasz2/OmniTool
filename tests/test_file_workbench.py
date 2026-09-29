from pathlib import Path
import json
import os
import sys
import time
import pytest

from omnitool_core.file_workbench import FileToolError, compare, component, dual_plan, inventory, read_regular
from omnitool_core.rename import Renamer, RenameError, plan
from omnitool_core.file_tasks import FileTasks, Leases

@pytest.fixture
def folders(tmp_path):
    a, b = tmp_path/'reference', tmp_path/'targets'
    a.mkdir(); b.mkdir()
    (a/'Example.JPG').write_bytes(b'reference')
    (b/'DSC001.png').write_bytes(b'original pixels')
    return a, b, Renamer(tmp_path/'state')

def test_dual_preview_apply_undo(folders):
    a,b,engine = folders
    pairs=[{'reference':'Example.JPG','source':'DSC001.png'}]
    p=dual_plan(a,b,pairs)
    assert p['rows'][0]['target']=='Example.png' and not p['conflicts']
    assert not engine.state.exists()
    doc=engine.apply_dual(a,b,pairs,p['fingerprint'])
    assert (b/'Example.png').read_bytes()==b'original pixels'
    assert (a/'Example.JPG').read_bytes()==b'reference'
    assert engine.recover(doc['id'])['status']=='restored'
    assert (b/'DSC001.png').read_bytes()==b'original pixels'

@pytest.mark.parametrize('change',['reference','target','new-name','directory'])
def test_dual_stale_plan(folders,change):
    a,b,engine=folders; pairs=[{'reference':'Example.JPG','source':'DSC001.png'}]; p=dual_plan(a,b,pairs)
    if change=='reference': (a/'Example.JPG').write_bytes(b'new reference')
    elif change=='target': (b/'DSC001.png').write_bytes(b'changed bytes')
    elif change=='new-name': (b/'unrelated').touch()
    else: (b/'new-directory').mkdir()
    with pytest.raises(RenameError,match='changed after preview'): engine.apply_dual(a,b,pairs,p['fingerprint'])
    assert (b/'DSC001.png').exists()

@pytest.mark.parametrize('name,kind',[('Example.png','file'),('example.PNG','file'),('Example.png','directory')])
def test_dual_collision(folders,name,kind):
    a,b,engine=folders
    p=b/name
    p.mkdir() if kind=='directory' else p.write_bytes(b'keep')
    pairs=[{'reference':'Example.JPG','source':'DSC001.png'}]; preview=dual_plan(a,b,pairs)
    assert preview['conflicts']
    with pytest.raises(RenameError,match='collisions'):engine.apply_dual(a,b,pairs,preview['fingerprint'])
    assert (b/'DSC001.png').exists()

def test_duplicate_and_overlapping_pairs(folders):
    a,b,_=folders
    p={'reference':'Example.JPG','source':'DSC001.png'}
    with pytest.raises(FileToolError):dual_plan(a,b,[p,p])
    with pytest.raises(FileToolError):dual_plan(a,a,[p])

@pytest.mark.parametrize('name',['../x','/x','C:\\x','bad:name','CON.txt','NUL','trailing.','a/b','a\x00b',''])
def test_bad_components(name):
    with pytest.raises(FileToolError):component(name)

def test_case_only_dual(folders):
    a,b,e=folders;(a/'dsc001.jpg').write_bytes(b'ref')
    pairs=[{'reference':'dsc001.jpg','source':'DSC001.png'}]
    p=dual_plan(a,b,pairs);doc=e.apply_dual(a,b,pairs,p['fingerprint'])
    assert 'dsc001.png' in {x.name for x in b.iterdir()}
    e.recover(doc['id']);assert 'DSC001.png' in {x.name for x in b.iterdir()}

def test_dual_failure_rolls_back(folders,monkeypatch):
    a,b,e=folders;import omnitool_core.rename as module
    old=module.move_noreplace;count=0
    def fail_once(src,dst):
        nonlocal count
        count+=1
        if count==2:raise OSError('simulated')
        old(src,dst)
    monkeypatch.setattr(module,'move_noreplace',fail_once)
    pairs=[{'reference':'Example.JPG','source':'DSC001.png'}];p=dual_plan(a,b,pairs)
    with pytest.raises(RenameError,match='restored'):e.apply_dual(a,b,pairs,p['fingerprint'])
    assert (b/'DSC001.png').exists() and not list(b.glob('.omnitool-*'))

def test_dual_interrupted_recovery(folders,monkeypatch):
    a,b,e=folders;old=e._save;hit=False
    def crash(doc):
        nonlocal hit
        if not hit and doc['pending'] is None and list(b.glob('.omnitool-*')):
            hit=True;raise KeyboardInterrupt('interruption')
        old(doc)
    monkeypatch.setattr(e,'_save',crash)
    pairs=[{'reference':'Example.JPG','source':'DSC001.png'}];p=dual_plan(a,b,pairs)
    with pytest.raises(KeyboardInterrupt):e.apply_dual(a,b,pairs,p['fingerprint'])
    monkeypatch.setattr(e,'_save',old)
    e.recover(e.recent()[0]['id']);assert (b/'DSC001.png').exists()

def test_dual_recovery_refuses_modified_output(folders):
    a,b,e=folders;pairs=[{'reference':'Example.JPG','source':'DSC001.png'}];p=dual_plan(a,b,pairs)
    doc=e.apply_dual(a,b,pairs,p['fingerprint']);(b/'Example.png').write_bytes(b'changed')
    with pytest.raises(RenameError):e.recover(doc['id'])

def test_lowercase_backward_compatibility(tmp_path):
    root=tmp_path/'files';root.mkdir();(root/'Photo.JPG').write_bytes(b'picture')
    e=Renamer(tmp_path/'state');doc=e.apply(root,plan(root)['fingerprint'])
    # A pre-existing v1 journal has no kind field.
    data=e.load(doc['id']);data.pop('kind');e._save(data)
    assert e.recover(doc['id'])['status']=='restored' and (root/'Photo.JPG').exists()

def test_recursive_content_compare(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b';a.mkdir();b.mkdir()
    for root in (a,b):
        (root/'nested').mkdir();(root/'nested'/'same.txt').write_text('same');(root/'empty').mkdir()
    (a/'change.bin').write_bytes(b'aaaa');(b/'change.bin').write_bytes(b'bbbb')
    (a/'left.txt').touch();(b/'right.txt').touch()
    (a/'type').mkdir();(b/'type').touch()
    report=compare(a,b)
    assert report['read_only'] and report['scope_complete']
    statuses={r['path']:r['status'] for r in report['rows']}
    assert statuses['nested/same.txt']=='same-content'
    assert statuses['change.bin']=='different-content'
    assert statuses['left.txt']=='left-only' and statuses['type']=='type-conflict'
    assert statuses['empty']=='directory-both'
    assert (a/'change.bin').read_bytes()==b'aaaa'

def test_paths_preserve_extensions_and_no_false_content_match(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b';a.mkdir();b.mkdir()
    (a/'same.txt').write_text('aaa');(b/'same.txt').write_text('bbb');(b/'same.jpg').touch()
    paths=compare(a,b,'paths');assert 'same-path' in paths['counts'] and 'same-content' not in paths['counts']
    stems=compare(a,b,'stems');assert stems['rows'][0]['status']=='ambiguous'

def test_top_level_and_skipped_paths(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b';a.mkdir();b.mkdir();(a/'sub').mkdir();(a/'sub'/'inside').touch();(a/'.git').mkdir()
    report=compare(a,b,'paths',False)
    assert 'sub/inside' not in {r['path'] for r in report['rows']}
    assert not report['scope_complete'] and report['skipped']['left'][0]['path']=='.git'

def test_read_limits_and_links(tmp_path):
    p=tmp_path/'file';p.write_bytes(b'x'*10)
    with pytest.raises(FileToolError):read_regular(p,9)
    try:(tmp_path/'link').symlink_to(p)
    except OSError:pytest.skip('No symlink privileges')
    with pytest.raises(FileToolError):read_regular(tmp_path/'link',100)
    assert any(x['reason']=='link or reparse point' for x in inventory(tmp_path)['skipped'])

def test_hash_budget(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b';a.mkdir();b.mkdir()
    for root in (a,b):(root/'large').write_bytes(b'x'*(600*1024))
    with pytest.raises(FileToolError,match='budget'):compare(a,b,hash_budget_mib=1)

def test_changed_during_hash_fails(tmp_path,monkeypatch):
    a,b=tmp_path/'a',tmp_path/'b';a.mkdir();b.mkdir()
    for root in (a,b):(root/'file').write_bytes(b'same')
    import omnitool_core.file_workbench as module
    original=module._digest
    def changed(root,row,budget):
        result=original(root,row,budget)
        (a/'new').touch()
        return result
    monkeypatch.setattr(module,'_digest',changed)
    with pytest.raises(FileToolError,match='changed during'):compare(a,b)

def test_lease_ownership_consumption_expiry():
    leases=Leases();token=leases.put('alice','preview',{'x':1})
    with pytest.raises(FileToolError):leases.get('bob',token,'preview')
    with pytest.raises(FileToolError):leases.get('alice',token,'different-kind')
    assert leases.get('alice',token,'preview',consume=True)=={'x':1}
    with pytest.raises(FileToolError):leases.get('alice',token,'preview')
    token=leases.put('alice','preview',{});leases.items[('alice',token)]['expires']=0
    with pytest.raises(FileToolError):leases.get('alice',token,'preview')

def test_worker_comparison_and_owner_isolation(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b';a.mkdir();b.mkdir()
    for root in (a,b):(root/'same').write_bytes(b'same')
    tasks=FileTasks(Path(__file__).parents[1])
    try:
        token=tasks.submit('alice',{'action':'compare','left':str(a),'right':str(b)})
        tasks.get('alice',token)['future'].result(timeout=20)
        task=tasks.get('alice',token)
        assert task['state']=='succeeded',task['error']
        assert task['result']['counts']['same-content']==1
        with pytest.raises(FileToolError):tasks.get('bob',token)
        tasks.dismiss('alice',token)
        with pytest.raises(FileToolError):tasks.get('alice',token)
    finally:tasks.shutdown()
