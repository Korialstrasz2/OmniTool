from pathlib import Path
import os
import pytest
from omnitool_core.rename import Renamer, RenameError, identity, move_noreplace, plan

@pytest.fixture
def tree(tmp_path):
    root = tmp_path / 'data'; root.mkdir()
    (root / 'Report.TXT').write_text('report')
    nested = root / 'Nested'; nested.mkdir()
    (nested / 'PHOTO.JPG').write_bytes(b'photo')
    (root / 'unchanged.txt').write_text('same')
    return root, Renamer(tmp_path / 'state')

def test_preview_does_not_write(tree):
    root, engine = tree
    doc = plan(root)
    assert len(doc['rows']) == 2 and not doc['conflicts']
    assert (root / 'Report.TXT').exists() and not engine.state.exists()

def test_apply_and_undo(tree):
    root, engine = tree
    doc = engine.apply(root, plan(root)['fingerprint'])
    assert doc['status'] == 'applied'
    assert (root / 'report.txt').read_text() == 'report'
    assert (root / 'Nested/photo.jpg').read_bytes() == b'photo'
    assert 'Nested' in {p.name for p in root.iterdir()} and 'nested' not in {p.name for p in root.iterdir()}
    assert engine.recover(doc['id'])['status'] == 'restored'
    assert (root / 'Report.TXT').read_text() == 'report'
    assert (root / 'Nested/PHOTO.JPG').exists()
    assert engine.recover(doc['id'])['status'] == 'restored'

@pytest.mark.parametrize('kind', ['file', 'directory', 'unicode'])
def test_collision_blocks_whole_plan(tree, kind):
    root, engine = tree
    if kind == 'unicode':
        (root / 'Straße.TXT').write_text('one')
        (root / 'STRASSE.txt').write_text('two')
    elif kind == 'file':
        if os.name == 'nt': pytest.skip('Distinct case collisions require a case-sensitive filesystem')
        (root / 'report.txt').write_text('other')
    else:
        if os.name == 'nt': pytest.skip('Distinct case collisions require a case-sensitive filesystem')
        (root / 'report.txt').mkdir()
    preview = plan(root)
    assert preview['conflicts']
    with pytest.raises(RenameError, match='collisions'):
        engine.apply(root, preview['fingerprint'])
    assert (root / 'Report.TXT').exists() and (root / 'Nested/PHOTO.JPG').exists()

@pytest.mark.parametrize('change', ['content', 'new-file', 'replacement'])
def test_stale_preview_rejected(tree, change):
    root, engine = tree; preview = plan(root)
    if change == 'content': (root / 'Report.TXT').write_text('changed')
    elif change == 'new-file': (root / 'NEW').write_text('new')
    else:
        (root / 'Report.TXT').unlink()
        (root / 'Report.TXT').write_text('new')
    with pytest.raises(RenameError, match='changed after preview'):
        engine.apply(root, preview['fingerprint'])
    assert (root / 'Nested/PHOTO.JPG').exists()

def test_no_replace_primitive(tree):
    root, _ = tree
    source, target = root / 'Report.TXT', root / 'occupied.txt'
    target.write_text('keep')
    with pytest.raises(FileExistsError): move_noreplace(source, target)
    assert source.read_text() == 'report' and target.read_text() == 'keep'

def test_links_and_exclusions(tree):
    root, _ = tree
    excluded = root / '.git'; excluded.mkdir(); (excluded / 'CONFIG').write_text('keep')
    try: (root / 'LINK').symlink_to(root / 'Report.TXT')
    except OSError: pytest.skip('Symlink privileges unavailable')
    preview = plan(root)
    assert len(preview['rows']) == 2
    assert len(preview['skipped']) == 2

def test_root_symlink_rejected(tree, tmp_path):
    root, _ = tree
    link = tmp_path / 'link'
    try: link.symlink_to(root, target_is_directory=True)
    except OSError: pytest.skip('Symlink privileges unavailable')
    with pytest.raises(RenameError): plan(link)

def test_failure_rolls_back(tree, monkeypatch):
    root, engine = tree
    import omnitool_core.rename as module
    original = module.move_noreplace
    count = 0
    def fail_once(source, target):
        nonlocal count
        count += 1
        if count == 3: raise OSError('simulated failure')
        original(source, target)
    monkeypatch.setattr(module, 'move_noreplace', fail_once)
    with pytest.raises(RenameError, match='original names were restored'):
        engine.apply(root, plan(root)['fingerprint'])
    assert (root / 'Report.TXT').exists() and (root / 'Nested/PHOTO.JPG').exists()
    assert not list(root.rglob('.omnitool-*.tmp'))

def test_crash_between_rename_and_journal_save(tree, monkeypatch):
    root, engine = tree
    original = engine._save
    crashed = False
    def crash(doc):
        nonlocal crashed
        if not crashed and doc['pending'] is None and any(root.rglob('.omnitool-*.tmp')):
            crashed = True
            raise KeyboardInterrupt('power-loss simulation')
        original(doc)
    monkeypatch.setattr(engine, '_save', crash)
    with pytest.raises(KeyboardInterrupt): engine.apply(root, plan(root)['fingerprint'])
    operation = engine.recent()[0]['id']
    assert engine.load(operation)['status'] == 'applying'
    monkeypatch.setattr(engine, '_save', original)
    assert engine.recover(operation)['status'] == 'restored'
    assert (root / 'Report.TXT').exists() and (root / 'Nested/PHOTO.JPG').exists()

def test_undo_refuses_changed_file(tree):
    root, engine = tree
    doc = engine.apply(root, plan(root)['fingerprint'])
    (root / 'report.txt').write_text('changed after rename')
    with pytest.raises(RenameError): engine.recover(doc['id'])
    assert (root / 'Nested/photo.jpg').exists()  # All identities are checked first.

def test_undo_refuses_occupied_original(tree):
    if os.name == 'nt': pytest.skip('Case-sensitive collision fixture')
    root, engine = tree
    doc = engine.apply(root, plan(root)['fingerprint'])
    (root / 'Report.TXT').write_text('new occupant')
    with pytest.raises(RenameError): engine.recover(doc['id'])
    assert (root / 'Report.TXT').read_text() == 'new occupant'
    assert (root / 'report.txt').read_text() == 'report'

@pytest.mark.parametrize('value', ['../evil', '', 'a'*31, '/tmp/test'])
def test_operation_id_validation(tree, value):
    with pytest.raises(RenameError): tree[1].load(value)

def test_journals_outside_target(tree):
    root, _ = tree
    engine = Renamer(root / 'state')
    # State creation itself changes the snapshot; either stale or placement check must stop execution.
    with pytest.raises(RenameError): engine.apply(root, plan(root)['fingerprint'])
    assert (root / 'Report.TXT').exists()

def test_large_operation_is_bounded(tmp_path):
    root=tmp_path/'many';root.mkdir()
    for index in range(501): (root/f'FILE{index}.TXT').touch()
    with pytest.raises(RenameError,match='500 renames'): plan(root)

def test_nonportable_paths_are_skipped(tree):
    if os.name == 'nt': pytest.skip('Windows does not permit colon filenames')
    root,_=tree
    (root/'Bad:Name.TXT').touch()
    assert any(x['reason']=='nonportable path component' for x in plan(root)['skipped'])

def test_invalid_journal_blocks_new_operation(tree):
    root,engine=tree;engine.state.mkdir()
    (engine.state/('a'*32+'.json')).write_text('not-json')
    with pytest.raises(RenameError,match='unfinished operation'): engine.apply(root,plan(root)['fingerprint'])
    assert (root/'Report.TXT').exists()
