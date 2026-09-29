import base64
import io
import json
from pathlib import Path
import pytest
from PIL import Image
import fitz
from omnitool_core import conversion
from omnitool_core.conversion import convert, decode_input, inspect_input
from omnitool_core.file_workbench import FileToolError
from omnitool_core.file_tasks import FileTasks

@pytest.fixture
def picture(tmp_path):
    p=tmp_path/'input.png';Image.new('RGBA',(40,20),(12,34,56,100)).save(p);return p

@pytest.mark.parametrize('encoded',[False,True,'data-uri'])
def test_image_conversion_no_overwrite(picture,tmp_path,encoded):
    source=picture
    if encoded:
        source=tmp_path/'input.b64';text=base64.b64encode(picture.read_bytes()).decode()
        source.write_text(('data:image/png;base64,' if encoded=='data-uri' else '')+text)
    p=inspect_input(source);out=tmp_path/'output'
    assert not out.exists() and p['selected_pages']==1
    result=convert(source,out,expected_sha256=p['sha256'])
    assert result['completed'] and picture.exists()
    with Image.open(out/'image.png') as im:
        assert im.size==(40,20) and im.getpixel((0,0))[3]==100
    assert json.loads((out/'conversion.json').read_text())['sha256']==p['sha256']
    with pytest.raises(FileToolError,match='NEW'):convert(source,out)

@pytest.mark.parametrize('text',['','%%%','Zm9v@@==','Zm9v==','Zh==','data:text/html;base64,SGVsbG8=','data:image/png,abc','€'])
def test_invalid_base64(tmp_path,text):
    p=tmp_path/'data.txt';p.write_text(text)
    with pytest.raises((FileToolError,ValueError)):decode_input(p)

@pytest.mark.parametrize('dpi,pages',[(0,1),(601,1),(True,1),(220,0),(220,101),(220,True)])
def test_cli_library_bounds(picture,dpi,pages):
    with pytest.raises(FileToolError):inspect_input(picture,dpi,pages)

def test_pdf_inspect_render_and_truncation(tmp_path):
    p=tmp_path/'input.pdf'
    doc=fitz.open()
    for _ in range(3):doc.new_page(width=72,height=144)
    doc.save(p);doc.close()
    info=inspect_input(p,dpi=72,max_pages=2)
    assert info['omitted_pages']==1 and info['images'][0]['width']==72
    out=tmp_path/'result';convert(p,out,dpi=72,max_pages=2,expected_sha256=info['sha256'])
    assert len(list(out.glob('*.png')))==2
    with Image.open(out/'page-001.png') as image:assert image.size==(72,144)

def test_pdf_password_rejected(tmp_path):
    p=tmp_path/'encrypted.pdf';doc=fitz.open();doc.new_page()
    doc.save(p,encryption=fitz.PDF_ENCRYPT_AES_256,owner_pw='owner',user_pw='password');doc.close()
    with pytest.raises(FileToolError,match='Password'):inspect_input(p)

def test_oversized_pdf_page_rejected_before_render(tmp_path):
    p=tmp_path/'huge.pdf';doc=fitz.open();doc.new_page(width=10000,height=10000);doc.save(p);doc.close()
    with pytest.raises(FileToolError,match='20 million'):inspect_input(p,600)

def test_input_change_rejected(picture,tmp_path):
    info=inspect_input(picture);Image.new('RGB',(5,5)).save(picture)
    with pytest.raises(FileToolError,match='changed since preview'):convert(picture,tmp_path/'out',expected_sha256=info['sha256'])
    assert not (tmp_path/'out').exists()

def test_conversion_failure_cleans_stage(picture,tmp_path,monkeypatch):
    def fail(*a,**k):raise OSError('simulated render failure')
    monkeypatch.setattr(conversion,'move_noreplace',fail)
    with pytest.raises(OSError):convert(picture,tmp_path/'out')
    assert not (tmp_path/'out').exists() and not list(tmp_path.glob('.omnitool-convert-*'))

def test_competing_output_never_overwritten(picture,tmp_path,monkeypatch):
    old=conversion.move_noreplace;out=tmp_path/'out'
    def race(src,dst):
        dst.mkdir();(dst/'keep').write_text('keep');old(src,dst)
    monkeypatch.setattr(conversion,'move_noreplace',race)
    with pytest.raises(FileExistsError):convert(picture,out)
    assert (out/'keep').read_text()=='keep' and not (out/'image.png').exists()

def test_exif_orientation_and_metadata_stripping(tmp_path):
    p=tmp_path/'image.jpg';image=Image.new('RGB',(30,10));exif=Image.Exif();exif[274]=6;exif[315]='private author';image.save(p,exif=exif)
    info=inspect_input(p);assert info['images'][0]['width']==10
    convert(p,tmp_path/'out')
    with Image.open(tmp_path/'out/image.png') as output:
        assert output.size==(10,30) and not output.getexif()

def test_multiframe_rejected(tmp_path):
    p=tmp_path/'anim.gif';Image.new('RGB',(10,10),'red').save(p,save_all=True,append_images=[Image.new('RGB',(10,10),'blue')])
    with pytest.raises(FileToolError,match='Multi-frame'):inspect_input(p)

def test_decoded_size_limit(picture,monkeypatch):
    monkeypatch.setattr(conversion,'MAX_DECODED',2)
    with pytest.raises(FileToolError,match='24 MiB'):inspect_input(picture)

def test_worker_real_conversion(picture,tmp_path):
    info=inspect_input(picture);tasks=FileTasks(Path(__file__).parents[1])
    try:
        token=tasks.submit('owner',{'action':'convert','input':str(picture),'out':str(tmp_path/'worker-output'),
                                 'expected_sha256':info['sha256'],'dpi':72,'max_pages':1})
        task=tasks.get('owner',token);task['future'].result(timeout=20)
        assert task['state']=='succeeded',task['error']
        assert (tmp_path/'worker-output/image.png').exists()
    finally:tasks.shutdown()


def test_selected_image_thumbnail_is_bounded(picture):
    from omnitool_core.file_worker import thumbnail
    from omnitool_core.file_workbench import stamp
    data = thumbnail(picture, stamp(picture.lstat()))
    with Image.open(io.BytesIO(data)) as image:
        assert image.format == 'PNG' and max(image.size) <= 192
    assert picture.exists()

def test_thumbnail_refuses_stale_selection(picture):
    from omnitool_core.file_worker import thumbnail
    from omnitool_core.file_workbench import stamp
    expected = stamp(picture.lstat())
    Image.new('RGB', (5, 5)).save(picture)
    with pytest.raises(FileToolError, match='changed'):
        thumbnail(picture, expected)
