"""Optional Chromium checks against real templates/CSS and mocked HTTP APIs.
Install playwright and Chromium separately. This is NOT Flask HTTP integration.
"""
from pathlib import Path
import re
import shutil
from jinja2 import Environment, FileSystemLoader, select_autoescape
from playwright.sync_api import sync_playwright

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'.omnitool-local'/'file-ui';OUT.mkdir(parents=True,exist_ok=True)
env=Environment(loader=FileSystemLoader(BASE/'templates'),autoescape=select_autoescape())
MOCK=r'''
window.calls=[];
window.OmniAPI=async (url,data)=>{
 window.calls.push({url,data});
 if(url.endsWith('/release')||url.endsWith('/dismiss'))return {ok:true};
 if(url==='/api/maintenance/rename/operations')return {items:[]};
 if(url==='/api/files/dual/open')return {token:'inventory',left:data.left,right:data.right,skipped:{left:0,right:1}};
 if(url.includes('/list/')){
  const parsed=new URL(url,'http://localhost');const page=Number(parsed.searchParams.get('page')||1);
  const items=Array.from({length:60},(_,i)=>({index:i,path:(url.includes('/left')?'Reference':'Target')+'-'+String(i).padStart(3,'0')+'.png',bytes:1000}));
  return {items:items.slice((page-1)*24,page*24),page,pages:3,total:60};
 }
 if(url==='/api/files/dual/preview')return {token:'plan',count:1,conflicts:[],rows:[{source:'Target-000.png',target:'Reference-000.png'}]};
 if(url==='/api/files/dual/apply'){if(data.confirmation!=='RENAME 1')throw new Error('Wrong confirmation');return {id:'mock-operation',status:'applied',count:1};}
 if(url==='/api/files/compare')return {task:'comparison'};
 if(url.startsWith('/api/files/tasks/comparison')){
  const params=new URL(url,'http://localhost').searchParams;const page=Number(params.get('page')||1);
  const all=Array.from({length:56},(_,i)=>({path:'nested/file-'+i+'.txt',status:'same-content',left:['nested/file-'+i+'.txt'],right:['nested/file-'+i+'.txt']}));
  return {state:'succeeded',action:'compare',result:{left_root:'/fixture/left',right_root:'/fixture/right',mode:'content',rows:all.slice((page-1)*50,page*50),counts:{'same-content':56},skipped:{left:[],right:[]},total:56,page,pages:2}};
 }
 if(url==='/api/files/convert/preview')return {task:'inspection'};
 if(url==='/api/files/tasks/inspection')return {state:'succeeded',action:'inspect',result:{kind:'pdf',sha256:'a'.repeat(64),selected_pages:2,total_pages:3,omitted_pages:1,decoded_bytes:2048,images:[{filename:'page-001.png',width:72,height:144},{filename:'page-002.png',width:72,height:144}]}};
 if(url==='/api/files/convert/apply'){if(data.confirmation!=='CONVERT 2')throw new Error('Wrong confirmation');return {task:'conversion'};}
 if(url==='/api/files/tasks/conversion')return {state:'succeeded',action:'convert',result:{selected_pages:2,output:'/fixture/new-output'}};
 throw new Error('Unexpected mock request: '+url);
};
'''

def main():
 with sync_playwright() as p:
  browser=p.chromium.launch(headless=True,executable_path=shutil.which('chromium') or None)
  page=browser.new_page(viewport={'width':1440,'height':1000});errors=[]
  page.on('pageerror',lambda e:errors.append(str(e)))
  def render(mode):
   text=env.get_template('file_workbench.html').render(mode=mode,signed_in=True,csrf_token='mock',conversion_ready=True,
    url_for=lambda name,**kw:'/static/'+kw['filename'] if name=='static' else '/'+name,
    get_flashed_messages=lambda **kw:[])
   text=re.sub(r'<script[^>]*>.*?</script>|<link[^>]*rel="stylesheet"[^>]*>','',text,flags=re.S)
   page.set_content(text)
   for name in ('style.css','file_workbench.css'):page.add_style_tag(content=(BASE/'static'/name).read_text())
   page.add_script_tag(content=MOCK);page.add_script_tag(content=(BASE/'static/file_workbench.js').read_text())
  def mobile(mode):
   page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(80)
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),(mode,page.evaluate('[document.documentElement.scrollWidth,innerWidth]'))
   page.evaluate('window.scrollTo(0, 0)'); page.screenshot(path=str(OUT/(mode+'-mobile.png')),full_page=True)
   page.set_viewport_size({'width':1440,'height':1000})
  render('dual');page.locator('#left-root').fill('/fixture/left');page.locator('#right-root').fill('/fixture/right')
  page.locator('#dual-open button').click();page.wait_for_selector('#left-files button')
  assert page.locator('#left-files button').count()==24
  page.locator('#left-next').click();page.wait_for_timeout(80);assert '2/3' in page.locator('#left-page').inner_text()
  page.locator('#left-prev').click();page.wait_for_timeout(80)
  page.locator('#left-files button').first.click();page.locator('#right-files button').first.click();page.locator('#stage-pair').click()
  assert not page.evaluate("window.calls.some(x=>x.url.endsWith('/apply'))")
  page.locator('#preview-pairs').click();page.wait_for_selector('#dual-review:visible')
  assert 'Reference-000.png' in page.locator('#dual-plan').inner_text()
  page.evaluate('window.scrollTo(0, 0)'); page.screenshot(path=str(OUT/'dual-desktop.png'),full_page=True);mobile('dual')
  page.locator('#dual-confirm').fill('RENAME 1');page.locator('#dual-run').click()
  page.wait_for_timeout(80);assert 'Renamed 1 files' in page.locator('#file-message').inner_text()
  render('compare');page.locator('#compare-left').fill('/fixture/left');page.locator('#compare-right').fill('/fixture/right');page.locator('#compare-start').click()
  page.wait_for_selector('#comparison-report:visible');assert page.locator('#compare-rows tr').count()==50
  page.locator('#report-next').click();page.wait_for_timeout(80);assert page.locator('#compare-rows tr').count()==6
  assert '/export/csv' in page.locator('#report-csv').get_attribute('href');mobile('compare')
  render('convert');page.locator('#convert-input').fill('/fixture/input.pdf');page.locator('#convert-out').fill('/fixture/new-output');page.locator('#inspect-start').click()
  page.wait_for_selector('#conversion-review:visible');assert '1 trailing pages' in page.locator('#conversion-omitted').inner_text()
  page.locator('#convert-dpi').fill('144');assert page.locator('#conversion-review').is_hidden()
  page.locator('#inspect-start').click();page.wait_for_selector('#conversion-review:visible')
  page.evaluate('window.scrollTo(0, 0)'); page.screenshot(path=str(OUT/'convert-desktop.png'),full_page=True);mobile('convert')
  page.locator('#convert-confirm').fill('CONVERT 2');page.locator('#convert-run').click();page.wait_for_selector('#conversion-result:visible')
  assert '/fixture/new-output' in page.locator('#conversion-output').inner_text()
  assert not errors,errors
  print('PASS: staged selection, dual pagination/apply, report pagination/export, conversion preview invalidation/apply, 390px layouts, no JS page errors. Mock API only.')
  browser.close()

if __name__=='__main__':main()
