"""Optional: pip install playwright; playwright install chromium; python tests/ui_smoke.py. Uses mock APIs."""
import os, shutil
import json
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[1]
reports=root/'.omnitool-local'/'ui-preview'
reports.mkdir(parents=True,exist_ok=True)
env=Environment(loader=FileSystemLoader(root/'templates'),autoescape=select_autoescape())
def url_for(name,**kw):return '/static/'+kw['filename'] if name=='static' else '/'+name
items=json.loads((root/'tools.json').read_text())+[json.loads((root/'tools/duplicate-finder/tool.json').read_text())]
for t in items:t['availability']='disabled' if t.get('disabled') else 'ready'
items.sort(key=lambda t:t['name'])
mock='''window.scale=false; window.OmniAPI=async (url,data={})=>{
if(url==='/api/vaults')return {items:[{id:'a'.repeat(32),locked:true}]};
let tools=window.scale?Array.from({length:1000},(_,i)=>({...window.fixture[0],id:'tool-'+i,name:'Tool '+String(i).padStart(4,'0')})):window.fixture;
const count=tools.length;
tools=tools.filter(t=>(t.name+' '+t.description).toLowerCase().includes((data.query||'').toLowerCase())&&(!data.folder||t.folder===data.folder)&&(!data.status||t.availability===data.status));
if(data.favorites_only)tools=tools.filter(t=>(data.pinned||[]).includes(t.id));
return {items:tools.slice((data.page-1)*data.size,data.page*data.size),total:tools.length,total_catalog:count,folders:[...new Set(window.fixture.map(t=>t.folder))].sort(),page:data.page,pages:Math.max(1,Math.ceil(tools.length/data.size)),errors:[]};};'''
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,executable_path=shutil.which('chromium') or None)
 page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
 errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 def render(mode):
  text=env.get_template('index.html').render(mode=mode,csrf_token='mock',signed_in=True,url_for=url_for,get_flashed_messages=lambda **kw:[])
  import re
  text=re.sub(r'<script[^>]*>.*?</script>|<link[^>]*rel="stylesheet"[^>]*>','',text,flags=re.S)
  page.set_content(text)
  page.add_style_tag(content=(root/'static/style.css').read_text())
  page.evaluate('(items)=>window.fixture=items',items)
  page.add_script_tag(content=mock)
  page.add_script_tag(content=(root/'static/workspace.js').read_text())
 render('catalog');page.wait_for_selector('.tool-card');assert page.locator('.tool-card').count()==12
 page.screenshot(path=str(reports/'desktop.png'),full_page=True)
 page.keyboard.press('Control+k');assert page.locator('#search').evaluate('(e)=>e===document.activeElement')
 page.locator('#search').fill('CSV');page.wait_for_timeout(400);assert page.locator('.tool-card').count()==1
 page.locator('#reset').click();page.wait_for_timeout(250)
 page.locator('#add-tool').click();assert page.locator('#instructions').evaluate('(e)=>e.open')
 page.keyboard.press('Escape');assert not page.locator('#instructions').evaluate('(e)=>e.open')
 page.evaluate('window.scale=true');page.locator('#reset').click();page.wait_for_timeout(250);assert page.locator('.tool-card').count()==24
 page.locator('#next').click();page.wait_for_timeout(250);assert page.locator('#page-number').inner_text()=='Page 2 of 42'
 page.evaluate('window.scale=false');page.locator('#reset').click();page.wait_for_timeout(250)
 page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(200)
 assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),page.evaluate('[document.documentElement.scrollWidth,innerWidth]')
 page.screenshot(path=str(reports/'mobile.png'),full_page=True)
 render('vaults');page.wait_for_selector('#results button');assert 'Secret' not in page.locator('#results').inner_text()
 page.locator('#results button').first.click();page.locator('#vault-passphrase').fill('test-only-not-real')
 page.keyboard.press('Escape');page.wait_for_timeout(100);assert page.locator('#vault-passphrase').input_value()==''
 assert not errors,errors
 print('UI smoke passed: 12-tool view, search, Ctrl+K, dialogs, 1,000-tool pagination, mobile overflow, locked metadata, password clearing, no JS errors. Mock API; not Flask integration.')
 browser.close()
