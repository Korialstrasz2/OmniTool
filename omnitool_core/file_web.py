"""Authenticated file-workbench routes; the parent app enforces access and CSRF."""
from __future__ import annotations
import atexit
import importlib.util
import base64
import csv
import io
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

from flask import Blueprint, abort, jsonify, render_template, request, send_file, session

from .conversion import output_path
from .file_tasks import FileTasks, Leases
from .file_workbench import FileToolError, dual_plan, integer, inventory
from .rename import RenameError


def register(app):
    bp = Blueprint('file_workbench', __name__)
    base = Path(app.template_folder).parent
    leases, tasks = Leases(), FileTasks(base)
    engine = app.extensions['omnitool_renamer']
    thumbs = threading.BoundedSemaphore(2)

    def owner():
        return session['sid']

    def body():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise FileToolError('Expected a JSON object')
        return data

    def path_value(data, name):
        value = data.get(name)
        if not isinstance(value, str) or not value.strip() or len(value) > 4096 or '\x00' in value:
            raise FileToolError(f'Enter a local {name} path')
        return Path(value)

    def number_arg(name, default):
        try:
            return int(request.args.get(name, default))
        except (ValueError, TypeError):
            raise FileToolError('Invalid page number') from None

    @bp.errorhandler(FileToolError)
    @bp.errorhandler(RenameError)
    @bp.errorhandler(OSError)
    def bad_request(error):
        return jsonify(error=str(error)), 400

    @bp.get('/files/<kind>')
    def page(kind):
        if kind not in {'dual', 'compare', 'convert'}:
            abort(404)
        return render_template('file_workbench.html', mode=kind, conversion_ready=all(importlib.util.find_spec(m) for m in ('PIL', 'fitz')))

    @bp.post('/api/files/dual/open')
    def dual_open():
        data = body()
        left, right = inventory(path_value(data, 'left'), False), inventory(path_value(data, 'right'), False)
        a, b = Path(left['root']), Path(right['root'])
        if a.is_relative_to(b) or b.is_relative_to(a):
            raise FileToolError('Choose separate, non-overlapping folders')
        token = leases.put(owner(), 'dual-inventory', {'left': left, 'right': right})
        return jsonify(token=token, left=left['root'], right=right['root'],
                       skipped={'left': len(left['skipped']), 'right': len(right['skipped'])})

    def side_rows(token, side):
        if side not in {'left', 'right'}:
            abort(404)
        scan = leases.get(owner(), token, 'dual-inventory')[side]
        return scan, [r for r in scan['rows'] if r['kind'] == 'file']

    @bp.get('/api/files/dual/<token>/list/<side>')
    def listing(token, side):
        scan, rows = side_rows(token, side)
        query = request.args.get('query', '')[:256].casefold()
        chosen = [(i, r) for i, r in enumerate(rows) if query in r['path'].casefold()]
        pages = max(1, (len(chosen) + 23) // 24)
        page = min(max(1, number_arg('page', 1)), pages)
        return jsonify(root=scan['root'], total=len(chosen), page=page, pages=pages,
                       items=[{'index': i, 'path': r['path'], 'bytes': r['bytes']} for i, r in chosen[(page-1)*24:page*24]])

    @bp.get('/api/files/dual/<token>/thumbnail/<side>/<int:index>')
    def thumbnail(token, side, index):
        scan, rows = side_rows(token, side)
        if index >= len(rows):
            abort(404)
        if not thumbs.acquire(blocking=False):
            abort(429)
        process = None
        try:
            row = rows[index]
            data = {'action': 'thumbnail', 'path': str(Path(scan['root']) / row['path']), 'stamp': row['_stamp']}
            env = os.environ.copy(); env.pop('OMNITOOL_ACCESS_TOKEN', None)
            kwargs = {'start_new_session': True} if os.name != 'nt' else {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
            process = subprocess.Popen([sys.executable, '-B', '-m', 'omnitool_core.file_worker'], cwd=base,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, **kwargs)
            raw, _ = process.communicate(json.dumps(data).encode(), timeout=8)
            if len(raw) > 400000:
                raise FileToolError('Thumbnail response too large')
            result = json.loads(raw)
            if not result.get('ok'):
                raise FileToolError(result.get('error', 'Thumbnail unavailable'))
            return send_file(io.BytesIO(base64.b64decode(result['result']['png'], validate=True)), mimetype='image/png', max_age=0)
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            raise FileToolError('Thumbnail unavailable or timed out; selection by filename still works') from None
        finally:
            if process and process.poll() is None:
                tasks.terminate(process); process.communicate()
            thumbs.release()

    @bp.post('/api/files/dual/preview')
    def dual_preview():
        data = body()
        token = data.get('token', '')
        source = leases.get(owner(), token, 'dual-inventory')
        pairs = data.get('pairs')
        if not isinstance(pairs, list) or not 1 <= len(pairs) <= 500:
            raise FileToolError('Stage 1 to 500 mappings')
        left = [r for r in source['left']['rows'] if r['kind'] == 'file']
        right = [r for r in source['right']['rows'] if r['kind'] == 'file']
        names = []
        for pair in pairs:
            if not isinstance(pair, dict):
                raise FileToolError('Invalid mapping')
            i = integer(pair.get('left'), 0, len(left)-1, 'Reference index')
            j = integer(pair.get('right'), 0, len(right)-1, 'Target index')
            names.append({'reference': left[i]['path'], 'source': right[j]['path']})
        for scan in source.values():
            if inventory(Path(scan['root']), False)['fingerprint'] != scan['fingerprint']:
                raise FileToolError('Folders changed since loading; reload them and stage a fresh mapping')
        plan = dual_plan(Path(source['left']['root']), Path(source['right']['root']), names)
        preview = leases.put(owner(), 'dual-plan', plan)
        return jsonify(token=preview, count=len(plan['rows']), conflicts=plan['conflicts'],
                       rows=[{k: v for k, v in row.items() if k != 'identity'} for row in plan['rows']])

    @bp.post('/api/files/dual/apply')
    def dual_apply():
        data = body()
        with leases.lock:
            plan = leases.get(owner(), data.get('token', ''), 'dual-plan')
            if data.get('confirmation') != f"RENAME {len(plan['rows'])}":
                raise FileToolError(f"Type RENAME {len(plan['rows'])} to apply this preview")
            leases.get(owner(), data['token'], 'dual-plan', consume=True)
        doc = engine.apply_dual(Path(plan['reference']), Path(plan['root']), plan['pairs'], plan['fingerprint'])
        return jsonify(id=doc['id'], status=doc['status'], count=len(doc['rows']))

    @bp.post('/api/files/leases/<token>/release')
    def release(token):
        leases.discard(owner(), token)
        return jsonify(ok=True)

    @bp.post('/api/files/compare')
    def compare_start():
        data = body()
        value = {'action': 'compare', 'left': str(path_value(data, 'left')), 'right': str(path_value(data, 'right')),
                 'mode': data.get('mode', 'content'), 'recursive': data.get('recursive', True),
                 'hash_budget_mib': integer(data.get('hash_budget_mib', 1024), 1, 4096, 'Hash budget')}
        if value['mode'] not in {'content', 'paths', 'stems'} or not isinstance(value['recursive'], bool):
            raise FileToolError('Invalid comparison options')
        return jsonify(task=tasks.submit(owner(), value)), 202

    @bp.post('/api/files/convert/preview')
    def convert_preview():
        data = body()
        value = {'action': 'inspect', 'input': str(path_value(data, 'input')),
                 'out': str(output_path(path_value(data, 'out'))),
                 'dpi': integer(data.get('dpi', 220), 36, 600, 'DPI'),
                 'max_pages': integer(data.get('max_pages', 100), 1, 100, 'Maximum pages')}
        return jsonify(task=tasks.submit(owner(), value)), 202

    @bp.post('/api/files/convert/apply')
    def convert_apply():
        data = body()
        with tasks.lock:
            task = tasks.get(owner(), data.get('task', ''))
            if task['action'] != 'inspect' or task['state'] != 'succeeded' or task.get('used'):
                raise FileToolError('Create a fresh successful conversion preview')
            info = task['result']
            if data.get('confirmation') != f"CONVERT {info['selected_pages']}":
                raise FileToolError(f"Type CONVERT {info['selected_pages']} to confirm")
            value = dict(task['request'], action='convert', expected_sha256=info['sha256'])
            token = tasks.submit(owner(), value)
            task['used'] = True
        return jsonify(task=token), 202

    @bp.get('/api/files/tasks/<token>')
    def task_status(token):
        task = tasks.get(owner(), token)
        value = {'state': task['state'], 'error': task['error'], 'action': task['action']}
        result = task['result']
        if result is not None:
            result = dict(result)
            if 'rows' in result:
                rows = result.pop('rows')
                query = request.args.get('query', '')[:256].casefold()
                status = request.args.get('status', '')
                rows = [r for r in rows if (not status or r['status'] == status) and query in r['path'].casefold()]
                pages = max(1, (len(rows) + 49) // 50); page = min(max(1, number_arg('page', 1)), pages)
                result.update(rows=rows[(page-1)*50:page*50], total=len(rows), page=page, pages=pages)
            value['result'] = result
        return jsonify(value)

    @bp.post('/api/files/tasks/<token>/dismiss')
    def dismiss(token):
        tasks.dismiss(owner(), token)
        return jsonify(ok=True)

    @bp.get('/api/files/tasks/<token>/export/<format>')
    def export(token, format):
        task = tasks.get(owner(), token)
        if task['state'] != 'succeeded':
            raise FileToolError('Complete the operation before exporting')
        result = task['result']
        if format == 'json':
            raw = json.dumps(result, ensure_ascii=True, indent=2).encode()
            mime = 'application/json'
        elif format == 'csv' and task['action'] == 'compare':
            out = io.StringIO(newline=''); writer = csv.writer(out)
            writer.writerow(['path', 'status', 'left', 'right'])
            for row in result['rows']:
                values = [row['path'], row['status'], ' | '.join(row['left']), ' | '.join(row['right'])]
                writer.writerow(["'" + v if re.match(r'^[\s]*[=+\-@]', v) else v for v in values])
            raw = out.getvalue().encode('utf-8-sig'); mime = 'text/csv'
        else:
            abort(404)
        return send_file(io.BytesIO(raw), mimetype=mime, as_attachment=True,
                         download_name=f'omnitool-{task["action"]}.{format}', max_age=0)

    app.register_blueprint(bp)
    app.extensions['omnitool_file_tasks'] = tasks
    app.extensions['omnitool_file_leases'] = leases
    atexit.register(tasks.shutdown)
