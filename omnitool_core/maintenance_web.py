"""Authenticated workspace pages for the repaired maintenance tools.

The parent app's local-access, host/origin, CSRF and no-store policies cover all
routes. Browser records never pass through these endpoints.
"""
from __future__ import annotations
import io
import secrets
import threading
import time
from pathlib import Path
from flask import Blueprint, abort, jsonify, render_template, request, send_file, session
from .rename import Renamer, RenameError, plan

from .browser_package import extension_archive


def register(app, state: Path | None = None) -> None:
    bp = Blueprint('maintenance', __name__)
    base = Path(app.template_folder).parent
    engine = Renamer(state)
    previews: dict[tuple[str, str], dict] = {}
    mutex = threading.RLock()

    def body():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise RenameError('Expected a JSON object')
        return data

    def get_preview(token: str) -> dict:
        with mutex:
            for key, value in list(previews.items()):
                if value['expires'] <= time.monotonic():
                    previews.pop(key, None)
            value = previews.get((session['sid'], token))
            if value is None:
                raise RenameError('Preview expired or belongs to another session; preview again')
            return value

    def summary(value: dict, token: str, page: int = 1):
        data = value['plan']
        pages = max(1, (len(data['rows']) + 49) // 50)
        page = min(max(1, page), pages)
        return dict(token=token, root=data['root'], count=len(data['rows']), scanned=data['scanned'],
                    conflicts=data['conflicts'], skipped=data['skipped'][:100], skipped_count=len(data['skipped']),
                    rows=data['rows'][(page-1)*50:page*50], page=page, pages=pages)

    @bp.errorhandler(RenameError)
    @bp.errorhandler(OSError)
    def bad_request(error):
        return jsonify(error=str(error)), 400

    @bp.get('/maintenance/browser/<kind>')
    def browser_guide(kind):
        if kind not in {'history', 'cookies'}:
            abort(404)
        return render_template('maintenance_browser.html', kind=kind)

    @bp.get('/maintenance/browser/package/<flavor>')
    def browser_package(flavor):
        if flavor not in {'chromium', 'firefox'}:
            abort(404)
        return send_file(io.BytesIO(extension_archive(base, flavor)), as_attachment=True,
                         download_name=f'omnitool-browser-{flavor}.zip', mimetype='application/zip', max_age=0)

    @bp.get('/maintenance/lowercase')
    def lowercase():
        return render_template('lowercase.html')

    @bp.post('/api/maintenance/rename/preview')
    def preview():
        data = body()
        root = data.get('root')
        if not isinstance(root, str) or not root.strip() or len(root) > 4096:
            raise RenameError('Enter an existing working-folder path')
        value = {'plan': plan(Path(root)), 'expires': time.monotonic() + 600}
        token = secrets.token_hex(16)
        with mutex:
            for key, saved in list(previews.items()):
                if saved['expires'] <= time.monotonic() or key[0] == session['sid']:
                    previews.pop(key, None)
            if len(previews) >= 16:
                raise RenameError('Too many open previews; close another session or let it expire')
            previews[(session['sid'], token)] = value
        return jsonify(summary(value, token))

    @bp.get('/api/maintenance/rename/preview/<token>')
    def preview_page(token):
        return jsonify(summary(get_preview(token), token, int(request.args.get('page', 1))))

    @bp.post('/api/maintenance/rename/apply')
    def apply():
        data = body()
        token = data.get('token', '')
        if not isinstance(token, str):
            raise RenameError('Invalid preview token')
        with mutex:
            value = get_preview(token)
            count = len(value['plan']['rows'])
            if data.get('confirmation') != f'RENAME {count}':
                raise RenameError(f'Type RENAME {count} to confirm the preview')
            previews.pop((session['sid'], token))  # A preview can authorize only one attempt.
        result = engine.apply(Path(value['plan']['root']), value['plan']['fingerprint'])
        return jsonify(id=result['id'], status=result['status'], count=len(result['rows']))

    @bp.get('/api/maintenance/rename/operations')
    def operations():
        return jsonify(items=engine.recent())

    @bp.post('/api/maintenance/rename/recover')
    def recover():
        data = body()
        if data.get('confirmation') != 'UNDO':
            raise RenameError('Type UNDO to restore original names')
        operation = data.get('operation')
        if not isinstance(operation, str):
            raise RenameError('Select an operation')
        result = engine.recover(operation)
        return jsonify(id=result['id'], status=result['status'], count=len(result['rows']))

    app.register_blueprint(bp)
    app.extensions['omnitool_renamer'] = engine
