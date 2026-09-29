"""Authenticated prompt and lyrics pages, covered by the parent app's security hooks."""
from __future__ import annotations

from pathlib import Path
from flask import Blueprint, jsonify, render_template, request, session

from .content_files import ContentError
from .content_tasks import ContentTasks
from .prompt_creator import DEFAULT_SYSTEM, configuration


def register(app):
    bp = Blueprint('content_tools', __name__)
    base = Path(app.template_folder).parent
    tasks = ContentTasks(base)
    app.extensions['omnitool_content_tasks'] = tasks
    previous_shutdown = app.extensions['omnitool_shutdown']
    def shutdown():
        tasks.shutdown()
        previous_shutdown()
    app.extensions['omnitool_shutdown'] = shutdown

    def owner():
        return session['sid']

    def body():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ContentError('Expected a JSON object')
        return data

    def start(action, data):
        return jsonify(task=tasks.submit(owner(), action, data)), 202

    @bp.errorhandler(ContentError)
    def invalid(error):
        return jsonify(error=str(error)), 400

    @bp.get('/content/prompts')
    def prompts():
        try:
            config, error = configuration(), ''
        except ContentError as exc:
            config, error = {}, str(exc)
        return render_template('prompt_creator.html', config=config, config_error=error, system=DEFAULT_SYSTEM)

    @bp.get('/content/lyrics')
    def lyrics_page():
        return render_template('lyrics_workbench.html')

    @bp.post('/api/content/prompts/status')
    def prompt_status():
        body()
        return start('prompt-status', {})

    @bp.post('/api/content/prompts/generate')
    def prompt_generate():
        data = body()
        # Validate before enqueueing, without contacting a model.
        if data.keys() - {'idea', 'system', 'max_tokens', 'temperature'}:
            raise ContentError('Unknown prompt option; backend URLs are configured locally, not by requests')
        return start('prompt-generate', data)

    @bp.post('/api/content/lyrics/scan')
    def lyrics_scan():
        data = body()
        root = data.get('root')
        if not isinstance(root, str) or not root.strip() or len(root) > 4096:
            raise ContentError('Enter a music folder path')
        return start('lyrics-scan', {'root': root})

    @bp.post('/api/content/lyrics/prepare')
    def lyrics_prepare():
        data = body()
        item = tasks.get(owner(), data.get('scan_task'))
        if item['state'] != 'succeeded' or item['action'] != 'lyrics-scan':
            raise ContentError('Complete a music scan first')
        if data.get('provider') == 'lrclib' and data.get('external_consent') is not True:
            raise ContentError('Explicitly allow sending selected artist/title/album/duration to LRCLIB')
        out = data.get('out')
        if not isinstance(out, str) or not out.strip() or len(out) > 4096:
            raise ContentError('Enter a new output folder path')
        if type(data.get('replace', False)) is not bool:
            raise ContentError('Invalid replacement option')
        return start('lyrics-prepare', {'scan': item['result'], 'selections': data.get('selections'), 'out': out,
                                        'provider': data.get('provider', 'sidecar'), 'replace': data.get('replace', False)})

    @bp.post('/api/content/lyrics/apply')
    def lyrics_apply():
        data = body()
        # Serialize confirmation, token consumption, and submission against duplicate clicks.
        with tasks.mutex:
            item = tasks.get(owner(), data.get('preview_task'))
            if item['action'] != 'lyrics-prepare' or item['state'] != 'succeeded':
                raise ContentError('Prepare and review a lyrics batch first')
            count = item['result']['count']
            if not count or data.get('confirmation') != f'WRITE {count}':
                raise ContentError(f'Type WRITE {count} to create the reviewed tagged copies')
            result = tasks.consume(owner(), data['preview_task'], 'lyrics-prepare')
            return start('lyrics-apply', {'preview': result})

    @bp.get('/api/content/tasks/<token>')
    def task(token):
        item = tasks.get(owner(), token)
        return jsonify(state=item['state'], action=item['action'], result=item['result'], error=item['error'])

    @bp.post('/api/content/tasks/<token>/stop')
    def stop(token):
        body()
        tasks.stop(owner(), token)
        return jsonify(ok=True)

    @bp.post('/api/content/clear')
    def clear():
        body()
        tasks.clear(owner())
        return jsonify(ok=True)

    @app.before_request
    def stop_on_signout():
        # The parent CSRF/auth hook runs first. Clear results and request cancellation.
        if request.endpoint == 'logout' and request.method == 'POST' and session.get('sid'):
            tasks.clear(session['sid'])

    app.register_blueprint(bp)
