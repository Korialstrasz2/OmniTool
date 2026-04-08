import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for

app = Flask(__name__)
app.config['SECRET_KEY'] = 'change-me'

BASE_DIR = Path(__file__).parent
TOOLS_FILE = BASE_DIR / 'tools.json'
SCRIPTS_DIR = BASE_DIR / 'scripts'
HUNYUAN_DEFAULT_REPO = SCRIPTS_DIR / 'hunyuan3d-2.1'
HUNYUAN_CONFIG = SCRIPTS_DIR / 'hunyuan3d_config.json'
HUNYUAN_REPO_URL = 'https://github.com/tencent-hunyuan/hunyuan3d-2.1.git'
LYRICS_CONFIG = SCRIPTS_DIR / 'lyrics_embedder_config.json'
MEDIA_HARVESTER_SCRIPT = SCRIPTS_DIR / 'media_harvester.py'
MEDIA_PASSIVE_CAPTURE_SCRIPT = SCRIPTS_DIR / 'media_passive_capture.py'
MEDIA_HARVESTER_DEFAULTS = SCRIPTS_DIR / 'media_harvester_defaults.json'

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional runtime dependency
    OpenAI = None  # type: ignore





def read_media_harvester_defaults():
    if not MEDIA_HARVESTER_DEFAULTS.exists():
        return {}
    try:
        data = json.loads(MEDIA_HARVESTER_DEFAULTS.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_media_harvester_defaults(values):
    MEDIA_HARVESTER_DEFAULTS.write_text(json.dumps(values, indent=2), encoding='utf-8')


def coerce_media_harvester_values(values):
    """Normalize UI settings so frontend choices stay coherent with backend flags."""
    mode = values.get('mode', 'preview')
    site_mode = values.get('site_mode', 'general')

    if mode == 'preview':
        values['include_page_with_ytdlp'] = ''
        values['download_all_resolutions'] = ''
    elif mode == 'download':
        values['download_all_resolutions'] = ''
    elif mode == 'download-all':
        values['include_page_with_ytdlp'] = '1'
        values['download_all_resolutions'] = '1'

    if site_mode == 'instagram-public':
        values['access_strategy'] = 'resilient'
        values['include_page_with_ytdlp'] = '1'
        try:
            workers = int(values.get('workers', '2') or '2')
        except ValueError:
            workers = 2
        values['workers'] = str(max(1, min(workers, 2)))

    if site_mode == 'all-media-html-sources':
        values['all_page'] = '1'

    return values

def load_tools():
    """Load tool definitions from tools.json."""
    if TOOLS_FILE.exists():
        with open(TOOLS_FILE) as f:
            return json.load(f)
    return []


def load_hunyuan_settings() -> Dict[str, str]:
    """Load persisted configuration for the Hunyuan3D manager."""
    defaults = {
        'repo_path': str(HUNYUAN_DEFAULT_REPO),
        'repo_url': HUNYUAN_REPO_URL,
        'env_overrides': '',
    }
    if HUNYUAN_CONFIG.exists():
        try:
            with open(HUNYUAN_CONFIG) as config_file:
                stored = json.load(config_file)
                defaults.update(stored)
        except json.JSONDecodeError:
            pass
    return defaults


def save_hunyuan_settings(settings: Dict[str, str]) -> None:
    """Persist configuration for the Hunyuan3D manager."""
    HUNYUAN_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with open(HUNYUAN_CONFIG, 'w') as config_file:
        json.dump(settings, config_file, indent=2)


def load_lyrics_settings() -> Dict[str, str]:
    defaults = {
        'root_path': r'D:\Music',
        'workers': '50',
        'max_inflight': '0',
        'min_delay': '0.0',
        'timeout': '20.0',
        'retries': '2',
        'backoff': '1.25',
        'status_every': '5.0',
        'cache_flush_every': '5000',
        'max_files': '0',
        'ai_rounds': '2',
        'ai_model': 'gpt-5-mini-2025-08-07',
        'ai_log_file': str(SCRIPTS_DIR / 'lyrics_ai_calls.log'),
        'exts': '.mp3,.flac,.m4a,.mp4,.aac,.ogg,.opus',
        'log_level': 'DEBUG',
        'log_file': str(SCRIPTS_DIR / 'lyrics_embedder.log'),
        'ai_if_failed': '1',
    }
    if LYRICS_CONFIG.exists():
        try:
            with open(LYRICS_CONFIG) as config_file:
                stored = json.load(config_file)
                defaults.update(stored)
        except json.JSONDecodeError:
            pass
    return defaults


def save_lyrics_settings(settings: Dict[str, str]) -> None:
    LYRICS_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with open(LYRICS_CONFIG, 'w') as config_file:
        json.dump(settings, config_file, indent=2)


def parse_env_lines(env_text: str) -> Dict[str, str]:
    """Parse KEY=VALUE pairs from user input."""
    env: Dict[str, str] = {}
    for line in env_text.splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith('#'):
            continue
        if '=' in cleaned:
            key, value = cleaned.split('=', 1)
            env[key.strip()] = value.strip()
    return env


def resolve_tool_command(command: str) -> str:
    """Resolve platform-specific launcher scripts for tools."""
    if not command:
        return command
    tokens = shlex.split(command, posix=os.name != 'nt')
    if not tokens:
        return command

    executable = tokens[0]
    executable_path = Path(executable)
    if os.name != 'nt' and executable.lower().endswith('.bat'):
        candidate = executable_path.with_suffix('.sh')
        if (SCRIPTS_DIR / candidate).exists():
            tokens[0] = str(candidate)
    if os.name == 'nt' and executable.lower().endswith('.sh'):
        candidate = executable_path.with_suffix('.bat')
        if (SCRIPTS_DIR / candidate).exists():
            tokens[0] = str(candidate)

    normalized = os.path.normpath(tokens[0])
    if not os.path.isabs(normalized):
        has_separator = any(sep in normalized for sep in (os.sep, '/', '\\'))
        if has_separator and not normalized.startswith(('.', os.sep)):
            normalized = f".{os.sep}{normalized}"
    tokens[0] = normalized

    if os.name == 'nt':
        return subprocess.list2cmdline(tokens)
    return shlex.join(tokens)


def run_checked_command(command: List[str], cwd: Path | None = None) -> Tuple[bool, str]:
    """Run a subprocess command and return (success, output)."""
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=True,
        )
        return True, (completed.stdout + completed.stderr).strip()
    except subprocess.CalledProcessError as exc:  # pragma: no cover - relies on system state
        output = (exc.stdout or '') + (exc.stderr or '')
        return False, output.strip()


def git_status(repo_path: Path) -> Dict[str, str | bool]:
    """Collect basic git status information for the repo if available."""
    status: Dict[str, str | bool] = {
        'exists': repo_path.exists(),
        'message': 'Repository not found.',
    }
    if not repo_path.exists():
        return status

    ok, revision = run_checked_command(['git', 'rev-parse', '--short', 'HEAD'], cwd=repo_path)
    if ok:
        status['revision'] = revision
    ok, branch = run_checked_command(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=repo_path)
    if ok:
        status['branch'] = branch
    ok, summary = run_checked_command(['git', 'status', '--short'], cwd=repo_path)
    if ok:
        status['dirty'] = bool(summary)
    status['message'] = 'Repository ready.' if ok else 'Git metadata unavailable.'
    return status


@app.route('/')
def index():
    tools = load_tools()
    return render_template('index.html', tools=tools)


@app.route('/tool/<tool_id>')
def tool_detail(tool_id):
    tools = {t['id']: t for t in load_tools()}
    tool = tools.get(tool_id)
    if not tool:
        flash('Tool not found.', 'error')
        return redirect(url_for('index'))
    if tool_id == 'hunyuan3d-manager':
        return redirect(url_for('hunyuan3d'))
    if tool_id == 'lyrics-embedder-manager':
        return redirect(url_for('lyrics_embedder'))
    if tool_id == 'csv-editor':
        return redirect(url_for('csv_editor'))
    if tool_id == 'media-harvester-manager':
        return redirect(url_for('media_harvester'))
    return render_template('tool.html', tool=tool)




@app.route('/media-harvester', methods=['GET', 'POST'])
def media_harvester():
    """Media discovery/downloader with preview and privacy controls."""
    output = ''
    defaults = read_media_harvester_defaults()
    values = {
        'url': '',
        'urls_text': '',
        'output_dir': str(defaults.get('output', SCRIPTS_DIR / 'media_downloads')),
        'proxy': str(defaults.get('proxy', '')),
        'timeout': str(defaults.get('timeout', '45')),
        'workers': str(defaults.get('workers', '6')),
        'mode': str(defaults.get('mode', 'preview')),
        'diagnostics': '1' if defaults.get('diagnostics', True) else '',
        'download_all_resolutions': '1' if defaults.get('download_all_resolutions', True) else '',
        'include_page_with_ytdlp': '1' if defaults.get('include_page_with_ytdlp', True) else '',
        'all_page': '1' if defaults.get('all_page', False) else '',
        'all_scroll': '1' if defaults.get('all_scroll', False) else '',
        'access_strategy': str(defaults.get('access_strategy', 'standard')),
        'site_mode': str(defaults.get('site_mode', 'general')),
        'passive_page_url': str(defaults.get('passive_page_url', '')),
        'passive_output_dir': str(defaults.get('passive_output_dir', SCRIPTS_DIR / 'media_downloads')),
        'passive_mode': str(defaults.get('passive_mode', 'slow')),
        'passive_capture_seconds': str(defaults.get('passive_capture_seconds', '90')),
        'passive_idle_seconds': str(defaults.get('passive_idle_seconds', '12')),
        'passive_headless': '1' if defaults.get('passive_headless', True) else '',
        'passive_auto_download': '1' if defaults.get('passive_auto_download', True) else '',
    }
    values = coerce_media_harvester_values(values)

    if request.method == 'POST':
        workflow = request.form.get('workflow', 'active').strip() or 'active'

        if workflow == 'passive':
            values.update({
                'passive_page_url': request.form.get('passive_page_url', values['passive_page_url']).strip(),
                'passive_output_dir': request.form.get('passive_output_dir', values['passive_output_dir']).strip(),
                'passive_mode': request.form.get('passive_mode', values['passive_mode']).strip() or 'slow',
                'passive_capture_seconds': request.form.get('passive_capture_seconds', values['passive_capture_seconds']).strip() or '90',
                'passive_idle_seconds': request.form.get('passive_idle_seconds', values['passive_idle_seconds']).strip() or '12',
                'passive_headless': '1' if request.form.get('passive_headless') else '',
                'passive_auto_download': '1' if request.form.get('passive_auto_download') else '',
            })

            if values['passive_mode'] not in {'slow', 'fast', 'elusive-sites'}:
                values['passive_mode'] = 'slow'

            command = [
                sys.executable,
                str(MEDIA_PASSIVE_CAPTURE_SCRIPT),
                '--output', values['passive_output_dir'],
                '--mode', values['passive_mode'],
                '--capture-seconds', values['passive_capture_seconds'],
                '--idle-seconds', values['passive_idle_seconds'],
            ]
            if values['passive_page_url']:
                command.extend(['--url', values['passive_page_url']])
            if values['passive_headless']:
                command.append('--headless')
            if values['passive_auto_download']:
                command.append('--auto-download')

            try:
                completed = subprocess.run(
                    command,
                    cwd=str(SCRIPTS_DIR),
                    capture_output=True,
                    text=True,
                    timeout=60 * 20,
                )
                output = (completed.stdout or '') + ('\n' + completed.stderr if completed.stderr else '')
                if completed.returncode == 0:
                    flash('Passive capture completed.', 'success')
                else:
                    flash('Passive capture finished with errors. Review run output below.', 'error')
            except subprocess.TimeoutExpired:
                output = 'Passive capture timed out after 20 minutes.'
                flash(output, 'error')
            except Exception as exc:  # pragma: no cover
                output = f'Failed to run passive capture: {exc}'
                flash(output, 'error')

            return render_template('media_harvester.html', values=values, output=output)

        values.update({
            'url': request.form.get('url', '').strip(),
            'urls_text': request.form.get('urls_text', '').strip(),
            'output_dir': request.form.get('output_dir', values['output_dir']).strip(),
            'proxy': request.form.get('proxy', '').strip(),
            'timeout': request.form.get('timeout', '45').strip() or '45',
            'workers': request.form.get('workers', '6').strip() or '6',
            'mode': request.form.get('mode', 'preview').strip() or 'preview',
            'diagnostics': '1' if request.form.get('diagnostics') else '',
            'download_all_resolutions': '1' if request.form.get('download_all_resolutions') else '',
            'include_page_with_ytdlp': '1' if request.form.get('include_page_with_ytdlp') else '',
            'all_page': '1' if request.form.get('all_page') else '',
            'all_scroll': '1' if request.form.get('all_scroll') else '',
            'access_strategy': request.form.get('access_strategy', values['access_strategy']).strip() or 'standard',
            'site_mode': request.form.get('site_mode', values['site_mode']).strip() or 'general',
        })

        if values['access_strategy'] not in {'standard', 'resilient'}:
            values['access_strategy'] = 'standard'
        if values['site_mode'] not in {'general', 'instagram-public', 'all-media-html-sources'}:
            values['site_mode'] = 'general'
        values = coerce_media_harvester_values(values)

        if request.form.get('save_defaults'):
            payload = {
                'output': values['output_dir'],
                'proxy': values['proxy'],
                'timeout': values['timeout'],
                'workers': values['workers'],
                'mode': values['mode'],
                'diagnostics': bool(values['diagnostics']),
                'download_all_resolutions': bool(values['download_all_resolutions']),
                'include_page_with_ytdlp': bool(values['include_page_with_ytdlp']),
                'all_page': bool(values['all_page']),
                'all_scroll': bool(values['all_scroll']),
                'access_strategy': values['access_strategy'],
                'site_mode': values['site_mode'],
                'passive_page_url': values['passive_page_url'],
                'passive_output_dir': values['passive_output_dir'],
                'passive_mode': values['passive_mode'],
                'passive_capture_seconds': values['passive_capture_seconds'],
                'passive_idle_seconds': values['passive_idle_seconds'],
                'passive_headless': bool(values['passive_headless']),
                'passive_auto_download': bool(values['passive_auto_download']),
            }
            try:
                write_media_harvester_defaults(payload)
                flash('Saved media harvester defaults.', 'success')
            except Exception as exc:
                flash(f'Failed to save defaults: {exc}', 'error')
            return render_template('media_harvester.html', values=values, output=output)

        raw_urls = values['urls_text'] or values['url']
        urls = [line.strip() for line in raw_urls.splitlines() if line.strip()]
        if not urls:
            flash('Please enter at least one target URL.', 'error')
        else:
            values['url'] = urls[0]
            values['urls_text'] = '\n'.join(urls)
            all_successful = True
            output_chunks = []
            for index, target_url in enumerate(urls, start=1):
                command = [
                    sys.executable,
                    str(MEDIA_HARVESTER_SCRIPT),
                    target_url,
                    '--output', values['output_dir'],
                    '--timeout', values['timeout'],
                    '--workers', values['workers'],
                ]
                if values['proxy']:
                    command.extend(['--proxy', values['proxy']])
                if values['mode'] == 'preview':
                    command.append('--preview-only')
                if values['mode'] == 'download':
                    command.append('--include-page-with-ytdlp')
                if values['mode'] == 'download-all':
                    command.extend(['--include-page-with-ytdlp', '--download-all-resolutions'])
                if values['diagnostics']:
                    command.append('--diagnostics')
                if values['download_all_resolutions'] and '--download-all-resolutions' not in command:
                    command.append('--download-all-resolutions')
                if values['include_page_with_ytdlp'] and '--include-page-with-ytdlp' not in command:
                    command.append('--include-page-with-ytdlp')
                if values['all_page']:
                    command.append('--all-page')
                if values['all_scroll']:
                    command.append('--all-scroll')
                command.extend(['--access-strategy', values['access_strategy']])
                command.extend(['--site-mode', values['site_mode']])

                try:
                    completed = subprocess.run(
                        command,
                        cwd=str(SCRIPTS_DIR),
                        capture_output=True,
                        text=True,
                        timeout=60 * 30,
                    )
                    command_output = (completed.stdout or '') + ('\n' + completed.stderr if completed.stderr else '')
                    output_chunks.append(f"=== [{index}/{len(urls)}] {target_url} ===\n{command_output}".strip())
                    if completed.returncode != 0:
                        all_successful = False
                except subprocess.TimeoutExpired:
                    all_successful = False
                    output_chunks.append(f"=== [{index}/{len(urls)}] {target_url} ===\nTimed out after 30 minutes.")
                except Exception as exc:  # pragma: no cover
                    all_successful = False
                    output_chunks.append(f"=== [{index}/{len(urls)}] {target_url} ===\nFailed to run Media Harvester: {exc}")

            output = '\n\n'.join(output_chunks)
            if all_successful:
                flash(f'Media Harvester completed for {len(urls)} URL(s).', 'success')
            else:
                flash('Media Harvester finished with one or more errors. Review run output below.', 'error')

    return render_template('media_harvester.html', values=values, output=output)



@app.post('/media-harvester/install-dependencies')
def media_harvester_install_dependencies():
    command = [sys.executable, str(MEDIA_HARVESTER_SCRIPT), '--install-dependencies', '--list-tools']
    try:
        completed = subprocess.run(command, cwd=str(SCRIPTS_DIR), capture_output=True, text=True, timeout=60 * 10)
        output = (completed.stdout or '') + ('\n' + completed.stderr if completed.stderr else '')
        flash(output or 'Dependency installation command completed.', 'success' if completed.returncode == 0 else 'error')
    except Exception as exc:  # pragma: no cover
        flash(f'Failed to install dependencies: {exc}', 'error')
    return redirect(url_for('media_harvester'))


@app.post('/media-harvester/check-tools')
def media_harvester_check_tools():
    command = [sys.executable, str(MEDIA_HARVESTER_SCRIPT), '--list-tools']
    try:
        completed = subprocess.run(command, cwd=str(SCRIPTS_DIR), capture_output=True, text=True, timeout=120)
        output = (completed.stdout or '') + ('\n' + completed.stderr if completed.stderr else '')
        flash(output or 'Tool check completed.', 'success' if completed.returncode == 0 else 'error')
    except Exception as exc:  # pragma: no cover
        flash(f'Failed to run tool check: {exc}', 'error')
    return redirect(url_for('media_harvester'))


@app.route('/csv-editor')
def csv_editor():
    """CSV import/export formatter with delimiter controls and previews."""
    return render_template('csv_editor.html')


@app.route('/tool/<tool_id>/run', methods=['POST'])
def run_tool(tool_id):
    tools = {t['id']: t for t in load_tools()}
    tool = tools.get(tool_id)
    if not tool:
        flash('Tool not found.', 'error')
        return redirect(url_for('index'))
    command = resolve_tool_command(tool['command'])
    try:
        subprocess.Popen(command, shell=True, cwd=str(SCRIPTS_DIR))
        flash(f"Started {tool['name']}", 'success')
    except Exception as exc:  # pragma: no cover
        flash(f"Error starting {tool['name']}: {exc}", 'error')
    return redirect(url_for('tool_detail', tool_id=tool_id))


@app.route('/hunyuan3d')
def hunyuan3d():
    """Dashboard for managing the Hunyuan3D 2.1 toolchain."""
    settings = load_hunyuan_settings()
    repo_path = Path(settings['repo_path']).expanduser()
    status = git_status(repo_path)
    requirements_path = repo_path / 'requirements.txt'
    setup_exists = (repo_path / 'setup.py').exists() or (repo_path / 'pyproject.toml').exists()
    return render_template(
        'hunyuan3d.html',
        settings=settings,
        repo_status=status,
        requirements_exists=requirements_path.exists(),
        setup_exists=setup_exists,
    )


@app.post('/hunyuan3d/save-settings')
def hunyuan3d_save_settings():
    repo_path = request.form.get('repo_path', str(HUNYUAN_DEFAULT_REPO)).strip()
    repo_url = request.form.get('repo_url', HUNYUAN_REPO_URL).strip() or HUNYUAN_REPO_URL
    env_overrides = request.form.get('env_overrides', '').strip()
    settings = {
        'repo_path': repo_path,
        'repo_url': repo_url,
        'env_overrides': env_overrides,
    }
    save_hunyuan_settings(settings)
    flash('Hunyuan3D settings saved.', 'success')
    return redirect(url_for('hunyuan3d'))


@app.post('/hunyuan3d/sync')
def hunyuan3d_sync_repo():
    settings = load_hunyuan_settings()
    repo_path = Path(request.form.get('repo_path', settings['repo_path'])).expanduser()
    repo_url = request.form.get('repo_url', settings['repo_url']).strip() or HUNYUAN_REPO_URL
    repo_path.parent.mkdir(parents=True, exist_ok=True)

    if repo_path.exists():
        ok, output = run_checked_command(['git', 'pull'], cwd=repo_path)
        action = 'Updated existing repository.'
    else:
        ok, output = run_checked_command(['git', 'clone', repo_url, str(repo_path)])
        action = 'Cloned repository.'

    flash(f"{action if ok else 'Unable to sync repository.'} {output}", 'success' if ok else 'error')
    return redirect(url_for('hunyuan3d'))


@app.post('/hunyuan3d/install-deps')
def hunyuan3d_install_deps():
    settings = load_hunyuan_settings()
    repo_path = Path(request.form.get('repo_path', settings['repo_path'])).expanduser()
    extras = [pkg.strip() for pkg in request.form.get('extra_packages', '').splitlines() if pkg.strip()]
    commands = []
    requirements_file = repo_path / 'requirements.txt'
    if requirements_file.exists():
        commands.append([sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)])
    setup_exists = (repo_path / 'setup.py').exists() or (repo_path / 'pyproject.toml').exists()
    if setup_exists:
        commands.append([sys.executable, '-m', 'pip', 'install', '-e', str(repo_path)])
    for pkg in extras:
        commands.append([sys.executable, '-m', 'pip', 'install', pkg])

    if not commands:
        flash('No installable targets found. Ensure the repository is cloned first.', 'error')
        return redirect(url_for('hunyuan3d'))

    log: List[str] = []
    overall_success = True
    for cmd in commands:
        ok, output = run_checked_command(cmd, cwd=repo_path if repo_path.exists() else None)
        overall_success = overall_success and ok
        display_cmd = shlex.join(cmd)
        log.append(f"$ {display_cmd}\n{output or 'Done.'}")

    flash_message = '\n\n'.join(log)
    flash(flash_message, 'success' if overall_success else 'error')
    return redirect(url_for('hunyuan3d'))


@app.post('/hunyuan3d/run')
def hunyuan3d_run():
    settings = load_hunyuan_settings()
    repo_path = Path(request.form.get('repo_path', settings['repo_path'])).expanduser()
    if not repo_path.exists():
        flash('Repository path does not exist. Clone it first.', 'error')
        return redirect(url_for('hunyuan3d'))

    entry_script = request.form.get('entry_script', 'app.py').strip() or 'app.py'
    mode = request.form.get('mode', 'demo')
    prompt = request.form.get('prompt', '').strip()
    negative_prompt = request.form.get('negative_prompt', '').strip()
    steps = request.form.get('steps', '').strip()
    seed = request.form.get('seed', '').strip()
    cfg_scale = request.form.get('cfg_scale', '').strip()
    output_dir = request.form.get('output_dir', '').strip()
    checkpoint = request.form.get('checkpoint', '').strip()
    config_path = request.form.get('config_path', '').strip()
    camera_path = request.form.get('camera_path', '').strip()
    precision = request.form.get('precision', '').strip()
    device = request.form.get('device', '').strip()
    extra_args = request.form.get('extra_args', '').strip()
    env_overrides = request.form.get('env_overrides', settings['env_overrides'])
    if not env_overrides.strip():
        env_overrides = settings['env_overrides']
    flags = request.form.getlist('flags')

    command: List[str] = [sys.executable, entry_script]

    def append_arg(flag: str, value: str) -> None:
        if value:
            command.extend([flag, value])

    if mode == 'demo':
        if 'share' in flags:
            command.append('--share')
    else:
        append_arg('--prompt', prompt)
        append_arg('--negative_prompt', negative_prompt)
        append_arg('--steps', steps)
        append_arg('--seed', seed)
        append_arg('--cfg-scale', cfg_scale)
        append_arg('--output-dir', output_dir)
        append_arg('--checkpoint', checkpoint)
        append_arg('--config', config_path)
        append_arg('--camera-path', camera_path)
        append_arg('--precision', precision)
        append_arg('--device', device)
        if 'use_xformers' in flags:
            command.append('--use-xformers')
        if 'enable_fp16' in flags:
            command.append('--fp16')

    if extra_args:
        command.extend(shlex.split(extra_args))

    env = os.environ.copy()
    env.update(parse_env_lines(env_overrides))

    try:
        subprocess.Popen(command, cwd=str(repo_path), env=env)
        flash(f"Started command: {shlex.join(command)}", 'success')
    except FileNotFoundError as exc:  # pragma: no cover - depends on filesystem
        flash(f"Unable to start command: {exc}", 'error')
    return redirect(url_for('hunyuan3d'))


@app.get('/lyrics-embedder')
def lyrics_embedder():
    settings = load_lyrics_settings()
    return render_template('lyrics_embedder.html', settings=settings)


@app.post('/lyrics-embedder/save-settings')
def lyrics_embedder_save_settings():
    fields = [
        'root_path', 'workers', 'max_inflight', 'min_delay', 'timeout', 'retries',
        'backoff', 'status_every', 'cache_flush_every', 'max_files', 'ai_rounds',
        'ai_model', 'ai_log_file', 'exts', 'log_level', 'log_file', 'ai_if_failed',
    ]
    settings = {field: request.form.get(field, '').strip() for field in fields}
    settings['ai_if_failed'] = '1' if request.form.get('ai_if_failed') else '0'
    save_lyrics_settings(settings)
    flash('Lyrics embedder settings saved.', 'success')
    return redirect(url_for('lyrics_embedder'))


@app.post('/lyrics-embedder/run')
def lyrics_embedder_run():
    settings = load_lyrics_settings()
    root_path = request.form.get('root_path', settings['root_path']).strip() or settings['root_path']
    command: List[str] = [sys.executable, 'lyrics_embedder.py', root_path]

    def append_arg(flag: str, value: str) -> None:
        cleaned = value.strip()
        if cleaned:
            command.extend([flag, cleaned])

    append_arg('--workers', request.form.get('workers', settings['workers']))
    append_arg('--max-inflight', request.form.get('max_inflight', settings['max_inflight']))
    append_arg('--min-delay', request.form.get('min_delay', settings['min_delay']))
    append_arg('--timeout', request.form.get('timeout', settings['timeout']))
    append_arg('--retries', request.form.get('retries', settings['retries']))
    append_arg('--backoff', request.form.get('backoff', settings['backoff']))
    append_arg('--status-every', request.form.get('status_every', settings['status_every']))
    append_arg('--cache-flush-every', request.form.get('cache_flush_every', settings['cache_flush_every']))
    append_arg('--max-files', request.form.get('max_files', settings['max_files']))
    append_arg('--ai-rounds', request.form.get('ai_rounds', settings['ai_rounds']))
    append_arg('--ai-model', request.form.get('ai_model', settings['ai_model']))
    append_arg('--ai-log-file', request.form.get('ai_log_file', settings.get('ai_log_file', '')))
    append_arg('--exts', request.form.get('exts', settings['exts']))
    append_arg('--log-level', request.form.get('log_level', settings['log_level']))
    append_arg('--log-file', request.form.get('log_file', settings['log_file']))

    if request.form.get('force'):
        command.append('--force')
    if request.form.get('keep_synced'):
        command.append('--keep-synced')
    ai_if_failed = request.form.get('ai_if_failed', settings.get('ai_if_failed', '1'))
    if ai_if_failed and ai_if_failed != '0':
        command.append('--ai-if-failed')

    try:
        subprocess.Popen(command, cwd=str(SCRIPTS_DIR), env=os.environ.copy())
        flash(f"Started lyrics embedding run: {shlex.join(command)}", 'success')
    except Exception as exc:  # pragma: no cover - environment specific
        flash(f"Unable to start lyrics embedder: {exc}", 'error')
    return redirect(url_for('lyrics_embedder'))


@app.post('/lyrics-embedder/ai-ping')
def lyrics_embedder_ai_ping():
    model = request.form.get('ai_model', '').strip()
    if not model and request.is_json:
        payload = request.get_json(silent=True) or {}
        model = str(payload.get('ai_model', '')).strip()
    if not model:
        model = (load_lyrics_settings().get('ai_model') or 'gpt-5-mini-2025-08-07').strip()

    if OpenAI is None:
        return jsonify({'ok': False, 'error': 'openai package is not installed in this environment.'}), 500

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return jsonify({'ok': False, 'error': 'OPENAI_API_KEY is not set.'}), 400

    try:
        client = OpenAI(api_key=api_key, timeout=20.0)
        response = client.responses.create(
            model=model,
            input='Share one short random interesting fact in a single sentence.',
        )
        text = (getattr(response, 'output_text', None) or '').strip()
        if not text:
            return jsonify({
                'ok': False,
                'error': 'Ping succeeded but returned empty output_text.',
                'raw_response': str(response),
            }), 502
        return jsonify({'ok': True, 'model': model, 'response': text})
    except Exception as exc:  # pragma: no cover - depends on runtime service/config
        return jsonify({'ok': False, 'error': f'{type(exc).__name__}: {exc}', 'model': model}), 502


@app.get('/lyrics-embedder/logs')
def lyrics_embedder_logs():
    settings = load_lyrics_settings()
    configured = request.args.get('log_file', settings.get('log_file', '')).strip()
    level_filter = request.args.get('level', '').strip().upper()
    query = request.args.get('q', '').strip().lower()
    raw_max_lines = request.args.get('max_lines', '300').strip()
    try:
        max_lines = int(raw_max_lines)
    except ValueError:
        max_lines = 300
    max_lines = min(max(max_lines, 1), 2000)
    log_path = Path(configured).expanduser() if configured else SCRIPTS_DIR / 'lyrics_embedder.log'

    if not log_path.exists():
        return {'ok': False, 'log_path': str(log_path), 'lines': [], 'message': 'Log file does not exist yet.'}

    with open(log_path, encoding='utf-8', errors='replace') as log_file:
        lines = log_file.readlines()[-max_lines:]

    if level_filter:
        lines = [line for line in lines if f' {level_filter} ' in line or line.startswith(level_filter)]
    if query:
        lines = [line for line in lines if query in line.lower()]

    return {'ok': True, 'log_path': str(log_path), 'lines': [line.rstrip('\n') for line in lines]}


if __name__ == '__main__':
    app.run(debug=True)
