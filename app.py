import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from flask import Flask, flash, redirect, render_template, request, url_for

app = Flask(__name__)
app.config['SECRET_KEY'] = 'change-me'

BASE_DIR = Path(__file__).parent
TOOLS_FILE = BASE_DIR / 'tools.json'
SCRIPTS_DIR = BASE_DIR / 'scripts'
HUNYUAN_DEFAULT_REPO = SCRIPTS_DIR / 'hunyuan3d-2.1'
HUNYUAN_CONFIG = SCRIPTS_DIR / 'hunyuan3d_config.json'
HUNYUAN_REPO_URL = 'https://github.com/tencent-hunyuan/hunyuan3d-2.1.git'


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
    return render_template('tool.html', tool=tool)


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


if __name__ == '__main__':
    app.run(debug=True)
