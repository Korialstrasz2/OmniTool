import json
import subprocess
from pathlib import Path

from flask import Flask, flash, redirect, render_template, url_for

app = Flask(__name__)
app.config['SECRET_KEY'] = 'change-me'

BASE_DIR = Path(__file__).parent
TOOLS_FILE = BASE_DIR / 'tools.json'
SCRIPTS_DIR = BASE_DIR / 'scripts'


def load_tools():
    """Load tool definitions from tools.json."""
    if TOOLS_FILE.exists():
        with open(TOOLS_FILE) as f:
            return json.load(f)
    return []


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
    return render_template('tool.html', tool=tool)


@app.route('/tool/<tool_id>/run', methods=['POST'])
def run_tool(tool_id):
    tools = {t['id']: t for t in load_tools()}
    tool = tools.get(tool_id)
    if not tool:
        flash('Tool not found.', 'error')
        return redirect(url_for('index'))
    command = tool['command']
    try:
        subprocess.Popen(command, shell=True, cwd=str(SCRIPTS_DIR))
        flash(f"Started {tool['name']}", 'success')
    except Exception as exc:  # pragma: no cover
        flash(f"Error starting {tool['name']}: {exc}", 'error')
    return redirect(url_for('tool_detail', tool_id=tool_id))


if __name__ == '__main__':
    app.run(debug=True)
