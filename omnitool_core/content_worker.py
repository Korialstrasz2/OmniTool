"""Private protocol for the workspace worker; not a general command runner."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .content_files import ContentError
from .content_tasks import MAX_WIRE


def dispatch(request):
    action, value = request['action'], request['payload']
    if action.startswith('prompt-'):
        from . import prompt_creator
        if action == 'prompt-status':
            return prompt_creator.status()
        if action == 'prompt-generate':
            return prompt_creator.generate(value)
    from . import lyrics_workbench as lyrics
    if action == 'lyrics-scan':
        return lyrics.scan(Path(value['root']))
    if action == 'lyrics-prepare':
        return lyrics.prepare(value['scan'], value['selections'], Path(value['out']), value['provider'], value['replace'])
    if action == 'lyrics-apply':
        return lyrics.apply(value['preview'])
    raise ContentError('Unsupported content action')


def main():
    # OS resource limits supplement, but do not constitute, a native-code sandbox.
    if sys.platform.startswith('linux'):
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (1536 * 1024 * 1024, 1536 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    try:
        wire = sys.stdin.buffer.read(MAX_WIRE + 1)
        if len(wire) > MAX_WIRE:
            raise ContentError('Request too large')
        result = dispatch(json.loads(wire))
        output = json.dumps(result, ensure_ascii=True).encode('utf-8')
        if len(output) > MAX_WIRE:
            raise ContentError('Result too large; select fewer tracks')
        sys.stdout.buffer.write(output)
        return 0
    except Exception as exc:
        if isinstance(exc, ContentError):
            error = str(exc)
        elif isinstance(exc, ImportError):
            error = 'Optional dependency missing. Install requirements-content.txt in the workspace environment.'
        elif isinstance(exc, OSError):
            error = 'Local file operation failed. Check paths/permissions and any .partial output folder.'
        else:
            error = 'Content processing failed. Check input format; originals were not changed.'
        sys.stdout.write(json.dumps({'error': error}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
