"""One explicitly configured local model endpoint; no probing or fallback retries."""
from __future__ import annotations

import http.client
import ipaddress
import json
import math
import os
import socket
import time
from urllib.parse import urlsplit

from .content_files import ContentError

MAX_RESPONSE = 512 * 1024
DEFAULT_SYSTEM = ('Expand the user\'s idea into a clear text-to-image prompt. Describe the subject, '
                  'composition, setting, lighting, and style. Return only the finished prompt.')


def text(value, label: str, maximum: int, required: bool = True) -> str:
    if not isinstance(value, str) or len(value.encode('utf-8')) > maximum or '\0' in value:
        raise ContentError(f'Invalid {label}')
    value = value.strip()
    if required and not value:
        raise ContentError(f'{label} is required')
    return value


def configuration(env=None) -> dict:
    env = os.environ if env is None else env
    raw = env.get('OMNITOOL_PROMPT_URL') or env.get('KOBOLD_HOST') or 'http://127.0.0.1:5001'
    try:
        parsed = urlsplit(raw)
        host = parsed.hostname
        if host == 'localhost':
            host = '127.0.0.1'  # Avoid DNS and proxy routing entirely.
        address = ipaddress.ip_address(host)
        port = parsed.port
        if (parsed.scheme != 'http' or not address.is_loopback or port is None or
                not 1024 <= port <= 65535 or parsed.path not in {'', '/'} or
                parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment):
            raise ValueError()
    except (ValueError, TypeError):
        raise ContentError('Set OMNITOOL_PROMPT_URL to an HTTP loopback origin with an explicit port, for example http://127.0.0.1:5001') from None
    kind = env.get('OMNITOOL_PROMPT_KIND', 'kobold')
    if kind not in {'kobold', 'local-chat'}:
        raise ContentError('OMNITOOL_PROMPT_KIND must be kobold or local-chat')
    model = text(env.get('OMNITOOL_PROMPT_MODEL', ''), 'configured model', 256, False)
    return {'host': str(address), 'port': port, 'kind': kind, 'model': model,
            'url': f'http://[{address}]:{port}' if address.version == 6 else f'http://{address}:{port}'}


def request_json(config: dict, path: str, payload=None, timeout: float = 90) -> dict:
    """Direct connection: no environmental proxy, redirect, or credential forwarding."""
    deadline = time.monotonic() + timeout
    conn = http.client.HTTPConnection(config['host'], config['port'], timeout=min(timeout, 5))
    try:
        body = None if payload is None else json.dumps(payload).encode('utf-8')
        conn.request('GET' if body is None else 'POST', path, body=body,
                     headers={'Content-Type': 'application/json', 'Accept': 'application/json'})
        if conn.sock:
            conn.sock.settimeout(max(.1, deadline - time.monotonic()))
        response = conn.getresponse()
        if response.status != 200:
            raise ContentError(f'Local model returned HTTP {response.status}; no alternate endpoint was called')
        if response.getheader('Content-Encoding', 'identity') != 'identity':
            raise ContentError('Compressed local-model responses are not supported')
        data = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ContentError('Local model response timed out')
            # The enclosing worker also imposes an independent wall-clock limit.
            chunk = response.read1(min(8192, MAX_RESPONSE + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > MAX_RESPONSE:
                raise ContentError('Local model response is too large')
        decoded = json.loads(data)
        if not isinstance(decoded, dict):
            raise ContentError('Local model returned an unexpected response')
        return decoded
    except (OSError, socket.timeout, http.client.HTTPException):
        raise ContentError('Cannot reach the configured local model, or its response timed out') from None
    except (ValueError, UnicodeError) as exc:
        if isinstance(exc, ContentError):
            raise
        raise ContentError('Local model did not return valid JSON') from None
    finally:
        conn.close()


def status() -> dict:
    config = configuration()
    data = request_json(config, '/api/v1/model' if config['kind'] == 'kobold' else '/v1/models', timeout=10)
    if config['kind'] == 'kobold':
        model = text(data.get('result', data.get('model', '')), 'model response', 256)
    else:
        records = data.get('data', [])
        if not isinstance(records, list):
            raise ContentError('Unexpected model-list response')
        names = [x['id'] for x in records[:100] if isinstance(x, dict) and isinstance(x.get('id'), str)]
        model = config['model'] or (names[0] if names else '')
        model = text(model, 'loaded model', 256)
    return {'online': True, 'model': model, 'kind': config['kind'], 'url': config['url']}


def generate(values: dict) -> dict:
    if not isinstance(values, dict) or values.keys() - {'idea', 'system', 'max_tokens', 'temperature'}:
        raise ContentError('Unknown prompt settings; configure the backend in the local environment')
    config = configuration()
    idea = text(values.get('idea'), 'idea', 12000)
    system = text(values.get('system', DEFAULT_SYSTEM), 'system instruction', 12000)
    tokens = values.get('max_tokens', 384)
    temperature = values.get('temperature', .7)
    if type(tokens) is not int or not 16 <= tokens <= 2048:
        raise ContentError('Output tokens must be an integer between 16 and 2048')
    if type(temperature) not in {int, float} or not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ContentError('Temperature must be between 0 and 2')
    if config['kind'] == 'kobold':
        payload = {'prompt': f'System:\n{system}\n\nUser:\n{idea}\n\nAssistant:\n',
                   'max_length': tokens, 'temperature': temperature}
        data = request_json(config, '/api/v1/generate', payload)
        results = data.get('results')
        if not isinstance(results, list) or not results or not isinstance(results[0], dict):
            raise ContentError('Unexpected KoboldCpp generation response')
        content = results[0].get('text')
    else:
        if not config['model']:
            raise ContentError('Set OMNITOOL_PROMPT_MODEL for the local-chat backend')
        data = request_json(config, '/v1/chat/completions', {
            'model': config['model'], 'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': idea}],
            'temperature': temperature, 'max_tokens': tokens, 'stream': False})
        try:
            content = data['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError):
            raise ContentError('Unexpected local chat-completion response') from None
    return {'content': text(content, 'generated text', 64000), 'kind': config['kind'], 'url': config['url']}
