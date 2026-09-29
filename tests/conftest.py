"""Keep payloads out of test IDs, CI logs and PYTEST_CURRENT_TEST on Windows."""
import hashlib


def pytest_make_parametrize_id(config, val, argname):
    if isinstance(val, (str, bytes)) and len(val) > 128:
        data = val.encode('utf-8', errors='backslashreplace') if isinstance(val, str) else val
        return f'{argname}-{type(val).__name__}-{len(val)}-{hashlib.sha256(data).hexdigest()[:12]}'
    return None
