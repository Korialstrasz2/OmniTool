import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from omnitool_core.python_io import python_environment, text_chunks


class Pipe:
    def __init__(self, chunks): self.chunks = iter(chunks)
    def read1(self, size): return next(self.chunks, b'')


@pytest.mark.parametrize('split', range(1, 14))
def test_multibyte_output_can_split_anywhere(split):
    data = 'Café 漢字 😀'.encode('utf-8')
    assert ''.join(text_chunks(Pipe([data[:split], data[split:]]))) == 'Café 漢字 😀'


def test_invalid_output_is_replaced_without_losing_valid_text():
    assert ''.join(text_chunks(Pipe([b'ok\xff\xe2', b'']))) == 'ok\ufffd\ufffd'


def test_environment_does_not_leak_workspace_access_token():
    original = {'PYTHONIOENCODING': 'ascii', 'PYTHONUTF8': '0', 'OMNITOOL_ACCESS_TOKEN': 'test-only', 'PATH': 'path'}
    changed = python_environment(original)
    assert changed['PYTHONIOENCODING'] == 'utf-8' and changed['PYTHONUTF8'] == '1'
    assert 'OMNITOOL_ACCESS_TOKEN' not in changed and changed['PATH'] == 'path'
    assert original['OMNITOOL_ACCESS_TOKEN'] == 'test-only'


def test_native_python_pipe_with_unicode_working_directory(tmp_path):
    directory = tmp_path / 'Output à 漢字'; directory.mkdir()
    env = python_environment(os.environ | {'PYTHONIOENCODING': 'ascii', 'PYTHONUTF8': '0'})
    result = subprocess.run([sys.executable, '-c', "print('Caf\\u00e9 \\u6f22\\u5b57')"],
                            cwd=directory, env=env, capture_output=True, timeout=10)
    assert result.returncode == 0
    assert result.stdout.decode('utf-8').strip() == 'Café 漢字'
