"""The Python tool protocol uses UTF-8, including redirected Windows pipes."""
from __future__ import annotations

import codecs
import os
from collections.abc import Iterator, Mapping
from typing import BinaryIO


def python_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if source is None else source)
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    env.pop('OMNITOOL_ACCESS_TOKEN', None)
    return env


def text_chunks(stream: BinaryIO, size: int = 4096) -> Iterator[str]:
    """Preserve multibyte characters split across pipe reads; replace invalid input."""
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    while chunk := stream.read1(size):
        text = decoder.decode(chunk)
        if text:
            yield text
    final = decoder.decode(b'', final=True)
    if final:
        yield final
