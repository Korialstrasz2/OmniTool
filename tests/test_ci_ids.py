"""Large cases retain their data without unbounded diagnostic identifiers."""
import os
import pytest


@pytest.mark.parametrize('payload', [b'x' * (1024 * 1024 + 1), 'x' * 65536])
def test_large_payload_keeps_short_runtime_id(payload, request):
    assert len(payload) >= 65536
    assert len(request.node.nodeid) < 256
    assert len(os.environ['PYTEST_CURRENT_TEST']) < 300
