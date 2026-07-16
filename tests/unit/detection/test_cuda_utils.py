"""Tests for dependency-free CUDA device parsing."""

import pytest

from signdata.processors.detection._cuda_utils import parse_cuda_device_index


def test_parse_cuda_device_index_rejects_malformed_selectors():
    assert parse_cuda_device_index("cuda") == 0
    assert parse_cuda_device_index("cuda:2") == 2

    for device in ("cuda:", "cudafoo", "cudafoo:1"):
        with pytest.raises(RuntimeError, match="Invalid CUDA device"):
            parse_cuda_device_index(device)
