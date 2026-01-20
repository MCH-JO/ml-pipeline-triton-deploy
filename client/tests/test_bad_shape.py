import os
import numpy as np
import pytest
from client.triton_http import infer

URL = os.getenv("TRITON_URL", "localhost:8000")
MODEL = os.getenv("TRITON_MODEL", "simple")
INP = os.getenv("TRITON_INPUT", "input")
OUT = os.getenv("TRITON_OUTPUT", "output")


def test_bad_shape_rejected():
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # (1,3) instead of (1,4)

    with pytest.raises(Exception) as e:
        infer(URL, MODEL, INP, OUT, x)

    msg = str(e.value).lower()
    assert ("shape" in msg) or ("dimension" in msg) or ("invalid" in msg) or ("mismatch" in msg)
