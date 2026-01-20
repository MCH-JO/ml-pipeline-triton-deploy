import os
import numpy as np
import pytest
from client.triton_http import infer

URL = os.getenv("TRITON_URL", "localhost:8000")
MODEL = os.getenv("TRITON_MODEL", "simple")
INP = os.getenv("TRITON_INPUT", "input")
OUT = os.getenv("TRITON_OUTPUT", "output")


def test_bad_dtype_rejected():
    x = np.array([[1, 2, 3, 4]], dtype=np.int64)  # (1,4) but bad dtype

    with pytest.raises(Exception) as e:
        infer(URL, MODEL, INP, OUT, x)

    msg = str(e.value).lower()
    assert ("dtype" in msg) or ("type" in msg) or ("invalid" in msg) or ("fp32" in msg)
