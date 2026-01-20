import os
import numpy as np
from client.triton_http import infer

URL = os.getenv("TRITON_URL", "localhost:8000")
MODEL = os.getenv("TRITON_MODEL", "simple")
INP = os.getenv("TRITON_INPUT", "input")
OUT = os.getenv("TRITON_OUTPUT", "output")


def test_happy_path():
    x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)  # (1,4)
    y = infer(URL, MODEL, INP, OUT, x)

    assert y is not None
    assert isinstance(y, np.ndarray)
    assert y.shape == (1, 2)
    assert y.dtype == np.float32
