import numpy as np
import tritonclient.http as httpclient


def make_client(url: str) -> httpclient.InferenceServerClient:
    return httpclient.InferenceServerClient(url=url, verbose=False)


def infer(url: str, model_name: str, inp_name: str, out_name: str, data: np.ndarray, model_version: str = ""):
    client = make_client(url)

    if not client.is_server_live():
        raise RuntimeError("Triton server not live")
    if not client.is_server_ready():
        raise RuntimeError("Triton server not ready")
    if not client.is_model_ready(model_name, model_version):
        raise RuntimeError(f"Model not ready: {model_name} (version='{model_version}')")

    triton_dtype = httpclient.np_to_triton_dtype(data.dtype)

    inp = httpclient.InferInput(inp_name, data.shape, triton_dtype)
    inp.set_data_from_numpy(data)

    out = httpclient.InferRequestedOutput(out_name)

    res = client.infer(model_name=model_name, model_version=model_version, inputs=[inp], outputs=[out])
    return res.as_numpy(out_name)
