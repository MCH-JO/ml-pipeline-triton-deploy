import argparse
import numpy as np
import tritonclient.http as httpclient


def main():
    parser = argparse.ArgumentParser(
        description="Triton inference Client (HTTP) for 'simple' model (ONNX)."
    )
    parser.add_argument("--url", default="localhost:8000", help="Triton HTTP URL, ex: localhost:8000")
    parser.add_argument("--model", default="simple", help="Model name in Triton")
    parser.add_argument("--version", default="", help="Model version (empty = default)")
    parser.add_argument(
        "--input-name", default="input", help="Input name (must match config.pbtxt)"
    )
    parser.add_argument(
        "--output-name", default="output", help="Output name (must match config.pbtxt)"
    )
    args = parser.parse_args()

    # ------------------------
    # 1) Triton server connection
    # ------------------------
    client = httpclient.InferenceServerClient(url=args.url, verbose=False)

    # ------------------------
    # 2) Checks
    # ------------------------
    if not client.is_server_live():
        raise RuntimeError("Triton's server si not 'live'. Check docker-compose / ports.")
    if not client.is_server_ready():
        raise RuntimeError("Triton's server si not 'ready'.")

    # ------------------------
    # 3) Prepare version as a string (empty = default)
    # ------------------------
    model_version = args.version  # "" = défaut
    if not client.is_model_ready(args.model, model_version):
        raise RuntimeError(f"Model '{args.model}' is note ready (name/version ?)")

    # ------------------------
    # 4) Prepare entry that complies with config.pbtxt:
    # input: FP32 dims [1,4] ; output: FP32 dims [1,2]
    # ------------------------
    x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)  # shape (1,4)

    inp = httpclient.InferInput(args.input_name, x.shape, "FP32")
    inp.set_data_from_numpy(x)

    requested_outputs = [httpclient.InferRequestedOutput(args.output_name)]

    # ------------------------
    # 5) Inference call
    # ------------------------
    resp = client.infer(
        model_name=args.model,
        model_version=model_version,
        inputs=[inp],
        outputs=requested_outputs,
    )

    y = resp.as_numpy(args.output_name)

    print("Triton inference OK")
    print("Input :", x, x.shape, x.dtype)
    print("Output:", y, y.shape, y.dtype)

if __name__ == "__main__":
    main()
