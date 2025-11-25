import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, in_dim=4, out_dim=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)

def export_to_onnx(model: nn.Module, onnx_path: str):
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 4, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
    )

    print(f"ONNX model exported to {onnx_path}")

if __name__ == "__main__":
    model = SimpleModel()
    export_to_onnx(model, "simple.onnx")
