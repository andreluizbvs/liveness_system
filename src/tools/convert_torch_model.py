import torch

from src.models.AENet import AENet

base_ckpt_path = "../../ckpt/"

# Load the PyTorch model architecture
model = AENet()

# Load the PyTorch model weights
state_dict = torch.load(base_ckpt_path + "ckpt_iter.pth.tar")["state_dict"]
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("module.", "")
    new_state_dict[new_key] = v
del state_dict
model.load_state_dict(new_state_dict, strict=False)
model.eval()

# Export the model to ONNX format
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, base_ckpt_path + "aenet.onnx")