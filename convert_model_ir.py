from torch import load
from torch.nn import Linear
from openvino.tools.mo import convert_model
import openvino.runtime as ov

from torchvision.models import resnet50, ResNet50_Weights

saved_model = load("./models/vegetables.pt", map_location="cpu")

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.fc = Linear(2048, 15)
model.load_state_dict(saved_model["model_state_dict"])

ov_model = convert_model(model, compress_to_fp16=True)

ov.serialize(ov_model, "quantized_model.xml")