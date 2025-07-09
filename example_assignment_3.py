import requests
import torch
import torch.nn as nn
import os
from torchvision import models

os.makedirs("out/models", exist_ok=True)

TOKEN = "49390889"

#### SUBMISSION ####

# Create a dummy model and save model dict
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.weight.shape[1], 10)
torch.save(model.state_dict(), "out/models/dummy_submission.pt")

#### Tests ####
# (these are the assertions being ran on the eval endpoint for every submission)

allowed_models = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
}
with open("out/models/dummy_submission.pt", "rb") as f:
    try:
        model: torch.nn.Module = allowed_models["resnet18"](weights=None)
        model.fc = torch.nn.Linear(model.fc.weight.shape[1], 10)
    except Exception as e:
        raise Exception(
            f"Invalid model class, {e=}, only {allowed_models.keys()} are allowed",
        )
    try:
        state_dict = torch.load(f, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        out = model(torch.randn(1, 3, 32, 32))
    except Exception as e:
        raise Exception(f"Invalid model, {e=}")

    assert out.shape == (1, 10), "Invalid output shape"


# Send the model to the server, replace the string "TOKEN" with the string of token provided to you
response = requests.post("http://34.122.51.94:9090/robustness", files={"file": open("out/models/dummy_submission.pt", "rb")}, headers={"token": TOKEN, "model-name": "resnet18"})

# Should be 400, the clean accuracy is too low
print(response.json())
