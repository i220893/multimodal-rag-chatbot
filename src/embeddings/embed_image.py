import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading CLIP model for images on: {device}")

model_name = "ViT-B-32"
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name, pretrained="openai", device=device
)

tokenizer = open_clip.get_tokenizer(model_name)


def embed_image(image_path):
    """
    Returns a CLIP image embedding (512 dims).
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except:
        print(f"Could not load image: {image_path}")
        return None

    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()[0].tolist()
