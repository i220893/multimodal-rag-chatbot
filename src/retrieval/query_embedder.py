import torch
import open_clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading CLIP model for query encoding on: {device}")

# CLIP model shared across text & image queries
model_name = "ViT-B-32"
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name, pretrained="openai", device=device
)
tokenizer = open_clip.get_tokenizer(model_name)


def embed_text_query(query):
    tokens = tokenizer([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()[0].tolist()


def embed_image_query(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        img_features = model.encode_image(img_tensor)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)

    return img_features.cpu().numpy()[0].tolist()
