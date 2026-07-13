import torch
import open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading CLIP text model on: {device}")

model_name = "ViT-B-32"
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name, pretrained="openai", device=device
)

tokenizer = open_clip.get_tokenizer(model_name)


def embed_text(text):
    """
    Returns a 512-dimensional CLIP text embedding.
    """
    if not text or text.strip() == "":
        return None

    tokens = tokenizer([text]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu().numpy()[0].tolist()
