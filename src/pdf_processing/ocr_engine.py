import easyocr
import os
import torch

use_gpu = torch.cuda.is_available()
reader = easyocr.Reader(['en'], gpu=use_gpu)

def run_ocr(image_path):
    """
    Runs OCR on a given image and returns detected text.
    
    Returns:
        List of {bbox, text}
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = reader.readtext(image_path)

    ocr_output = []
    for (bbox, text, confidence) in results:
        ocr_output.append({
            "bbox": bbox,
            "text": text,
            "confidence": confidence
        })

    return ocr_output
