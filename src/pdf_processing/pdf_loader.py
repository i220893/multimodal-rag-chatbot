import fitz  # PyMuPDF
import os

def load_pdf(filepath):
    """
    Extracts text blocks and embedded images from a PDF.

    Returns:
        pages_data: list of dicts
            {
                "page_num": int,
                "text_blocks": [ { "bbox": tuple, "text": str } ],
                "images": [ { "xref": int, "image_bytes": bytes } ]
            }
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PDF not found: {filepath}")

    doc = fitz.open(filepath)
    pages_data = []

    for page_index, page in enumerate(doc):
        text_blocks = []
        
        # Extract text blocks with bbox
        blocks = page.get_text("blocks")
        for b in blocks:
            bbox = b[:4]
            text = b[4]
            text_blocks.append({"bbox": bbox, "text": text})

        # Extract images
        image_list = page.get_images(full=True)
        images = []
        for img in image_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            images.append({
                "xref": xref,
                "image_bytes": base_image["image"]
            })

        pages_data.append({
            "page_num": page_index + 1,
            "text_blocks": text_blocks,
            "images": images
        })

    return pages_data
