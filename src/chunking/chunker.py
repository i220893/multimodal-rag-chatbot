import uuid
import re

def clean_text(text):
    """Remove extra spaces, newlines, and junk characters."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text_blocks(pages_data, pdf_name, max_chars=600):
    """
    Chunk text blocks into controlled-size chunks with metadata.
    """
    chunks = []

    for page in pages_data:
        page_num = page["page_num"]
        text_blocks = page["text_blocks"]

        buffer = ""
        for block in text_blocks:
            text = clean_text(block["text"])

            if not text or len(text) < 3:
                continue

            if len(buffer) + len(text) < max_chars:
                buffer += " " + text
            else:
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "type": "text",
                    "content": buffer.strip(),
                    "source_pdf": pdf_name,
                    "page_num": page_num,
                    "bbox": None
                })
                buffer = text

        # last buffer
        if buffer.strip():
            chunks.append({
                "id": str(uuid.uuid4()),
                "type": "text",
                "content": buffer.strip(),
                "source_pdf": pdf_name,
                "page_num": page_num,
                "bbox": None
            })

    return chunks


def chunk_ocr_results(ocr_results, pdf_name, page_num):
    """
    Convert OCR blocks into chunk objects.
    """
    chunks = []

    for item in ocr_results:
        text = clean_text(item["text"])
        if len(text) < 2:
            continue

        chunks.append({
            "id": str(uuid.uuid4()),
            "type": "ocr",
            "content": text,
            "source_pdf": pdf_name,
            "page_num": page_num,
            "bbox": item["bbox"]
        })

    return chunks


def chunk_image(image_path, pdf_name, page_num):
    """
    Represent images as chunks. Embeddings will use CLIP.
    """
    return {
        "id": str(uuid.uuid4()),
        "type": "image",
        "content": image_path,
        "source_pdf": pdf_name,
        "page_num": page_num,
        "bbox": None
    }
