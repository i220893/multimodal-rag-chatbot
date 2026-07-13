import json
import os

from src.pdf_processing.pdf_loader import load_pdf
from src.pdf_processing.image_extractor import extract_page_images
from src.pdf_processing.ocr_engine import run_ocr
from src.chunking.chunker import (
    chunk_text_blocks,
    chunk_ocr_results,
    chunk_image
)
from src.utils.json_cleaner import to_python_type

def build_chunks_for_pdf(pdf_path):
    pdf_name = os.path.basename(pdf_path)
    chunks = []

    print(f"\n📄 Processing PDF: {pdf_name}")

    # 1. Extract text + embedded images
    pages_data = load_pdf(pdf_path)
    text_chunks = chunk_text_blocks(pages_data, pdf_name)
    chunks.extend(text_chunks)

    # 2. Extract page images
    page_images = extract_page_images(pdf_path)

    # 3. OCR + image chunks
    for idx, img_path in enumerate(page_images):
        page_num = idx + 1

        # OCR chunks
        ocr_results = run_ocr(img_path)
        ocr_chunks = chunk_ocr_results(ocr_results, pdf_name, page_num)
        chunks.extend(ocr_chunks)

        # Image chunk
        chunks.append(
            chunk_image(img_path, pdf_name, page_num)
        )

    return chunks


def build_all_chunks(pdf_folder="data", output="data/chunks.json"):
    all_chunks = []  # <-- this was missing

    # Get all PDF files from the folder
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

    for pdf_name in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_name)
        pdf_chunks = build_chunks_for_pdf(pdf_path)
        all_chunks.extend(pdf_chunks)

    print(f"\n🔢 Total chunks generated: {len(all_chunks)}")

    # Convert numpy.int32 etc. to Python types
    clean_chunks = [to_python_type(c) for c in all_chunks]

    # Save to JSON
    with open(output, "w", encoding="utf-8") as f:
        json.dump(clean_chunks, f, ensure_ascii=False, indent=2)

    print(f"💾 Chunks saved to: {output}")
