from src.pdf_processing.pdf_loader import load_pdf
from src.pdf_processing.image_extractor import extract_page_images
from src.pdf_processing.ocr_engine import run_ocr


PDF_PATH = "data/1. Annual Report 2023-24.pdf"   # change this to your file name

print("🔍 Testing PDF text extraction...")
pages = load_pdf(PDF_PATH)
print(f"Pages extracted: {len(pages)}")

print("First page text sample:")
print(pages[0]["text_blocks"][:3])  # print first 3 text blocks

print("\n🖼️ Testing page image extraction...")
page_imgs = extract_page_images(PDF_PATH)
print("Generated images:", page_imgs)

print("\n🔤 Testing OCR on first page image...")
ocr_text = run_ocr(page_imgs[0])
print("OCR sample:", ocr_text[:3])
