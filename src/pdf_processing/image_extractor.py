from pdf2image import convert_from_path
import os

def extract_page_images(pdf_path, dpi=200, output_folder="data/page_images"):
    """
    Converts each PDF page into a full-page image.

    Returns:
        list of file paths to the generated images
    """
    os.makedirs(output_folder, exist_ok=True)
    
    images = convert_from_path(pdf_path, dpi=dpi)
    saved_paths = []
    
    for i, img in enumerate(images):
        out_path = os.path.join(output_folder, f"page_{i+1}.png")
        img.save(out_path, "PNG")
        saved_paths.append(out_path)
    
    return saved_paths
