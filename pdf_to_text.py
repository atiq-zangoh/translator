#!/usr/bin/env python3
import logging
from pathlib import Path
from pdf2image import convert_from_path
import pytesseract

# ---------------- CONFIG ----------------
PDF_FILE = "sample.pdf"       # Path to your PDF
OUTPUT_FILE = "sample.txt"    # Output text file
DPI = 300                     # Image quality for OCR
LANG = "hin"                  # OCR language code (e.g., 'eng', 'hin' for Hindi)

# ---------------- LOGGING ----------------
log_file = "pdf_ocr_extraction.log"
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
)

# ---------------- FUNCTIONS ----------------
def ocr_extract_from_pdf(pdf_path: str) -> str:
    """Convert PDF pages to images and perform OCR on each page."""
    path = Path(pdf_path)
    if not path.is_file():
        logging.error(f"PDF file not found: {pdf_path}")
        raise FileNotFoundError(f"{pdf_path} does not exist")

    logging.info(f"Converting PDF to images: {pdf_path}")
    pages = convert_from_path(pdf_path, dpi=DPI)
    logging.info(f"PDF has {len(pages)} pages")

    extracted_text = ""
    for i, page in enumerate(pages, start=1):
        logging.info(f"Performing OCR on page {i}")
        text = pytesseract.image_to_string(page, lang=LANG)
        extracted_text += text + "\n\n"
    return extracted_text


def save_text(file_path: str, text: str):
    """Save extracted text to a .txt file."""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    logging.info(f"Text saved to {file_path}")


# ---------------- MAIN ----------------
def main():
    try:
        text = ocr_extract_from_pdf(PDF_FILE)
        save_text(OUTPUT_FILE, text)
        logging.info("✅ OCR PDF text extraction completed successfully.")
    except Exception as e:
        logging.error(f"OCR extraction failed: {e}")


if __name__ == "__main__":
    main()
