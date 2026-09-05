import pymupdf
# tests done. pymupdf returns [] if the PDF is scanned or is not a PDF at all. Which means that that should toggle the enable OCR thing.

def ocr_required(pdf_path: str) -> bool:
    """
    Checks if OCR needs to be toggled on or not.

    Parameters:
        pdf_path - str path to the PDF uploaded on the server

    Returns:
        a boolean value on whether OCR is required or not.
    """
    doc = pymupdf.open(pdf_path)
    texts = []
    for page in doc[5:]:
        texts.append(page.get_text())
    # for debugging
    print(f"PyMuPDF returned {texts}")
    # for debugging
    if texts == []:
        return True
    return False
