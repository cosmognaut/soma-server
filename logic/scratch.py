import pymupdf
# from docling.document_converter import DocumentConverter
#
# converter = DocumentConverter()
# result = converter.convert("/home/ishu/Projects/soma-server/app/uploads/a8a48000-a91d-11f1-9a47-172fe76ed7c7.pdf")
# print("Extracted text length:", len(result.document.export_to_markdown()))

NOT_A_PDF = "f03168c4-a914-11f1-9a47-172fe76ed7c7.jpg"
PERFECT_PDF = "a8a48000-a91d-11f1-9a47-172fe76ed7c7.pdf"
SCANNED_PDF = "scanned.pdf"

# tests done. pymupdf returns [] if the PDF is scanned or is not a PDF at all. Which means that that should toggle the enable OCR thing.

doc = pymupdf.open(f"/home/ishu/Projects/soma-server/app/uploads/{SCANNED_PDF}")
texts = []

print(f"Doc is {doc}")

for page in doc[5:]:
    texts.append(page.get_text())

print(texts)
