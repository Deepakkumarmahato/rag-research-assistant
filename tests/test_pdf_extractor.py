
from app.pdf_extractor import extract_pdf_bytes


PDF_PATH = "data/pdfs/machine_learning.pdf"


with open(PDF_PATH, "rb") as file:
    pdf_bytes = file.read()


pages = extract_pdf_bytes(
    pdf_bytes=pdf_bytes,
    filename="machine_learning_test.pdf"
)


print("\n========== UPLOAD EXTRACTION TEST ==========")

print("Total pages:", len(pages))

for page in pages:

    print("\nDocument:", page["document_name"])
    print("Page:", page["page_number"])
    print("Characters:", page["character_count"])
    print("Text:", page["text"][:300])
