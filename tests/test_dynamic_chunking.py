
import pandas as pd

from app.pdf_extractor import extract_pdf_bytes
from app.text_processor import process_documents


PDF_PATH = "data/pdfs/machine_learning.pdf"


with open(PDF_PATH, "rb") as file:

    pdf_bytes = file.read()


pages = extract_pdf_bytes(
    pdf_bytes,
    "machine_learning_test.pdf"
)


pages_df = pd.DataFrame(pages)


chunks = process_documents(pages_df)


chunks_df = pd.DataFrame(chunks)


print("\n========== DYNAMIC CHUNKING TEST ==========")

print("Total pages:", len(pages_df))

print("Total chunks:", len(chunks_df))

print(
    "Average chunk size:",
    round(chunks_df["character_count"].mean(), 2)
)

print("\nFirst three chunks:")

print(
    chunks_df[
        ["chunk_id", "page_number", "character_count"]
    ].head(3)
)
