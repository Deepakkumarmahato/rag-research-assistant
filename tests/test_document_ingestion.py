
import faiss

from app.rag_pipeline import load_rag_components
from app.document_ingestion import ingest_uploaded_pdf


# Load existing RAG components
model, original_index, original_metadata = load_rag_components()

# Clone the index so this test doesn't modify your original
index = faiss.clone_index(original_index)
metadata = original_metadata.copy()

# Read a PDF
pdf_path = "data/pdfs/machine_learning.pdf"

with open(pdf_path, "rb") as file:
    pdf_bytes = file.read()

# Record the original count
original_count = index.ntotal

# Ingest the PDF
index, metadata, added_chunks = ingest_uploaded_pdf(
    pdf_bytes,
    "new_test_upload.pdf",
    model,
    index,
    metadata
)

print("\n========== INGESTION TEST ==========")
print("Original vectors:", original_count)
print("New chunks:", added_chunks)
print("Updated vectors:", index.ntotal)
print("Updated metadata rows:", len(metadata))

assert index.ntotal == len(metadata)
assert index.ntotal == original_count + added_chunks

print("\nTEST PASSED!")
