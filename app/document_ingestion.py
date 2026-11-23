
import hashlib

import faiss
import numpy as np
import pandas as pd

from app.pdf_extractor import extract_pdf_bytes
from app.text_processor import process_documents
from app.embedding_generator import generate_embeddings


def ingest_uploaded_pdf(
    pdf_bytes,
    filename,
    model,
    index,
    metadata
):
    # 1. Validate the uploaded file
    if not pdf_bytes:
        raise ValueError("Uploaded PDF is empty.")

    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")

    # 2. Generate a unique document name
    file_hash = hashlib.md5(pdf_bytes).hexdigest()[:10]
    document_name = f"{file_hash}_{filename}"

    # Avoid adding the same document twice
    if document_name in metadata["document_name"].values:
        return index, metadata, 0

    # 3. Extract PDF pages
    pages = extract_pdf_bytes(pdf_bytes, document_name)

    pages_df = pd.DataFrame(pages)

    # 4. Generate chunks
    chunks = process_documents(pages_df)

    if not chunks:
        raise ValueError("No usable text found in this PDF.")

    chunks_df = pd.DataFrame(chunks)

    # 5. Generate embeddings
    texts = chunks_df["text"].tolist()

    new_embeddings = generate_embeddings(model, texts)

    new_embeddings = np.ascontiguousarray(
        new_embeddings,
        dtype=np.float32
    )

    # 6. Validate embedding dimensions
    if new_embeddings.shape[1] != index.d:
        raise ValueError("Embedding dimension does not match FAISS index.")

    # 7. Add embeddings to existing FAISS index
    index.add(new_embeddings)

    # 8. Add metadata in exactly the same order
    updated_metadata = pd.concat(
        [metadata, chunks_df],
        ignore_index=True
    )

    # 9. Verify synchronization
    if index.ntotal != len(updated_metadata):
        raise RuntimeError("FAISS and metadata counts do not match.")

    print(f"Successfully ingested: {filename}")
    print(f"Pages: {len(pages)}")
    print(f"Chunks added: {len(chunks_df)}")
    print(f"Total FAISS vectors: {index.ntotal}")

    return index, updated_metadata, len(chunks_df)
