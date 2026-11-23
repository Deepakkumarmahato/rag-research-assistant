import os 
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from app.vector_store import search_documents
from app.llm_generator import generate_rag_answer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMBEDDING_PATH = os.path.join(BASE_DIR,"data","embeddings","embeddings.npy")
METADATA_PATH = os.path.join(BASE_DIR,"data","embeddings","embeddings_metadata.csv")

FAISS_PATH = os.path.join(BASE_DIR,"data","vector_store","faiss.index")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_rag_components():
    embeddings = np.load(EMBEDDING_PATH)
    embeddings = np.ascontiguousarray(embeddings.astype("float32"))
    metadata = pd.read_csv(METADATA_PATH)

    if len(embeddings) != len(metadata):
        raise ValueError("Embedding and metadata count mismatch")

    index = faiss.read_index(FAISS_PATH)
    if index.ntotal != len(metadata):
        raise ValueError("FAISS index and metadata count mismatch")

    model = SentenceTransformer(MODEL_NAME)
    return model,index,metadata


def ask_question(question, model, index, metadata):

    # Validate user input
    if not isinstance(question, str):
        raise ValueError("Question must be text.")

    question = question.strip()

    if not question:
        raise ValueError("Please enter a question.")

    if len(question) > 2000:
        raise ValueError(
            "Question is too long. Maximum length is 2000 characters."
        )

    if index.ntotal == 0:
        raise ValueError(
            "The knowledge base is empty. Upload a PDF first."
        )

    if len(metadata) != index.ntotal:
        raise ValueError(
            "FAISS index and metadata are not synchronized."
        )

    retrieved_chunks = search_documents(
        query=question,
        model=model,
        index=index,
        metadata=metadata,
        top_k=3
    )

    if not retrieved_chunks:
        return (
            "No relevant information was found in the documents.",
            []
        )

    answer = generate_rag_answer(
        question=question,
        retrieved_chunks=retrieved_chunks
    )

    return answer, retrieved_chunks


def main():
    model,index,metadata = load_rag_components()
    question = input("\nAsk a question about your PDFs:")

    answer,sources = ask_question(
        question,
        model,
        index,
        metadata
    )
    print("\n========== GENERATED ANSWER ==========")
    print(answer)

    print("\n========== SOURCES ==========")

    for source in sources:

        print("\nDocument:", source["document_name"])
        print("Page:", source["page_number"])
        print("Similarity:", round(source["score"], 4))

if __name__ == "__main__":
    main()