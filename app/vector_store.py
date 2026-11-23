import os
from numpy.char import index
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMBEDDING_PATH = os.path.join(BASE_DIR, "data","embeddings", "embeddings.npy")
METADATA_PATH = os.path.join(BASE_DIR, "data","embeddings", "embeddings_metadata.csv")

INDEX_DIR = os.path.join(BASE_DIR, "data", "vector_store")
INDEX_PATH = os.path.join(INDEX_DIR,"faiss.index")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

def build_faiss_index(embeddings):
    #get vector dimension
    dimension = embeddings.shape[1]

    #create faiss index
    index = faiss.IndexFlatIP(dimension)

    # add embeddings
    index.add(embeddings)

    print("faiss index created successfully")
    print("total vectors:",index.ntotal)
    print("vector dimension:",index.d)
    return index

def save_faiss_index(index):
    os.makedirs(INDEX_DIR,exist_ok=True)
    faiss.write_index(index,INDEX_PATH)
    print("faiss index saved sucessfully at:",INDEX_PATH)


#load faiss index
def load_faiss_index():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"faiss index not found at {INDEX_PATH}. Please build the index first.")

    index = faiss.read_index(INDEX_PATH)
    print("faiss index loaded successfully")
    return index

def embedding_model():
    model = SentenceTransformer(MODEL_NAME)
    return model

def generate_query_embedding(model,query):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )
    return np.ascontiguousarray(query_embedding)



def search_documents(query, model, index, metadata, top_k=TOP_K):

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty or whitespace.")

    if index.ntotal == 0:
        raise ValueError(
            "The FAISS index is empty. Please build the index first."
        )

    if len(metadata) != index.ntotal:
        raise ValueError(
            "FAISS index and metadata count do not match."
        )

    query_embedding = generate_query_embedding(model, query)

    # Avoid requesting more results than available
    top_k = min(top_k, index.ntotal)

    # Search FAISS
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        row = metadata.iloc[int(idx)]

        result = {
            "chunk_id": row["chunk_id"],
            "document_name": row["document_name"],
            "page_number": row["page_number"],
            "text": row["text"],
            "score": float(score),
            "similarity_score": float(score)
        }

        results.append(result)

    
    return results


def main():

    # Check required files
    if not os.path.exists(EMBEDDING_PATH):
        raise FileNotFoundError(
            f"Embeddings not found: {EMBEDDING_PATH}"
        )

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}"
        )

    # Load embeddings
    embeddings = np.load(EMBEDDING_PATH)

    embeddings = np.ascontiguousarray(
        embeddings,
        dtype=np.float32
    )

    # Load metadata
    metadata = pd.read_csv(METADATA_PATH)

    # Validate alignment
    if len(embeddings) != len(metadata):
        raise ValueError(
            "Embedding count and metadata count do not match!"
        )

    print("Embedding shape:", embeddings.shape)
    print("Metadata shape:", metadata.shape)

    # Build FAISS index
    index = build_faiss_index(embeddings)

    # Save FAISS index
    save_faiss_index(index)

    # Load index
    index = load_faiss_index()

    # Load embedding model
    model = embedding_model()

    # Test semantic search
    query = "What is machine learning?"

    print("\nUser Query:", query)

    results = search_documents(
        query=query,
        model=model,
        index=index,
        metadata=metadata,
        top_k=TOP_K
    )

    print("\n========== SEARCH RESULTS ==========")

    for i, result in enumerate(results, start=1):

        print(f"\nResult {i}")

        print("Document:", result["document_name"])

        print("Page:", result["page_number"])

        print(
            "Similarity:",
            round(result["score"], 4)
        )

        print("Text:", result["text"][:500])



    queries = [
    "What is machine learning?",
    "Explain neural networks.",
    "How does self attention work?",
    "What is LoRA fine tuning?",
    "What is the difference between CNN and RNN?"
    ]

    for query in queries:

        print("\n" + "=" * 60)
        print("QUERY:", query)

        results = search_documents(
            query=query,
            model=model,
            index=index,
            metadata=metadata,
            top_k=3
        )

        for result in results:

            print("\nSource:", result["document_name"])
            print("Page:", result["page_number"])
            print("Score:", round(result["score"], 4))
            print("Text:", result["text"][:250])


if __name__ == "__main__":
    main()
