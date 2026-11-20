import os
import numpy as np 
import pandas as pd
from sentence_transformers import SentenceTransformer
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'Data','extracted' ,'chunks.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Data', 'embeddings')

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 32

def load_embedding_model():
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded successfully!")
    print(
        "Embedding dimension:",
        model.get_embedding_dimension()
    )
    return model

def generate_embeddings(model, texts):

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings


def main():
    if not os.path.exists(INPUT_PATH):
        print("Chunks csv not found:", INPUT_PATH)
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    print("Original dataframe shape:",df.shape)
    
    #remove missing or empty text
    df = df.dropna(subset = ['text']).copy()
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'] != ''].copy()
    print("Dataframe shape after removing missing or empty text:",df.shape)

    if df.empty:
        print("No valid text chunks found")
        return
    model = load_embedding_model()

    # convert text column into list
    texts = df['text'].tolist()

    # generate embedding
    embeddings = generate_embeddings(model,texts)

    #save embeddings vectors
    embeddings_path = os.path.join(OUTPUT_DIR, 'embeddings.npy')
    np.save(embeddings_path, embeddings)

    # save corresponding metadata
    metadata_path = os.path.join(OUTPUT_DIR, 'embeddings_metadata.csv')
    df.to_csv(metadata_path, index=False)
    print("Embeddings and metadata saved successfully!")


    # Summary
    print("\n========== EMBEDDING SUMMARY ==========")

    print("Total chunks:", len(texts))

    print("Embedding shape:", embeddings.shape)

    print("Embedding dimension:", embeddings.shape[1])

    print("Data type:", embeddings.dtype)

    print("Saved embeddings:", embeddings_path)

    print("Saved metadata:", metadata_path)

    print("\nFirst embedding:\n")

    print(embeddings[0][:10])


if __name__ == "__main__":
    main()
