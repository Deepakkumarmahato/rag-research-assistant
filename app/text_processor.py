import os
import re
import pandas as pd


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
INPUT_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "extracted",
    "extracted_pages.csv"
)
OUTPUT_DIR = os.path.join(BASE_DIR,"Data","extracted")

def clean_text(text):
    """
    Cleans the input text by removing unwanted characters and formatting.

    Args:
        text (str): The input text to be cleaned.
    """

    if not isinstance(text,str):
        return ""

    text = re.sub(r'\s+',' ',text) # replace excessive whitespace
    text = text.strip() # remove leading and trailing whitespace

    return text


def create_chunk(text,chunk_size=1000,chunk_overlap = 150):
    """
    Split the input text into chunks of specified size with optional overlap.
    
    Args:
        text (str): the input text to be chunked.
        chunk_size (int):the size of each chunk.Default is 1000 characters.
        overlap (int): the number of overlapping characters between consecutive chunks. Default is 150 characters.
    """
    chunks = []
    start = 0
    while start <len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks


def process_documents(df):
    """
    process the input DataFrame by cleaning the text and creating chunks for each document.
    
    Args:
        df (pd.DataFrame): the input DataFrame containing the documents to be processed. the DataFrame should have a column named 'text' containing the documnents to be processed.

    """
    all_chunks = []
    for _,row in df.iterrows():
        original_text = row['text']
        cleaned_text = clean_text(original_text)
        # generate chunks for the cleaned text
        page_chunks = create_chunk(cleaned_text,chunk_size = 1000,chunk_overlap = 150)

        for chunk_number, chunk in enumerate(page_chunks,start=1):
            chunk_data = {
                "document_name": row["document_name"],
                "page_number": row["page_number"],
                "chunk_number": chunk_number,
                "chunk_id": (
                    f"{row['document_name']}_"
                    f"page_{row['page_number']}_"
                    f"chunk_{chunk_number}"
                ),
                "text": chunk,
                "character_count": len(chunk)
            }

            all_chunks.append(chunk_data)
    return all_chunks



def main():

    # Check input file
    if not os.path.exists(INPUT_PATH):
        print("Input CSV not found:", INPUT_PATH)
        return

    # Load extracted pages
    df = pd.read_csv(INPUT_PATH)

    print("Original dataset shape:", df.shape)

    # Generate chunks
    all_chunks = process_documents(df)

    if not all_chunks:
        print("No chunks generated.")
        return

    # Convert chunks to DataFrame
    chunks_df = pd.DataFrame(all_chunks)

    # Save CSV
    csv_path = os.path.join(
        OUTPUT_DIR,
        "chunks.csv"
    )

    chunks_df.to_csv(
        csv_path,
        index=False
    )

    # Save JSONL
    json_path = os.path.join(
        OUTPUT_DIR,
        "chunks.jsonl"
    )

    chunks_df.to_json(
        json_path,
        orient="records",
        lines=True,
        force_ascii=False
    )

    # Summary
    print("\n========== CHUNKING SUMMARY ==========")

    print("Total original pages:", len(df))

    print("Total chunks:", len(chunks_df))

    print(
        "Average chunk size:",
        round(chunks_df["character_count"].mean(), 2)
    )

    print(
        "Maximum chunk size:",
        chunks_df["character_count"].max()
    )

    print(
        "Minimum chunk size:",
        chunks_df["character_count"].min()
    )

    print("\nFirst five chunks:")

    print(
        chunks_df[
            ["chunk_id", "character_count", "text"]
        ].head()
    )

    print("\nSaved CSV:", csv_path)

    print("Saved JSONL:", json_path)


if __name__ == "__main__":
    main()


