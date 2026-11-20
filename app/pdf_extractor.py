
import os
import fitz
import pandas as pd


# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "extracted")


def extract_pdf_text(pdf_path):
    """
    Extract text from a PDF page by page.

    Returns a list of dictionaries containing
    document name, page number and extracted text.
    """

    extracted_pages = []

    filename = os.path.basename(pdf_path)

    try:
        with fitz.open(pdf_path) as doc:

            for page_number, page in enumerate(doc, start=1):

                text = page.get_text("text").strip()

                page_data = {
                    "document_name": filename,
                    "page_number": page_number,
                    "text": text,
                    "character_count": len(text)
                }

                extracted_pages.append(page_data)

    except Exception as error:
        print(f"Error processing {filename}: {error}")

    return extracted_pages


def process_all_pdfs():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_pages = []

    for filename in os.listdir(PDF_DIR):

        if filename.lower().endswith(".pdf"):

            pdf_path = os.path.join(PDF_DIR, filename)

            print(f"Processing: {filename}")

            pages = extract_pdf_text(pdf_path)

            all_pages.extend(pages)

            print(f"Extracted pages: {len(pages)}")

    return all_pages


def main():

    all_pages = process_all_pdfs()

    if not all_pages:
        print("No pages extracted. Check your PDF folder.")
        return

    df = pd.DataFrame(all_pages)

    # Save extracted content
    csv_path = os.path.join(
        OUTPUT_DIR, "extracted_pages.csv"
    )

    json_path = os.path.join(
        OUTPUT_DIR, "extracted_pages.jsonl"
    )

    df.to_csv(csv_path, index=False)

    df.to_json(
        json_path,
        orient="records",
        lines=True,
        force_ascii=False
    )

    # Summary
    print("\n========== EXTRACTION SUMMARY ==========")

    print("Total pages:", len(df))

    print(
        "Total documents:",
        df["document_name"].nunique()
    )

    print(
        "Total characters:",
        df["character_count"].sum()
    )

    print(
        "Empty pages:",
        (df["text"].str.strip() == "").sum()
    )

    print("\nSample extracted text:\n")

    print(df.iloc[0]["text"][:1000])

    print("\nSaved CSV:", csv_path)
    print("Saved JSONL:", json_path)


if __name__ == "__main__":
    main()
