
import os
import fitz
import pandas as pd


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "extracted")



def extract_pdf_bytes(pdf_bytes, filename):

    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")

    extracted_pages = []

    try:

        with fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        ) as doc:

            if doc.is_encrypted:

                if not doc.authenticate(""):
                    raise ValueError(
                        "Password-protected PDFs are not supported."
                    )

            if len(doc) == 0:
                raise ValueError("The PDF contains no pages.")

            for page_number, page in enumerate(doc, start=1):

                text = page.get_text("text").strip()

                page_data = {
                    "document_name": filename,
                    "page_number": page_number,
                    "text": text,
                    "character_count": len(text)
                }

                extracted_pages.append(page_data)

    except ValueError:
        raise

    except Exception as error:

        raise ValueError(
            f"Unable to read PDF '{filename}': {error}"
        ) from error

    # Check whether any page contains extractable text
    total_text = sum(
        page["character_count"]
        for page in extracted_pages
    )

    if total_text == 0:

        raise ValueError(
            "No extractable text found. "
            "The PDF may contain scanned images. "
            "OCR support is not available yet."
        )

    return extracted_pages


def extract_pdf_text(pdf_path):
    """
    Extract text from a PDF stored on disk.
    """

    filename = os.path.basename(pdf_path)

    try:
        with open(pdf_path, "rb") as file:
            pdf_bytes = file.read()

        return extract_pdf_bytes(pdf_bytes, filename)

    except Exception as error:
        print(f"Error processing {filename}: {error}")
        return []


def process_all_pdfs():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_pages = []

    if not os.path.exists(PDF_DIR):
        return all_pages

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

    print("\n========== EXTRACTION SUMMARY ==========")

    print("Total pages:", len(df))
    print("Total documents:", df["document_name"].nunique())
    print("Total characters:", df["character_count"].sum())

    print(
        "Empty pages:",
        (df["text"].str.strip() == "").sum()
    )

    print("\nSaved CSV:", csv_path)
    print("Saved JSONL:", json_path)


if __name__ == "__main__":
    main()
