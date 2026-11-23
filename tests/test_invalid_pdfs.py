
from app.pdf_extractor import extract_pdf_bytes


def run_test(name, pdf_bytes, filename):

    print(f"\nTesting: {name}")

    try:

        result = extract_pdf_bytes(
            pdf_bytes,
            filename
        )

        print("SUCCESS:", len(result), "pages extracted")

    except ValueError as error:

        print("EXPECTED ERROR:", error)


# Test 1: Empty file
run_test(
    "Empty PDF",
    b"",
    "empty.pdf"
)


# Test 2: Invalid content with PDF extension
run_test(
    "Corrupted PDF",
    b"This is not a valid PDF",
    "corrupted.pdf"
)


# Test 3: Unsupported extension
run_test(
    "Wrong file extension",
    b"Some file content",
    "document.txt"
)


# Test 4: Valid PDF
with open(
    "data/pdfs/machine_learning.pdf",
    "rb"
) as file:

    valid_pdf = file.read()

run_test(
    "Valid PDF",
    valid_pdf,
    "machine_learning.pdf"
)
