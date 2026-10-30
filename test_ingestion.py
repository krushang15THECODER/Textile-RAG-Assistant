import pymupdf
from src.ingestion import extract_text_from_bytes


def main():
    pdf = pymupdf.open()
    page = pdf.new_page()

    page.insert_text(
        (72, 72),
        "Textile Manufacturing Report\n"
        "Cotton fabric production requires careful moisture control."
    )

    pdf_bytes = pdf.tobytes()
    pdf.close()

    pages = extract_text_from_bytes(pdf_bytes, "textile_test.pdf")

    for item in pages:
        print(f"Page {item['page_number']}:")
        print(item["text"])

    assert len(pages) == 1
    assert "Cotton fabric production" in pages[0]["text"]

    print("\nPDF ingestion test passed!")


if __name__ == "__main__":
    main()

