import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ingestion.cleaner import clean_text, find_repeated_edge_lines


def test_removes_page_numbers():
    text = "Revenue grew strongly this year.\n42\nPage 3 of 10\nCosts fell."
    out = clean_text(text)
    assert "42" not in out
    assert "Page 3" not in out
    assert "Revenue grew" in out


def test_fixes_hyphenation():
    assert "international" in clean_text("inter-\nnational trade grew")


def test_removes_repeated_headers():
    pages = []
    for i in range(6):
        # body lines differ by letters, so they are not mistaken for headers
        body = "\n".join(
            f"{chr(97 + i) * 5} {chr(98 + j) * 6} unique content line" for j in range(10)
        )
        pages.append(f"ACME Corp Annual Report 2023\n{body}\nACME Confidential")

    repeated = find_repeated_edge_lines(pages)
    cleaned = clean_text(pages[0], repeated)

    assert "ACME Corp" not in cleaned
    assert "ACME Confidential" not in cleaned
    assert "unique content line" in cleaned