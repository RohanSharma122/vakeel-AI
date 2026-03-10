# src/ingestion/parser.py
"""
Vakeel AI — PDF Parser (v2)
Handles the actual formatting found in Indian legal PDFs:
- Numbered sections like "128. Public servant..."
- "Section 128." format
- Chapter-based splitting as fallback
"""

import fitz
import pdfplumber
import json
import re
from pathlib import Path
from tqdm import tqdm

ACTS_DIR      = Path("data/raw/acts")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# TEXT CLEANER
# ═══════════════════════════════════════════════════════════

def clean_legal_text(text: str) -> str:
    text = re.sub(r'\n\s*\d{1,4}\s*\n', '\n', text)
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'_{3,}', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = text.replace('|', 'I')
    text = re.sub(r'\*{2,}', '', text)
    text = re.sub(r'\d+\*\*\*', '', text)
    return text.strip()


# ═══════════════════════════════════════════════════════════
# REMOVE TABLE OF CONTENTS
# ═══════════════════════════════════════════════════════════

def remove_toc(text: str) -> str:
    for pattern in [r'BE\s+it\s+enacted', r'CHAPTER\s+I\b', r'PRELIMINARY']:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return text[max(0, match.start() - 100):]
    return text


# ═══════════════════════════════════════════════════════════
# METADATA
# ═══════════════════════════════════════════════════════════

def extract_metadata(filename: str) -> dict:
    name_map = {
        "ipc":            "Indian Penal Code, 1860",
        "crpc":           "Code of Criminal Procedure, 1973",
        "hindu_marriage": "Hindu Marriage Act, 1955",
        "it_act":         "Information Technology Act, 2000",
        "motor_vehicles": "Motor Vehicles Act, 1988",
        "pocso":          "Protection of Children from Sexual Offences Act, 2012",
        "consumer":       "Consumer Protection Act, 2019",
        "rti":            "Right to Information Act, 2005",
        "domestic":       "Protection of Women from Domestic Violence Act, 2005",
    }
    act_name = filename.replace("_", " ").replace(".pdf", "").title()
    for key, full_name in name_map.items():
        if key in filename.lower():
            act_name = full_name
            break

    year_match = re.search(r'\b(18|19|20)\d{2}\b', filename)
    return {
        "filename": filename,
        "act_name": act_name,
        "year":     year_match.group() if year_match else "unknown",
        "doc_type": "bare_act",
    }


# ═══════════════════════════════════════════════════════════
# SMART SECTION SPLITTER
# ═══════════════════════════════════════════════════════════

def split_into_sections(text: str, act_name: str) -> list:
    sections = []

    # Strategy 1: "123. Title text" — most common in Indian PDFs
    pattern_numbered = re.compile(
        r'(?<!\d)(\d{1,3}[A-Z]?)\.\s{1,4}([A-Z][^\n]{5,}(?:\n(?!\d{1,3}[A-Z]?\.\s)[^\n]*)*)',
        re.MULTILINE
    )

    # Strategy 2: "Section 123." format
    pattern_section = re.compile(
        r'(Section\s+\d+[A-Za-z]?\s*[\.\-\u2014].*?)(?=Section\s+\d+[A-Za-z]?\s*[\.\-\u2014]|$)',
        re.DOTALL | re.IGNORECASE
    )

    # Strategy 3: Chapter-based
    pattern_chapter = re.compile(
        r'(CHAPTER\s+[IVXLCDM\d]+.*?)(?=CHAPTER\s+[IVXLCDM\d]+|$)',
        re.DOTALL | re.IGNORECASE
    )

    # Try Strategy 1
    matches1 = pattern_numbered.findall(text)
    if len(matches1) >= 10:
        print(f"   Strategy 1 (numbered): {len(matches1)} sections")
        for sec_num, sec_text in matches1:
            full_text = f"{sec_num}. {sec_text}".strip()
            if len(full_text) < 30:
                continue
            title_match = re.match(r'\d+[A-Z]?\.\s+([^.\u2014\n]{5,})', full_text)
            title = title_match.group(1).strip() if title_match else ""
            sections.append({
                "section_id": f"sec_{sec_num}",
                "title":      title,
                "text":       full_text,
                "type":       "section",
                "act":        act_name,
            })
        if len(sections) >= 10:
            return sections

    # Try Strategy 2
    sections = []
    matches2 = pattern_section.findall(text)
    if len(matches2) >= 5:
        print(f"   Strategy 2 (Section X.): {len(matches2)} sections")
        for match in matches2:
            match = match.strip()
            if len(match) < 50:
                continue
            sec_num = re.search(r'Section\s+(\d+[A-Za-z]?)', match, re.IGNORECASE)
            sec_id  = f"sec_{sec_num.group(1)}" if sec_num else "unknown"
            title_match = re.match(r'Section\s+\d+[A-Za-z]?\s*[\.\-\u2014]\s*([^.]{5,})', match)
            title = title_match.group(1).strip() if title_match else ""
            sections.append({
                "section_id": sec_id,
                "title":      title,
                "text":       match,
                "type":       "section",
                "act":        act_name,
            })
        if len(sections) >= 5:
            return sections

    # Try Strategy 3: Chapters
    sections = []
    matches3 = pattern_chapter.findall(text)
    if len(matches3) >= 2:
        print(f"   Strategy 3 (chapters): {len(matches3)} chapters")
        for i, match in enumerate(matches3):
            match = match.strip()
            if len(match) < 100:
                continue
            title_match = re.match(r'CHAPTER\s+\S+\s*\n\s*([^\n]+)', match, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else f"Chapter {i+1}"
            sections.append({
                "section_id": f"chapter_{i+1}",
                "title":      title,
                "text":       match,
                "type":       "chapter",
                "act":        act_name,
            })
        return sections

    # Fallback: paragraph chunks
    print(f"   Strategy 4 (paragraphs fallback)")
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 150]
    for i, para in enumerate(paragraphs):
        sections.append({
            "section_id": f"para_{i+1}",
            "title":      "",
            "text":       para,
            "type":       "paragraph",
            "act":        act_name,
        })
    return sections


# ═══════════════════════════════════════════════════════════
# PDF EXTRACTORS
# ═══════════════════════════════════════════════════════════

def extract_with_pymupdf(pdf_path: str) -> str:
    try:
        doc  = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        print(f"   PyMuPDF error: {e}")
        return ""

def extract_with_pdfplumber(pdf_path: str) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(
                p.extract_text() for p in pdf.pages if p.extract_text()
            )
    except Exception as e:
        print(f"   pdfplumber error: {e}")
        return ""

def extract_text(pdf_path: str) -> tuple:
    t1 = extract_with_pymupdf(pdf_path)
    t2 = extract_with_pdfplumber(pdf_path)
    if len(t1) >= len(t2):
        print(f"   Using PyMuPDF ({len(t1):,} chars)")
        return t1, "pymupdf"
    else:
        print(f"   Using pdfplumber ({len(t2):,} chars)")
        return t2, "pdfplumber"


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def parse_pdf(pdf_path: Path):
    print(f"\n{'='*50}")
    print(f"Parsing: {pdf_path.name}")

    raw_text, method = extract_text(str(pdf_path))
    if len(raw_text) < 500:
        print(f"   Too little text — skipping")
        return None

    clean_text = clean_legal_text(raw_text)
    clean_text = remove_toc(clean_text)
    metadata   = extract_metadata(pdf_path.name)
    sections   = split_into_sections(clean_text, metadata["act_name"])

    print(f"   {metadata['act_name']}: {len(sections)} sections, {len(clean_text):,} chars")

    return {
        "metadata":       metadata,
        "extraction":     method,
        "total_chars":    len(clean_text),
        "total_sections": len(sections),
        "sections":       sections,
    }


def parse_all_acts():
    print("="*60)
    print("PARSING ALL BARE ACTS (v2)")
    print("="*60)

    pdf_files = list(ACTS_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDFs found in data/raw/acts/")
        return

    summary        = []
    total_sections = 0

    for pdf_path in pdf_files:
        result = parse_pdf(pdf_path)
        if not result:
            continue

        save_path = PROCESSED_DIR / f"{pdf_path.stem}.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        total_sections += result["total_sections"]
        summary.append(result)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in summary:
        print(f"  {r['metadata']['act_name']}: {r['total_sections']} sections")

    print(f"\nTotal sections: {total_sections:,}")
    print(f"Saved to: {PROCESSED_DIR.absolute()}")


if __name__ == "__main__":
    parse_all_acts()