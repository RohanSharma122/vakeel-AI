# src/rag/clean_metadata.py
"""
Strips HTML tags from judgment chunks in metadata.jsonl
Run once after downloading vector store from Colab.
"""

import json
import re
from pathlib import Path
from tqdm import tqdm

VECTOR_STORE_DIR = Path("data/vector_store")


def strip_html(text: str) -> str:
    """Remove HTML tags and decode HTML entities."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode common HTML entities
    text = text.replace('&#x27;', "'")
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")
    text = text.replace('&nbsp;', ' ')
    # Clean up extra whitespace left by removed tags
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def clean_metadata():
    metadata_path = VECTOR_STORE_DIR / "metadata.jsonl"
    backup_path   = VECTOR_STORE_DIR / "metadata_backup.jsonl"
    clean_path    = VECTOR_STORE_DIR / "metadata.jsonl"

    # Load all metadata
    print("📂 Loading metadata...")
    records = []
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"✅ Loaded {len(records):,} records")

    # Backup original
    import shutil
    shutil.copy(metadata_path, backup_path)
    print(f"✅ Backup saved → {backup_path}")

    # Clean HTML from all text fields
    print("🧹 Cleaning HTML tags...")
    cleaned = 0
    for record in tqdm(records):
        original = record.get("text", "")
        record["text"] = strip_html(original)
        if original != record["text"]:
            cleaned += 1

    print(f"✅ Cleaned {cleaned:,} records with HTML")

    # Save cleaned metadata
    with open(clean_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Clean metadata saved → {clean_path}")

    # Show before/after example
    print("\n📋 Before/After example:")
    sample = next((r for r in records if r.get("source") == "court_judgment"), None)
    if sample:
        print(f"  Clean text: {sample['text'][:200]}")


if __name__ == "__main__":
    clean_metadata()
    print("\n✅ Done! Restart retriever to use clean text.")