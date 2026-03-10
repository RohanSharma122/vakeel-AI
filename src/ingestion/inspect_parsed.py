# src/ingestion/inspect_parsed.py
"""
Quick inspection of parsed acts to diagnose section splitting.
"""
import json
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

for json_file in PROCESSED_DIR.glob("*.json"):
    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    act_name  = data["metadata"]["act_name"]
    sections  = data["sections"]
    total_chars = data["total_chars"]

    print(f"\n{'='*60}")
    print(f"📜 {act_name}")
    print(f"   Sections found: {len(sections)}")
    print(f"   Total chars: {total_chars:,}")
    print(f"   Extraction method: {data['extraction']}")

    # Show first 3 sections
    print(f"\n   First 3 sections:")
    for sec in sections[:3]:
        preview = sec["text"][:150].replace("\n", " ")
        print(f"   [{sec['section_id']}] {preview}...")

    # Show a raw sample of the text to diagnose patterns
    print(f"\n   Raw text sample (chars 500-1000):")
    # Rebuild full text from sections
    sample = " ".join(s["text"] for s in sections[:5])
    print(f"   {repr(sample[500:800])}")                    