import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert evidence-level gold annotations to chunk-level gold annotations"
    )
    parser.add_argument("--evidence", required=True, help="Path to current_kb_gold_evidence.csv")
    parser.add_argument("--output", required=True, help="Output path for gold_chunks.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence_path = Path(args.evidence)
    output_path = Path(args.output)

    rows = []
    with evidence_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = (row.get("qid") or "").strip()
            chunk_id = str(row.get("chunk_id") or "").strip()
            reason = (row.get("reason") or "").strip()
            if not qid or not chunk_id:
                continue
            rows.append(
                {
                    "qid": qid,
                    "chunk_id": chunk_id,
                    "relevance": "1",
                    "reason": reason,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "chunk_id", "relevance", "reason"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
