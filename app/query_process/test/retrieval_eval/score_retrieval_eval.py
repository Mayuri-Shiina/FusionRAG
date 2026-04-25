import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_questions(path: Path) -> Dict[str, dict]:
    questions: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = (row.get("qid") or "").strip()
            if not qid:
                continue
            questions[qid] = row
    return questions


def load_gold(path: Path) -> Dict[str, List[str]]:
    gold: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = (row.get("qid") or "").strip()
            chunk_id = str(row.get("chunk_id") or "").strip()
            if not qid or not chunk_id:
                continue
            gold.setdefault(qid, []).append(chunk_id)
    return gold


def normalize_prediction_row(obj: dict) -> List[str]:
    if isinstance(obj.get("retrieved_chunk_ids"), list):
        return [str(x).strip() for x in obj["retrieved_chunk_ids"] if str(x).strip()]

    results = obj.get("results")
    if isinstance(results, list):
        out: List[str] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            chunk_id = item.get("chunk_id") or item.get("id")
            if chunk_id is not None:
                out.append(str(chunk_id).strip())
        return [x for x in out if x]

    return []


def load_run(path: Path) -> Dict[str, List[str]]:
    data: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            qid = str(obj.get("qid") or "").strip()
            if not qid:
                raise ValueError(f"{path} line {line_no}: missing qid")
            data[qid] = normalize_prediction_row(obj)
    return data


def first_hit_rank(predicted: List[str], gold: List[str]) -> int:
    gold_set = set(gold)
    for idx, chunk_id in enumerate(predicted, start=1):
        if chunk_id in gold_set:
            return idx
    return 0


def compute_metrics(
    questions: Dict[str, dict],
    gold_map: Dict[str, List[str]],
    run_map: Dict[str, List[str]],
) -> Tuple[dict, List[dict]]:
    qids = [qid for qid in questions.keys() if qid in gold_map]
    total = len(qids)
    if total == 0:
        raise ValueError("No overlapped qid between questions.csv and gold_chunks.csv")

    hit_at = {1: 0, 3: 0, 5: 0, 10: 0}
    mrr_sum = 0.0
    rows: List[dict] = []

    for qid in qids:
        predicted = run_map.get(qid, [])
        gold = gold_map[qid]
        rank = first_hit_rank(predicted, gold)
        for k in hit_at:
            if rank and rank <= k:
                hit_at[k] += 1
        if rank:
            mrr_sum += 1.0 / rank

        rows.append(
            {
                "qid": qid,
                "query": questions[qid].get("query", ""),
                "gold_chunks": "|".join(gold),
                "predicted_top10": "|".join(predicted[:10]),
                "first_hit_rank": rank,
                "hit@10": 1 if rank and rank <= 10 else 0,
            }
        )

    metrics = {
        "total": total,
        "recall@1": hit_at[1] / total,
        "recall@3": hit_at[3] / total,
        "recall@5": hit_at[5] / total,
        "recall@10": hit_at[10] / total,
        "mrr": mrr_sum / total,
    }
    return metrics, rows


def print_metrics(name: str, metrics: dict) -> None:
    print(f"[{name}]")
    print(f"  total      : {metrics['total']}")
    print(f"  recall@1   : {metrics['recall@1']:.4f}")
    print(f"  recall@3   : {metrics['recall@3']:.4f}")
    print(f"  recall@5   : {metrics['recall@5']:.4f}")
    print(f"  recall@10  : {metrics['recall@10']:.4f}")
    print(f"  mrr        : {metrics['mrr']:.4f}")


def compare_runs(left_name: str, left_rows: List[dict], right_name: str, right_rows: List[dict]) -> None:
    left_map = {row["qid"]: row for row in left_rows}
    right_map = {row["qid"]: row for row in right_rows}
    shared_qids = sorted(set(left_map.keys()) & set(right_map.keys()))

    improved = []
    regressed = []
    unchanged = []

    for qid in shared_qids:
        left_hit = left_map[qid]["hit@10"]
        right_hit = right_map[qid]["hit@10"]
        if left_hit == 0 and right_hit == 1:
            improved.append(qid)
        elif left_hit == 1 and right_hit == 0:
            regressed.append(qid)
        else:
            unchanged.append(qid)

    print(f"\n[compare] {left_name} -> {right_name}")
    print(f"  improved  : {len(improved)}")
    print(f"  regressed : {len(regressed)}")
    print(f"  unchanged : {len(unchanged)}")

    if improved:
        print("  improved_qids:")
        for qid in improved[:20]:
            print(f"    - {qid}")

    if regressed:
        print("  regressed_qids:")
        for qid in regressed[:20]:
            print(f"    - {qid}")


def relative_uplift(base: float, new: float) -> float:
    if base == 0:
        return 0.0
    return (new - base) / base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score retrieval offline evaluation")
    parser.add_argument("--questions", required=True, help="Path to questions.csv")
    parser.add_argument("--gold", required=True, help="Path to gold_chunks.csv")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run definition in the form name=path/to/run.jsonl",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASE", "NEW"),
        help="Compare two named runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_questions(Path(args.questions))
    gold_map = load_gold(Path(args.gold))

    if not args.run:
        raise ValueError("At least one --run name=path is required")

    run_results: Dict[str, dict] = {}
    run_rows: Dict[str, List[dict]] = {}

    for item in args.run:
        if "=" not in item:
            raise ValueError(f"Invalid --run value: {item}")
        name, path_str = item.split("=", 1)
        name = name.strip()
        path = Path(path_str.strip())
        run_map = load_run(path)
        metrics, rows = compute_metrics(questions, gold_map, run_map)
        run_results[name] = metrics
        run_rows[name] = rows
        print_metrics(name, metrics)
        print()

    if args.compare:
        base_name, new_name = args.compare
        if base_name not in run_results or new_name not in run_results:
            raise ValueError("--compare names must appear in --run")
        base = run_results[base_name]["recall@10"]
        new = run_results[new_name]["recall@10"]
        uplift = relative_uplift(base, new)
        print(f"[uplift] {base_name} -> {new_name}")
        print(f"  recall@10 base : {base:.4f}")
        print(f"  recall@10 new  : {new:.4f}")
        print(f"  relative uplift: {uplift:.2%}")
        compare_runs(base_name, run_rows[base_name], new_name, run_rows[new_name])


if __name__ == "__main__":
    main()
