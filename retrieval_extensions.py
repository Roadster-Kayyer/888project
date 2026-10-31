"""Duplicate sensitivity and paired FAQ-level bootstrap, without changing Experiment 1."""

from itertools import combinations
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
METRICS = ["Recall@1", "Recall@5", "Recall@10", "Hit@5", "MRR", "nDCG@5"]


def relevance_metrics(order, relevant):
    relevant = np.asarray(list(relevant), dtype=np.int32)
    positions = np.flatnonzero(np.isin(order, relevant)) + 1
    if len(positions) != len(relevant) or len(relevant) == 0:
        raise ValueError("Full ranking must contain every relevant document exactly once.")
    result = {f"Recall@{cutoff}": float((positions <= cutoff).sum() / len(relevant))
              for cutoff in [1, 5, 10]}
    result["Hit@5"] = float((positions <= 5).any())
    result["MRR"] = float(1 / positions.min())
    dcg = np.sum(1 / np.log2(positions[positions <= 5] + 1))
    ideal = np.sum(1 / np.log2(np.arange(1, min(5, len(relevant)) + 1) + 1))
    result["nDCG@5"] = float(dcg / ideal)
    return result


def bootstrap_tables(detail, metrics=METRICS, repeats=2000, seed=42, group="doc_id"):
    """Same group draws for all methods: paired differences, percentile intervals."""
    summaries, contrasts = [], []
    for query_type in ["overall"] + sorted(detail.query_type.unique().tolist()):
        subset = detail if query_type == "overall" else detail[detail.query_type == query_type]
        for metric in metrics:
            aggregated = subset.groupby([group, "method"])[metric].agg(["mean", "count", "size"])
            complete_means = aggregated["mean"].where(aggregated["count"] == aggregated["size"])
            grouped = complete_means.unstack("method")
            grouped = grouped.dropna()
            if grouped.empty:
                continue
            values = grouped.to_numpy()
            rng = np.random.default_rng(seed)
            sampled = np.empty((repeats, values.shape[1]))
            for start in range(0, repeats, 100):
                stop = min(start + 100, repeats)
                draws = rng.integers(0, len(values), size=(stop - start, len(values)))
                sampled[start:stop] = values[draws].mean(axis=1)
            for index, method in enumerate(grouped.columns):
                low, high = np.quantile(sampled[:, index], [0.025, 0.975])
                summaries.append({"query_type": query_type, "metric": metric, "method": method,
                                  "estimate": values[:, index].mean(), "ci_low": low, "ci_high": high,
                                  "n_groups": len(values), "bootstrap_repeats": repeats})
            for first, second in combinations(range(values.shape[1]), 2):
                differences = sampled[:, second] - sampled[:, first]
                low, high = np.quantile(differences, [0.025, 0.975])
                contrasts.append({"query_type": query_type, "metric": metric,
                                  "contrast": f"{grouped.columns[second]} - {grouped.columns[first]}",
                                  "difference": (values[:, second] - values[:, first]).mean(),
                                  "ci_low": low, "ci_high": high, "n_groups": len(values)})
    return (pd.DataFrame(summaries, columns=['query_type', 'metric', 'method', 'estimate', 'ci_low', 'ci_high', 'n_groups', 'bootstrap_repeats']),
            pd.DataFrame(contrasts, columns=['query_type', 'metric', 'contrast', 'difference', 'ci_low', 'ci_high', 'n_groups']))


def run(data_dir=None, output_dir=None, repeats=2000):
    data_dir = Path(data_dir or ROOT / "legal_faq_results")
    output_dir = Path(output_dir or ROOT / "retrieval_supplement")
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_csv(data_dir / "corpus.csv", dtype={"doc_id": str})
    queries = pd.read_csv(data_dir / "queries.csv", dtype={"doc_id": str})
    if not corpus.doc_id.is_unique or not queries.query_id.is_unique:
        raise ValueError("IDs must be unique.")
    index_by_id = dict(zip(corpus.doc_id, corpus.index))
    equivalents = corpus.groupby("text", sort=False).indices
    duplicate_ids = set(corpus.loc[corpus.text.duplicated(keep=False), "doc_id"])
    rows = []
    for method in ["BM25", "Dense", "Hybrid"]:
        rankings = np.load(data_dir / f"{method.lower()}_full_rankings.npy", mmap_mode="r")
        if rankings.shape != (len(queries), len(corpus)):
            raise ValueError("Ranking dimensions do not match corpus/queries.")
        print("Duplicate sensitivity:", method, flush=True)
        for query_index, query in enumerate(queries.itertuples(index=False)):
            gold = index_by_id[query.doc_id]
            order = rankings[query_index]
            for protocol, relevant in [("single_gold", [gold]),
                                       ("exact_answer_multi_gold", equivalents[corpus.iloc[gold].text])]:
                rows.append({"method": method, "query_id": query.query_id, "doc_id": query.doc_id,
                             "query_type": query.query_type, "protocol": protocol,
                             "duplicate_gold": query.doc_id in duplicate_ids,
                             "n_relevant": len(relevant), **relevance_metrics(order, relevant)})
    detail = pd.DataFrame(rows)
    # Exclude queries whose gold belongs to an exact-duplicate answer group; keep corpus fixed.
    excluded = detail[(detail.protocol == "single_gold") & ~detail.duplicate_gold].copy()
    excluded["protocol"] = "duplicate_gold_queries_excluded"
    detail = pd.concat([detail, excluded], ignore_index=True)
    detail.to_csv(output_dir / "sensitivity_per_query.csv", index=False, encoding="utf-8-sig")
    overall = detail.groupby(["protocol", "method"])[METRICS].mean().reset_index()
    overall["query_type"] = "overall"
    by_type = detail.groupby(["protocol", "method", "query_type"])[METRICS].mean().reset_index()
    pd.concat([overall, by_type]).to_csv(output_dir / "sensitivity_summary.csv", index=False)
    all_intervals, all_contrasts = [], []
    for protocol, subset in detail.groupby("protocol"):
        intervals, contrasts = bootstrap_tables(subset, repeats=repeats)
        intervals["protocol"], contrasts["protocol"] = protocol, protocol
        all_intervals.append(intervals)
        all_contrasts.append(contrasts)
    pd.concat(all_intervals).to_csv(output_dir / "bootstrap_ci.csv", index=False)
    pd.concat(all_contrasts).to_csv(output_dir / "paired_differences_ci.csv", index=False)
    # Audit balanced reformulations, without quietly excluding reviewed items from the benchmark.
    audit = pd.concat([queries[queries.query_type == style].sample(n=25, random_state=42)
                       for style in ["formal", "casual", "search", "hard"]])
    original = queries[queries.query_type == "original"].set_index("doc_id")["query"]
    audit["original_question"] = audit.doc_id.map(original)
    audit["reference_answer"] = audit.doc_id.map(corpus.set_index("doc_id").text)
    for column in ["meaning_preserved_yes_no_uncertain", "semantic_drift", "invalid", "notes"]:
        audit[column] = ""
    audit_path = output_dir / "query_quality_audit_100.csv"
    if not audit_path.exists():
        audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    metadata = {"bootstrap_seed": 42, "resampling_unit": "doc_id", "repeats": repeats,
                "ci": "paired percentile 95%; exploratory, no multiple-comparison adjustment",
                "multi_gold": "exact stripped answer string equality only, no semantic merging",
                "exclusion": "remove duplicate-gold queries; corpus unchanged; not a deduplicated index",
                "duplicate_gold_rows": len(duplicate_ids), "audit_review_status": "awaiting human review"}
    (output_dir / "protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    from review_audit import audit_reformulations
    audit_reformulations(output_dir, data_dir)
    return overall


if __name__ == "__main__":
    print(run().to_string(index=False))
