"""Rebuild formal reports from saved model outputs without loading GPU models."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import pandas as pd

import rag_pipeline as rag
import judge_parsing


def import_researcher_review():
    output = rag.ROOT / "rag_formal"
    sample = pd.read_csv(output / "researcher_review_sample_80.csv", keep_default_na=False, dtype=str)
    full = pd.read_csv(output / "human_review_blinded.csv", keep_default_na=False, dtype=str).set_index("review_id")
    if not sample.review_id.is_unique or not sample.review_id.isin(full.index).all():
        raise ValueError("Unknown or duplicate review IDs.")
    sample = sample.set_index("review_id")
    columns = ["human_correctness", "human_relevance", "human_faithfulness", "human_notes"]
    for column in ["query", "reference_answer", "context", "answer"]:
        if not sample[column].eq(full.loc[sample.index, column]).all():
            raise ValueError(f"Source text changed: {column}.")
    for column in columns[:3]:
        if not sample[column].isin(["", "0", "1", "2"]).all():
            raise ValueError("Review scores must be 0, 1, 2, or blank.")
    if sample[columns[:3]].eq("").all().all():
        raise ValueError("The researcher review sample has no scores yet.")
    if (sample.context.eq("") & sample.human_faithfulness.ne("")).any():
        raise ValueError("Faithfulness must be blank without retrieved evidence.")
    for column in columns:
        supplied = sample[column].ne("")
        ids = sample.index[supplied]
        existing = full.loc[ids, column]
        if (existing.ne("") & existing.ne(sample.loc[ids, column])).any():
            raise ValueError("Conflicting existing human review; resolve manually before import.")
        full.loc[ids, column] = sample.loc[ids, column]
    backup = output / ("human_review_before_import_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".csv")
    shutil.copy2(output / "human_review_blinded.csv", backup)
    full.reset_index().to_csv(output / "human_review_blinded.csv", index=False, encoding="utf-8-sig")
    rag.RUN_LABEL = ""
    return rag.import_human_review("formal")


def closeout():
    rag.RUN_LABEL = ""
    output = rag.study_dir("formal")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    answer_paths = sorted((output / "answers").glob("*.json"))
    answers = pd.DataFrame([json.loads(path.read_text(encoding="utf-8")) for path in answer_paths])
    expected = pd.read_csv(output / "sample_queries.csv")
    expected_keys = {(query_id, system) for query_id in expected.query_id for system in rag.SYSTEMS}
    actual_keys = set(zip(answers.query_id, answers.system))
    if actual_keys != expected_keys or len(answers) != len(expected_keys):
        raise ValueError("Formal answer coverage is incomplete or duplicated.")
    if len(expected) != manifest["n_queries"] or answers.answer.fillna("").str.strip().eq("").any():
        raise ValueError("Formal sample size or answer content is invalid.")
    backup = rag.ROOT / "report_backups" / ("experiment2_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    backup.mkdir(parents=True)
    for path in output.iterdir():
        if path.is_file():
            shutil.copy2(path, backup / path.name)
    answer_hashes = {path.name: rag.file_hash(path) for path in answer_paths}
    review_hash = rag.file_hash(output / "human_review_blinded.csv")
    contexts = {(row.query_id, row.system): bool(row.context) for row in answers.itertuples()}
    recovery = []
    for path in sorted((output / "judgments").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["status"] == "ok":
            continue
        row = {"file": path.name, "previous_status": record["status"], "recovered": False}
        for attempt_index, attempt in enumerate(record.get("attempts", [])):
            try:
                scores = judge_parsing.parse_judgment(attempt["answer"], contexts[(record["query_id"], record["system"])])
            except ValueError:
                continue
            (backup / "judgments").mkdir(exist_ok=True)
            shutil.copy2(path, backup / "judgments" / path.name)
            record["parsing_recovery"] = {
                "policy": "first complete valid JSON; all additional complete objects must agree on scores; ignore unfinished trailing duplicate; no score inference",
                "attempt_index": attempt_index, "original_error": record.get("error"),
                "parser_sha256": rag.file_hash(Path(judge_parsing.__file__)),
                "utc": datetime.now(timezone.utc).isoformat()}
            record.update(status="ok", scores=scores, error=None)
            rag.atomic_json(path, record)
            row.update(recovered=True, attempt_index=attempt_index)
            break
        recovery.append(row)
    pd.DataFrame(recovery, columns=["file", "previous_status", "recovered", "attempt_index"]).to_csv(
        backup / "parsing_recovery.csv", index=False)
    summary = rag.report("formal")
    calibration = rag.import_human_review("formal")
    frame = pd.read_csv(output / "answer_evaluation.csv")
    available = pd.read_csv(output / "answer_quality_available_case_summary.csv")
    error_source = output / "error_analysis_current.csv"
    if not error_source.exists():
        error_source = output / "error_analysis.csv"
    old_errors = pd.read_csv(error_source)
    annotations = old_errors[["query_id", "system", "manual_error_category", "manual_error_notes"]]
    frame.merge(annotations, on=["query_id", "system"], how="left", validate="one_to_one").to_csv(
        output / "error_analysis_current.csv", index=False, encoding="utf-8-sig")
    review = pd.read_csv(output / "human_review_blinded.csv", keep_default_na=False)
    keys = pd.read_csv(output / "human_review_key.csv")
    chosen_docs = expected[["doc_id"]].drop_duplicates().sample(n=20, random_state=42).doc_id.tolist()
    styles = ["original", "formal", "casual", "search", "hard"]
    selected_queries = [expected.loc[(expected.doc_id == doc_id) & (expected.query_type == styles[index % 5]),
                                    "query_id"].iloc[0] for index, doc_id in enumerate(chosen_docs)]
    selected_reviews = set(keys.loc[keys.query_id.isin(selected_queries), "review_id"])
    review_sample = review[review.review_id.isin(selected_reviews)].copy()
    sample_path = output / "researcher_review_sample_80.csv"
    if not sample_path.exists():
        review_sample.to_csv(sample_path, index=False, encoding="utf-8-sig")
    assert len(review_sample) == 80
    assert review_hash == rag.file_hash(output / "human_review_blinded.csv")
    assert answer_hashes == {path.name: rag.file_hash(path) for path in answer_paths}
    invalid = int(frame.judge_status.ne("ok").sum())
    total_recovered = sum("parsing_recovery" in json.loads(path.read_text(encoding="utf-8"))
                          for path in (output / "judgments").glob("*.json"))
    verified = []
    for path in sorted((output / "judgments").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if "parsing_recovery" not in record:
            continue
        attempt = record["attempts"][record["parsing_recovery"]["attempt_index"]]["answer"]
        if judge_parsing.parse_judgment(attempt, contexts[(record["query_id"], record["system"])]) != record["scores"]:
            raise ValueError("Recovered judgment cannot be reproduced with the versioned parser.")
        verified.append({"file": path.name, "sha256": rag.file_hash(path)})
    rag.atomic_json(output / "parser_verification.json", {
        "parser": "judge_parsing.py", "sha256": rag.file_hash(Path(judge_parsing.__file__)),
        "note": "Experiment 2 recovery parser isolated to preserve Experiment 3 source fingerprints.",
        "verified_recovered_records": verified})
    table = ["| System | Correctness (95% CI) | Relevance (95% CI) | Faithfulness (95% CI) |",
             "|---|---|---|---|"]
    for system, row in summary.iterrows():
        cells = []
        for metric in ["correctness", "relevance", "faithfulness"]:
            cells.append("N/A" if pd.isna(row[metric]) else
                         f"{row[metric]:.3f} [{row[metric + '_ci_low']:.3f}, {row[metric + '_ci_high']:.3f}], n={int(row['n_faq_groups_' + metric])} FAQs")
        table.append("| " + system + " | " + " | ".join(cells) + " |")
    text = "\n".join([
        "# Experiment 2 — formal closeout", "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"400 queries / 80 FAQs / 1,600 saved answers; invalid or missing judgments: {invalid}.",
        f"{total_recovered} judgment records have parsing-recovery provenance from their original model text ({sum(row['recovered'] for row in recovery)} recovered this pass). No generation or new model scoring was performed.",
        f"Backup and recovery audit: {backup}", "",
        "## Main comparison", "", *table, "",
        "Scores are ordinal 0–2 rubric ratings from a local model, not factual accuracy percentages. Means and percentile 95% CIs use the same complete paired FAQ groups; 2,000 bootstrap samples. Faithfulness excludes No-RAG and abstentions. Report missingness and excluded groups; do not mix these means with available-case CIs.",
        "RAG performance is conditional on this dataset, sample, generator and judge. These results do not establish New Zealand public-service generalization.",
        "The reused answers and original judge attempts are preserved. Parsing accepts the first complete valid object, requires other complete objects to agree on all scores, and ignores an unfinished trailing duplicate. This is a disclosed postprocessing change, not new human annotation.", "",
        "## Researcher review still required", "",
        "Formal paired human/model counts: " + calibration[["metric", "n_paired_reviews"]].to_json(orient="records"),
        "Fill researcher_review_sample_80.csv yourself: 20 distinct formal FAQs, one variant per FAQ, four systems, four questions per query type. Preserve review_id and all source text. Run `from closeout_experiment2 import import_researcher_review; import_researcher_review()` to safely import the scores, then rerun this script. No-RAG faithfulness is blank; use notes for N/A and pure abstention. Single-researcher review is a limited reference check, not independent multi-rater validation.",
        "If the rubric or judge prompt changes after reviewing this sample, label it as development and validate on different FAQs.", "",
        "## Remaining thesis work", "",
        "Inspect disagreements and document representative retrieval, generation and citation errors in error_analysis_current.csv. Review automatic ceiling effects; a zero-width bootstrap CI does not prove a perfect evaluator. Use paired differences to discuss strategy comparisons, not point estimates alone. Complete Experiment 3 independently and write the limitations and RQ discussion.", "",
        "Operational and available-case statistics:", "```", available.to_string(index=False), "```", ""])
    (output / "EXPERIMENT_2_CLOSEOUT.md").write_text(text, encoding="utf-8")
    print(summary.to_string())
    print(f"Recovered: {sum(row['recovered'] for row in recovery)}; remaining invalid: {invalid}; backup: {backup}")
    print(calibration.to_string(index=False))


if __name__ == "__main__":
    closeout()
