"""Recover saved PolicyBench judgments with versioned, auditable syntax repairs."""

from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import re
import shutil

import pandas as pd

import judge_parsing
import policybench_validation as pb
import rag_pipeline as rag


def recover(record, has_context):
    errors = []
    for index, attempt in enumerate(record.get("attempts", [])):
        try:
            scores = judge_parsing.parse_judgment(attempt["answer"], has_context)
            return scores, index, "existing_complete_object_parser", errors
        except ValueError as error:
            errors.append({"attempt": index, "error": str(error)})
    for index, attempt in enumerate(record.get("attempts", [])):
        original = attempt["answer"].strip()
        repaired, count = re.subn(r",(\s*\})$", r"\1", original, count=1)
        if not count:
            continue
        try:
            scores = judge_parsing.validate_judgment(json.loads(repaired), has_context)
        except ValueError:
            continue
        return scores, index, "single_object_remove_final_trailing_comma", errors
    raise ValueError("No unambiguous syntax-only recovery: " + str(errors))


def main():
    rag.RUN_LABEL = ""
    mode_key = "policybench_formal_v3"
    output = rag.study_dir(mode_key)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    sample = pd.read_csv(output / "sample_queries.csv")
    answers = {path.stem: json.loads(path.read_text(encoding="utf-8"))
               for path in (output / "answers").glob("*.json")}
    expected = {f"{query}_{system}" for query in sample.query_id for system in rag.SYSTEMS}
    if len(sample) != manifest["n_queries"] or set(answers) != expected:
        raise ValueError("Incomplete answer coverage.")
    paths = sorted((output / "judgments").glob("*.json"))
    if {path.stem for path in paths} != expected:
        raise ValueError("Incomplete judgment coverage.")
    pending = []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["status"] != "ok":
            scores, attempt, rule, errors = recover(record, bool(answers[path.stem]["context"]))
            pending.append((path, record, scores, attempt, rule, errors))
    if not pending:
        print("All judgments already valid; no changes needed.")
        return
    stamp = datetime.now(ZoneInfo("Pacific/Auckland")).strftime("%Y%m%d_%H%M%S_%f")
    backup = rag.ROOT / "report_backups" / ("experiment3_repair_" + stamp)
    (backup / "judgments").mkdir(parents=True)
    for path in output.iterdir():
        if path.is_file():
            shutil.copy2(path, backup / path.name)
    protected = [output / "manifest.json", output / "sample_queries.csv",
                 output / "human_review_blinded.csv", output / "human_review_key.csv"]
    protected += list((output / "answers").glob("*.json"))
    protected += [path for path in paths if path not in [item[0] for item in pending]]
    before = {str(path): rag.file_hash(path) for path in protected}
    audit = []
    for path, record, scores, attempt, rule, errors in pending:
        shutil.copy2(path, backup / "judgments" / path.name)
        audit.append({"file": path.name, "attempt_index": attempt, "rule": rule,
                      "previous_error": record.get("error"), "parser_errors": errors,
                      "original_sha256": rag.file_hash(path), "scores": scores})
        record["parsing_recovery"] = {
            "rule": rule, "attempt_index": attempt, "original_error": record.get("error"),
            "timestamp_auckland": stamp, "backup": str(backup / "judgments" / path.name),
            "script_sha256": rag.file_hash(Path(__file__)),
            "parser_sha256": rag.file_hash(Path(judge_parsing.__file__))}
        record.update(status="ok", scores=scores, error=None)
        rag.atomic_json(path, record)
    summary = pb.report(mode_key)
    rag.import_human_review(mode_key)
    frame = pd.read_csv(output / "answer_evaluation.csv")
    if len(frame) != len(expected) or not frame.judge_status.eq("ok").all():
        raise ValueError("Post-repair coverage check failed.")
    for path, record, scores, attempt, rule, errors in pending:
        saved = json.loads(path.read_text(encoding="utf-8"))
        original = json.loads((backup / "judgments" / path.name).read_text(encoding="utf-8"))
        assert saved["attempts"] == original["attempts"]
        assert recover(original, bool(answers[path.stem]["context"]))[0] == saved["scores"]
    assert before == {str(path): rag.file_hash(path) for path in protected}
    previous = pd.read_csv(backup / "answer_quality_summary.csv").set_index("system")
    comparison = previous.join(summary, lsuffix="_before", rsuffix="_after")
    comparison.to_csv(output / "parsing_recovery_summary_comparison.csv", encoding="utf-8-sig")
    rag.atomic_json(output / "parsing_recovery_audit.json", {
        "timestamp_auckland": stamp, "backup": str(backup), "n_recovered": len(audit),
        "records": audit, "all_answers_and_existing_valid_scores_unchanged": True,
        "remaining_invalid": 0, "formal_researcher_review": "See judge_human_calibration.csv; recovery is not human review."})
    error_source = output / "error_analysis_current.csv"
    if not error_source.exists():
        error_source = output / "error_analysis.csv"
    annotations = pd.read_csv(error_source)[["query_id", "system", "manual_error_category", "manual_error_notes"]]
    frame.merge(annotations, on=["query_id", "system"], how="left", validate="one_to_one").to_csv(
        output / "error_analysis_current.csv", index=False, encoding="utf-8-sig")
    print(f"Recovered {len(audit)}; all {len(frame)} judgments valid. Backup: {backup}")
    print(summary[["correctness", "n_faq_groups_correctness", "faithfulness", "n_faq_groups_faithfulness"]].to_string())


if __name__ == "__main__":
    main()
