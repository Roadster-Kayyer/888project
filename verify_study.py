"""Deterministic checks of relevance, paired resampling, parsing, and completed pilot records."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from rag_pipeline import token_f1, make_messages, citation_metrics, paired_quality_tables, SYSTEMS
from judge_parsing import parse_judgment
from retrieval_extensions import relevance_metrics, bootstrap_tables


def run_checks():
    order = np.arange(12)
    values = relevance_metrics(order, [0, 7])
    assert values["Recall@1"] == 0.5 and values["Hit@5"] == 1
    assert values["Recall@10"] == 1 and values["MRR"] == 1
    np.testing.assert_allclose(values["nDCG@5"], 1 / (1 + 1 / np.log2(3)))
    assert relevance_metrics(order, [11])["MRR"] == 1 / 12
    assert relevance_metrics(order, [0, 1, 2])["nDCG@5"] == 1
    data = pd.DataFrame([{"doc_id": str(group), "query_type": style, "method": method, "score": group / 10}
                         for group in range(5) for style in ["original", "hard"] for method in ["A", "B"]])
    intervals, contrasts = bootstrap_tables(data, ["score"], repeats=100)
    assert contrasts.ci_low.eq(0).all() and contrasts.ci_high.eq(0).all()
    data.loc[0, "score"] = np.nan
    intervals, contrasts = bootstrap_tables(data, ["score"], repeats=100)
    assert intervals.loc[intervals.query_type == "overall", "n_groups"].eq(4).all()
    assert parse_judgment('{"correctness":2,"relevance":2,"faithfulness":2,"reason":"supported"}', False)["faithfulness"] is None
    judgment = '{"correctness":2,"relevance":2,"faithfulness":2,"reason":"supported"}'
    assert parse_judgment(judgment + '\n' + judgment, True)["correctness"] == 2
    assert parse_judgment(judgment + '\n{"correctness":', True)["correctness"] == 2
    assert parse_judgment('```json\n' + judgment + '\n```', True)["relevance"] == 2
    try:
        parse_judgment(judgment + judgment.replace('"correctness":2', '"correctness":0'), True)
    except ValueError:
        pass
    else:
        raise AssertionError("Conflicting repeated judgments accepted.")
    for malformed in ['{}', '{"correctness":true,"relevance":2,"faithfulness":1,"reason":"x"}',
                      '{"correctness":2,"relevance":2,"reason":"missing faithfulness key"}', 'not json']:
        try:
            parse_judgment(malformed, True)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid judge output accepted.")
    assert token_f1("A B [Document 1]", "a b") == 1
    assert token_f1("A B [Document N] [Document 1, Document 2]", "a b") == 1
    assert token_f1("A B [Documents 1, 2]", "a b") == 1
    assert citation_metrics("No citations [ordinary text]", 0)["citation_count"] == 0
    assert citation_metrics("[Document N]", 5)["malformed_citation_count"] == 1
    assert citation_metrics("[Document 1] [Document N]", 0)["invalid_citation_count"] == 2
    assert citation_metrics("[Document 1, Document 2]", 2)["citation_count"] == 2
    assert citation_metrics("[documents 1, 2 and 3]", 3)["invalid_citation_count"] == 0
    assert citation_metrics("[Document 0] [Document 6]", 5)["out_of_range_citation_count"] == 2
    assert citation_metrics("[Document 1, N]", 1)["invalid_citation_count"] == 1
    assert citation_metrics("[Document 1-3]", 5)["malformed_citation_count"] == 1
    paired = pd.DataFrame([
        {"doc_id": str(group), "query_type": style, "system": system,
         "correctness": float(group), "faithfulness": np.nan if system == "No-RAG" else float(group)}
        for group in range(3) for style in ["original", "hard"] for system in SYSTEMS])
    paired.loc[paired.index[paired.system == "BM25-RAG"][0], "correctness"] = np.nan
    summary, intervals, contrasts = paired_quality_tables(paired, ["correctness", "faithfulness"], repeats=100)
    assert summary.correctness.eq(1.5).all()
    assert summary.n_faq_groups_correctness.eq(2).all()
    assert pd.isna(summary.loc["No-RAG", "faithfulness"])
    assert summary.loc["No-RAG", "n_faq_groups_faithfulness"] == 0
    assert summary.loc["Dense-RAG", "n_faq_groups_faithfulness"] == 3
    for row in intervals[intervals.query_type == "overall"].itertuples():
        assert summary.loc[row.method, row.metric] == row.estimate
        assert summary.loc[row.method, f"{row.metric}_ci_low"] == row.ci_low
        assert summary.loc[row.method, f"{row.metric}_ci_high"] == row.ci_high
    assert contrasts.difference.eq(0).all()
    paired["correctness"] = np.nan
    empty_summary, _, _ = paired_quality_tables(paired, ["correctness"], repeats=100)
    assert empty_summary.correctness.isna().all()
    assert empty_summary.n_faq_groups_correctness.eq(0).all()
    assert "reference" not in json.dumps(make_messages("test question", "evidence")).lower()
    print("PASS: multi-gold metrics, full MRR, paired CI, aligned means, missing scores, judge validation, citation detection and stripping.")
    pilot = Path(__file__).resolve().parent / "rag_pilot"
    if (pilot / "answer_evaluation.csv").exists():
        frame = pd.read_csv(pilot / "answer_evaluation.csv")
        assert len(frame) == 80
        assert frame.groupby("query_id").system.nunique().eq(4).all()
        assert not frame.duplicated(["query_id", "system"]).any()
        assert frame.answer.notna().all()
        assert frame.loc[frame.system == "No-RAG", "faithfulness"].isna().all()
        assert frame.loc[frame.system == "No-RAG", "context"].isna().all()
        assert frame.loc[frame.system != "No-RAG", "context"].notna().all()
        print("PASS: 80 paired pilot answers, unique keys, No-RAG context and faithfulness absent.")


if __name__ == "__main__":
    run_checks()
