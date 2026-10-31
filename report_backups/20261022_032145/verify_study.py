"""Deterministic checks of relevance, paired resampling, parsing, and completed pilot records."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from rag_pipeline import parse_judgment, token_f1, make_messages
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
    for malformed in ['{}', '{"correctness":true,"relevance":2,"faithfulness":1,"reason":"x"}',
                      '{"correctness":2,"relevance":2,"reason":"missing faithfulness key"}', 'not json']:
        try:
            parse_judgment(malformed, True)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid judge output accepted.")
    assert token_f1("A B [Document 1]", "a b") == 1
    assert "reference" not in json.dumps(make_messages("test question", "evidence")).lower()
    print("PASS: multi-gold metrics, full MRR, paired CI, missing scores, judge validation, citation stripping.")
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
