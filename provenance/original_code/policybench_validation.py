"""PolicyBench FAQ adaptation with pinned data and resumable local RAG runs."""
import gc
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from rank_bm25 import BM25Okapi

import rag_pipeline as rag
from model_runtime import CONFIG_PATH, LocalModel, ROOT
from retrieval_extensions import METRICS, bootstrap_tables, relevance_metrics

DATA = ROOT / "policybench_data"
REPO = "govtech/PolicyBenchFull"
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DENSE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
INSTRUCTION = "Represent this sentence for searching relevant passages: "
PROTOCOL = "policybench-faq-answer-corpus-v1"


def save_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def acquire():
    DATA.mkdir(exist_ok=True)
    manifest_path = DATA / "source_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        receipt = DATA / "download_info.json"
        revision = (json.loads(receipt.read_text(encoding="utf-8-sig"))["sha"]
                    if receipt.exists() else HfApi().dataset_info(REPO).sha)
        manifest = {"repo": REPO, "revision": revision,
                    "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                    "dataset_url": "https://huggingface.co/datasets/" + REPO,
                    "license": "No explicit dataset license found at initial inspection; verify before redistribution."}
    local = DATA / "processed_questions.csv"
    if not local.exists():
        cached = hf_hub_download(REPO, "processed_questions.csv", repo_type="dataset",
                                 revision=manifest["revision"])
        local.write_bytes(Path(cached).read_bytes())
    checksum = rag.file_hash(local)
    if manifest.get("sha256", checksum) != checksum:
        raise ValueError("Raw CSV changed; restore the pinned source before continuing.")
    manifest["sha256"] = checksum
    rag.atomic_json(manifest_path, manifest)
    return pd.read_csv(local, keep_default_na=False), manifest


def prepare_data(raw):
    required = ["identifier", "question", "expected_answer", "knowledge_base_identifier"]
    if not set(required) <= set(raw.columns):
        raise ValueError("Unexpected PolicyBench schema.")
    for column in required:
        if raw[column].astype(str).str.strip().eq("").any():
            raise ValueError("Blank required field: " + column)
    conflicts = raw.groupby("identifier")[required[1:]].nunique().gt(1).any(axis=1)
    if conflicts.any():
        raise ValueError("Conflicting repeated IDs: " + str(conflicts[conflicts].index.tolist()))
    unique = raw.drop_duplicates("identifier").sort_values("identifier").reset_index(drop=True)
    unique["text"] = unique.expected_answer.map(lambda value: html.unescape(value).strip())
    unique["doc_id"] = unique.text.map(lambda value: "PB_" + hashlib.sha256(value.encode()).hexdigest()[:20])
    corpus = unique[["doc_id", "text"]].drop_duplicates().reset_index(drop=True)
    if not corpus.doc_id.is_unique:
        raise ValueError("Document hash collision.")
    queries = pd.DataFrame({"query_id": unique.identifier, "doc_id": unique.doc_id,
                            "query_type": unique.knowledge_base_identifier,
                            "query": unique.question.map(lambda value: html.unescape(value).strip()),
                            "reference_answer": unique.text})
    if not queries.query_id.str.fullmatch(r"[A-Za-z0-9_-]+").all():
        raise ValueError("Unsafe source query ID.")
    save_csv(unique, DATA / "unique_questions.csv")
    save_csv(corpus, DATA / "corpus.csv")
    save_csv(queries, DATA / "queries.csv")
    audit = queries.copy()
    for column in ["source_url", "source_snapshot", "evidence_quote", "answer_supported",
                   "reviewed_by_researcher", "review_notes"]:
        audit[column] = ""
    if not (DATA / "source_quality_audit.csv").exists():
        save_csv(audit, DATA / "source_quality_audit.csv")
    counts = queries.groupby("query_type").size().rename("unique_queries").reset_index()
    save_csv(counts, DATA / "domain_counts.csv")
    return corpus, queries, counts


def retrieve(corpus, queries):
    from sentence_transformers import SentenceTransformer
    metadata_path = DATA / "retrieval_manifest.json"
    previous = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata = {"protocol": PROTOCOL,
                "corpus_sha256": rag.file_hash(DATA / "corpus.csv"),
                "queries_sha256": rag.file_hash(DATA / "queries.csv"),
                "dense_model": DENSE_MODEL,
                "dense_revision": DENSE_REVISION,
                "instruction": INSTRUCTION, "bm25_k1": 1.5, "bm25_b": 0.75, "rrf_k": 60,
                "code_sha256": rag.file_hash(__file__),
                "versions": {name: importlib.metadata.version(name) for name in
                             ["numpy", "pandas", "rank-bm25", "sentence-transformers", "transformers"]}}
    if previous and previous != metadata:
        raise ValueError("Retrieval configuration changed. Use a new DATA directory for a new protocol.")
    names = ["bm25", "dense", "hybrid"]
    if previous and all((DATA / (name + "_full_rankings.npy")).exists() for name in names):
        ranks = {name: np.load(DATA / (name + "_full_rankings.npy")) for name in names}
        if all(ranks[name].shape == (len(queries), len(corpus)) and
               rag.file_hash(DATA / (name + "_full_rankings.npy")) ==
               json.loads((DATA / "ranking_hashes.json").read_text())[name] for name in names):
            return ranks, metadata
        raise ValueError("Cached rankings failed integrity validation.")
    tokenize = lambda value: re.findall(r"\b\w+\b", value.lower())
    bm25 = BM25Okapi([tokenize(value) for value in corpus.text], k1=1.5, b=0.75)
    ranks = {"bm25": np.array([np.argsort(-bm25.get_scores(tokenize(query)), kind="stable")
                              for query in queries["query"]])}
    model = SentenceTransformer(DENSE_MODEL, revision=metadata["dense_revision"], device="cpu")
    try:
        query_texts = [INSTRUCTION + value for value in queries["query"]]
        length_rows = []
        for kind, ids, texts in [("document", corpus.doc_id, corpus.text),
                                 ("query", queries.query_id, query_texts)]:
            lengths = [len(model.tokenizer.encode(value, truncation=False)) for value in texts]
            length_rows.extend({"kind": kind, "id": identifier, "tokens": length,
                                "model_limit": model.max_seq_length, "truncated": length > model.max_seq_length}
                               for identifier, length in zip(ids, lengths))
        save_csv(pd.DataFrame(length_rows), DATA / "dense_token_audit.csv")
        docs = model.encode(corpus.text.tolist(), normalize_embeddings=True, batch_size=16, show_progress_bar=True)
        embeds = model.encode(query_texts, normalize_embeddings=True, batch_size=16, show_progress_bar=True)
        ranks["dense"] = np.argsort(-(embeds @ docs.T), axis=1, kind="stable")
    finally:
        del model
        gc.collect()
    ranks["hybrid"] = []
    for lexical, semantic in zip(ranks["bm25"], ranks["dense"]):
        scores = np.zeros(len(corpus))
        weights = 1 / (60 + np.arange(1, len(corpus) + 1))
        scores[lexical] += weights
        scores[semantic] += weights
        ranks["hybrid"].append(np.argsort(-scores, kind="stable"))
    ranks["hybrid"] = np.array(ranks["hybrid"])
    for name, values in ranks.items():
        np.save(DATA / (name + "_full_rankings.npy"), values)
    rag.atomic_json(DATA / "ranking_hashes.json",
                    {name: rag.file_hash(DATA / (name + "_full_rankings.npy")) for name in names})
    rag.atomic_json(metadata_path, metadata)
    return ranks, metadata


def retrieval_report(corpus, queries, rankings):
    indexes = dict(zip(corpus.doc_id, corpus.index))
    records = []
    top_rows = []
    for method, values in rankings.items():
        for position, query in enumerate(queries.to_dict("records")):
            order = values[position]
            if not np.array_equal(np.sort(order), np.arange(len(corpus))):
                raise ValueError("Invalid full ranking.")
            records.append({"query_id": query["query_id"], "doc_id": query["doc_id"],
                            "query_type": query["query_type"], "method": method,
                            **relevance_metrics(order, [indexes[query["doc_id"]]])})
            top_rows.extend({"query_id": query["query_id"], "method": method, "rank": rank,
                             "doc_id": corpus.iloc[int(index)].doc_id}
                            for rank, index in enumerate(order[:10], 1))
    detail = pd.DataFrame(records)
    intervals, contrasts = bootstrap_tables(detail)
    save_csv(detail, DATA / "retrieval_per_query.csv")
    save_csv(pd.DataFrame(top_rows), DATA / "retrieval_top10.csv")
    save_csv(intervals, DATA / "retrieval_ci.csv")
    save_csv(contrasts, DATA / "retrieval_paired_differences.csv")
    return intervals, contrasts


def prepare_run(corpus, queries, mode="pilot", label="v1"):
    if mode not in ["pilot", "formal"] or not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("Use pilot/formal and a simple run label.")
    pilot = queries.groupby("query_type", group_keys=False).sample(n=1, random_state=42)
    selected = pilot if mode == "pilot" else queries[~queries.doc_id.isin(pilot.doc_id)]
    selected = selected.copy().sort_values("query_id")
    selected["corpus_query_index"] = selected.index
    selected = selected.reset_index(drop=True)
    mode_key = "policybench_" + mode + "_" + label
    if rag.RUN_LABEL:
        raise ValueError("Set rag_pipeline.RUN_LABEL='' in this notebook; use label here instead.")
    output = rag.study_dir(mode_key)
    sample_bytes = selected.to_csv(index=False).encode("utf-8-sig")
    manifest = {"protocol": PROTOCOL, "mode": mode, "n_queries": len(selected), "seed": 42,
                "sample_sha256": hashlib.sha256(sample_bytes).hexdigest(),
                "models": json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["models"],
                "source": json.loads((DATA / "source_manifest.json").read_text()),
                "data_hashes": {name: rag.file_hash(DATA / name) for name in
                               ["corpus.csv", "queries.csv", "bm25_full_rankings.npy",
                                "dense_full_rankings.npy", "hybrid_full_rankings.npy"]},
                "code_hashes": {name: rag.file_hash(ROOT / name) for name in
                               ["policybench_validation.py", "rag_pipeline.py", "model_runtime.py",
                                "retrieval_extensions.py"]},
                "versions": {name: importlib.metadata.version(name) for name in
                             ["torch", "transformers", "bitsandbytes", "numpy", "pandas"]},
                "top_k": 5, "document_token_cap": 240, "max_input_tokens": 2048,
                "max_new_tokens": 160, "judge_max_input_tokens": 3072, "judge_max_new_tokens": 192,
                "generator_system": rag.GENERATOR_SYSTEM, "judge_system": rag.JUDGE_SYSTEM,
                "notes": "FAQ adaptation; expected answers form corpus. Not original leave-one-out benchmark."}
    rag.save_prepared_run(output, manifest, sample_bytes)
    return selected, output, manifest, mode_key


def generate(corpus, sample, rankings, output, manifest):
    pending = [(query, system) for query in sample.to_dict("records") for system in rag.SYSTEMS
               if not (output / "answers" / (query["query_id"] + "_" + system + ".json")).exists()]
    if pending:
        runtime = LocalModel("generator")
        try:
            for progress, (query, system) in enumerate(pending, 1):
                blocks, ids, trimmed = [], [], []
                if system in rag.RETRIEVERS:
                    order = rankings[rag.RETRIEVERS[system]][query["corpus_query_index"], :5]
                    for rank, index in enumerate(order, 1):
                        document = corpus.iloc[int(index)]
                        tokens = runtime.tokenizer.encode(document.text, add_special_tokens=False)
                        if len(tokens) > manifest["document_token_cap"]:
                            trimmed.append(document.doc_id)
                        excerpt = runtime.tokenizer.decode(tokens[:manifest["document_token_cap"]], skip_special_tokens=True)
                        blocks.append(f"[Document {rank}]\n{excerpt}")
                        ids.append(document.doc_id)
                context = "\n\n".join(blocks)
                messages = rag.make_messages(query["query"], context)
                result = runtime.generate(messages, manifest["max_new_tokens"], manifest["max_input_tokens"])
                record = {**query, "system": system, "context": context,
                          "retrieved_doc_ids": ids, "trimmed_doc_ids": trimmed,
                          "strict_gold_in_top5": query["doc_id"] in ids if ids else None,
                          "prompt_sha256": hashlib.sha256(json.dumps(messages).encode()).hexdigest(), **result}
                rag.atomic_json(output / "answers" / (query["query_id"] + "_" + system + ".json"), record)
                print(f"Generated {progress}/{len(pending)}: {query['query_id']} {system}", flush=True)
        finally:
            runtime.close()
    return rag.export_answers(output)


def report(mode_key):
    output = rag.study_dir(mode_key)
    manifest = json.loads((output / "manifest.json").read_text())
    if len(list((output / "answers").glob("*.json"))) != manifest["n_queries"] * 4:
        raise ValueError("Finish all four systems before reporting.")
    summary = rag.report(mode_key)
    notes = json.loads((output / "evaluation_notes.json").read_text())
    notes.update({"status": manifest["mode"] + " PolicyBench FAQ adaptation",
                  "pilot_warning": "Pilot has four questions; do not interpret its CI as thesis evidence.",
                  "external_validity": "Conditional on four Singapore domains and this static FAQ corpus; not NZ evidence.",
                  "bootstrap": "Paired exact-answer FAQ groups; does not model uncertainty across new government domains.",
                  "source_warning": "Expected-answer corpus; official source passage mapping has not been verified.",
                  "benchmark_comparability": "Do not compare with original PolicyBench leave-one-out abstention results."})
    rag.atomic_json(output / "evaluation_notes.json", notes)
    return summary

