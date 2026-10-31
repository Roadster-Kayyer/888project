"""Reproducible consolidation and notebook preparation, without changing research judgments."""

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import shutil

import nbformat
import pandas as pd

from project_support import ROOT, sha256

ORIGINAL_CODE = ['policybench_validation.py', 'rag_pipeline.py', 'model_runtime.py', 'retrieval_extensions.py']
CURRENT_CODE = ORIGINAL_CODE + ['project_support.py', 'judge_parsing.py', 'review_audit.py', 'review_tools.py']
NOTEBOOKS = ['Legal_FAQ_Retrieval_Experiment.ipynb', 'Experiment_1_Retrieval_Supplement.ipynb',
             'Experiment_2_RAG_Generation_Evaluation.ipynb', 'Experiment_3_PolicyBench_Validation.ipynb']


def prepare(backup):
    if (ROOT / 'provenance' / 'consolidation.json').exists():
        raise RuntimeError('One-time consolidation is complete. Re-running would overwrite review sample copies.')
    backup = Path(backup).resolve()
    if not (backup / 'backup_manifest.json').exists():
        raise ValueError('A verified external backup is required.')
    provenance = ROOT / 'provenance'
    (provenance / 'original_code').mkdir(parents=True, exist_ok=True)
    for name in ORIGINAL_CODE:
        source = backup / 'project' / name
        target = provenance / 'original_code' / name
        if not target.exists():
            shutil.copy2(source, target)
    first, second = ROOT / 'rag_formal', ROOT / 'rag_formal_v2'
    if second.exists():
        left = pd.read_csv(first / 'generated_answers.csv', keep_default_na=False, dtype=str)
        right = pd.read_csv(second / 'generated_answers.csv', keep_default_na=False, dtype=str)
        fields = ['query_id', 'system', 'query', 'answer', 'context', 'reference_answer']
        if not left[fields].sort_values(fields[:2]).reset_index(drop=True).equals(
                right[fields].sort_values(fields[:2]).reset_index(drop=True)):
            raise ValueError('The formal runs are not duplicates; consolidation refused.')
        keys = pd.read_csv(first / 'human_review_key.csv', dtype=str)
        if not keys.equals(pd.read_csv(second / 'human_review_key.csv', dtype=str)):
            raise ValueError('Formal review IDs differ between versions.')
        latest = pd.read_csv(second / 'human_review_blinded.csv', keep_default_na=False, dtype=str)
        current = pd.read_csv(first / 'human_review_blinded.csv', keep_default_na=False, dtype=str)
        for field in ['review_id', 'query', 'reference_answer', 'context', 'answer']:
            if not latest[field].equals(current[field]):
                raise ValueError('Review source changed: ' + field)
        review_columns = ['human_correctness', 'human_relevance', 'human_faithfulness', 'human_notes']
        selected = latest[review_columns].ne('').any(axis=1)
        current.loc[selected, review_columns] = latest.loc[selected, review_columns]
        current.to_csv(first / 'human_review_blinded.csv', index=False, encoding='utf-8-sig')
        receipt = {'canonical_run': 'rag_formal', 'merged_from': 'rag_formal_v2',
                   'same_answer_context_rows': len(left), 'reviewed_rows_merged': int(selected.sum()),
                   'human_score_source': 'researcher file in rag_formal_v2; no assistant scoring changes',
                   'external_backup': str(backup), 'merged_source_sha256': sha256(second / 'human_review_blinded.csv')}
        (provenance / 'consolidation.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    for name in ['rag_formal', 'rag_policybench_formal_v3']:
        directory = ROOT / name
        review = pd.read_csv(directory / 'human_review_blinded.csv', keep_default_na=False, dtype=str)
        selected = review[['human_correctness', 'human_relevance', 'human_faithfulness']].ne('').any(axis=1)
        sample = review.loc[selected]
        if len(sample) != 80:
            raise ValueError('Expected the existing 80 researcher-reviewed answers; no new sampling performed.')
        sample.to_csv(directory / 'researcher_review_sample_80.csv', index=False, encoding='utf-8-sig')
        keys = pd.read_csv(directory / 'human_review_key.csv', dtype=str)
        queries = pd.read_csv(directory / 'sample_queries.csv', dtype=str)
        metadata = sample.merge(keys, on='review_id').merge(queries[['query_id', 'query_type']], on='query_id')
        selection = {'n_answers': len(sample), 'n_queries': metadata.query_id.nunique(),
                     'queries_by_type': metadata.drop_duplicates('query_id').query_type.value_counts().to_dict(),
                     'method': 'Existing scored review IDs preserved; no outcome-based resampling by maintenance.',
                     'sampling_provenance': 'fixed seed 42 sampling documented in original E2 closeout' if name == 'rag_formal'
                     else 'Researcher selection mechanism not recorded; observed domain counts are not equal allocation.'}
        (directory / 'review_sample_metadata.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
        old = directory / 'error_analysis_current.csv'
        if old.exists():
            current_errors = pd.read_csv(directory / 'error_analysis.csv', keep_default_na=False)
            annotations = pd.read_csv(old, keep_default_na=False).set_index(['query_id', 'system'])
            current_errors = current_errors.set_index(['query_id', 'system'])
            for field in ['manual_error_category', 'manual_error_notes']:
                nonempty = annotations[field].ne('')
                existing = current_errors.loc[annotations.index[nonempty], field]
                proposed = annotations.loc[nonempty, field]
                if (existing.ne('') & existing.ne(proposed)).any():
                    raise ValueError('Conflicting error-analysis annotations.')
                current_errors.loc[proposed.index, field] = proposed
            current_errors.reset_index().to_csv(directory / 'error_analysis.csv', index=False, encoding='utf-8-sig')
    (provenance / 'backup_location.json').write_text(json.dumps({'path': str(backup)}, indent=2), encoding='utf-8')


def receipts():
    original = {name: sha256(ROOT / 'provenance' / 'original_code' / name) for name in ORIGINAL_CODE}
    receipt = {'created_auckland': datetime.now(ZoneInfo('Pacific/Auckland')).isoformat(),
               'scope': 'Parser, annotation validation, reports, cache replay and explicit source compatibility only. Retrieval/generation parameters unchanged.',
               'original_source_hashes': original,
               'current_source_hashes': {name: sha256(ROOT / name) for name in CURRENT_CODE}}
    (ROOT / 'provenance' / 'source_compatibility.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')


def notebooks():
    for name in NOTEBOOKS:
        path = ROOT / name
        notebook = nbformat.read(path, as_version=4)
        if notebook.metadata.get('maintenance_replay_version') == 1:
            continue
        if name == NOTEBOOKS[0]:
            revise_retrieval_notebook(notebook)
        elif name == NOTEBOOKS[1]:
            notebook.cells.append(nbformat.v4.new_code_cell(
                "from review_audit import audit_reformulations\naudit_summary, audit_status = audit_reformulations()\ndisplay(pd.Series(audit_status))\ndisplay(audit_summary)\ndisplay(pd.read_csv(RESULT_DIR / 'query_quality_audit_issues.csv'))"))
        elif name == NOTEBOOKS[2]:
            for cell in notebook.cells:
                if cell.cell_type == 'code' and 'MODE =' in cell.source:
                    cell.source = cell.source.replace("MODE = 'pilot'", "MODE = 'formal'")
                    cell.source = cell.source.replace("rag_pipeline.RUN_LABEL = 'v2'", "rag_pipeline.RUN_LABEL = ''")
                    cell.source += "\nRUN_INFERENCE = False\nprint('Canonical run: rag_formal; inference enabled:', RUN_INFERENCE)"
                elif cell.cell_type == 'code' and 'answers = generate(MODE)' in cell.source:
                    cell.source = cell.source.replace('answers = generate(MODE)',
                        "if RUN_INFERENCE:\n    answers = generate(MODE)\nelse:\n    answers = rag_pipeline.export_answers(output_dir)\n    assert len(answers) == len(sample) * 4, 'Enable RUN_INFERENCE to generate missing answers.'")
                elif cell.cell_type == 'code' and cell.source.strip() == 'judge(MODE)':
                    cell.source = "recovered = rag_pipeline.recover_saved_judgments(output_dir)\nprint('Recovered saved JSON:', recovered)\nif RUN_INFERENCE:\n    judge(MODE)\nassert len(list((output_dir / 'judgments').glob('*.json'))) == len(sample) * 4"
                elif cell.cell_type == 'markdown' and 'closeout_experiment2' in cell.source:
                    cell.source = review_instructions('formal', 'rag_formal')
        else:
            for cell in notebook.cells:
                if cell.cell_type == 'code' and 'RUN_GENERATION =' in cell.source:
                    lines = [line for line in cell.source.splitlines() if not line.lstrip().startswith('# RUN_GENERATION')]
                    cell.source = '\n'.join(lines).replace('RUN_GENERATION = True', 'RUN_GENERATION = False')
                elif cell.cell_type == 'code' and 'generated = pb.generate' in cell.source:
                    cell.source = "if RUN_GENERATION:\n    generated = pb.generate(corpus, sample, rankings, output_dir, run_manifest)\nelse:\n    generated = rag.export_answers(output_dir)\nassert len(generated) == len(sample) * 4, 'Enable RUN_GENERATION for missing answers.'\ndisplay(generated[['query_id', 'system', 'answer', 'seconds', 'hit_token_limit']].head())"
                elif cell.cell_type == 'code' and 'rag.judge(mode_key)' in cell.source:
                    cell.source = "print('Recovered JSON:', rag.recover_saved_judgments(output_dir))\nif RUN_GENERATION:\n    rag.judge(mode_key)\nassert len(list((output_dir / 'judgments').glob('*.json'))) == len(sample) * 4\nsummary = pb.report(mode_key)\ndisplay(summary)\ndisplay(pd.read_csv(output_dir / 'judge_coverage.csv'))\ndisplay(pd.read_csv(output_dir / 'answer_paired_differences_ci.csv').head(12))"
            notebook.cells.append(nbformat.v4.new_markdown_cell(review_instructions('policybench_formal_v3', 'rag_policybench_formal_v3')))
        notebook.cells.insert(0, nbformat.v4.new_markdown_cell(
            "**已保存结果复核模式。** 默认读取固定版本的数据与已完成的模型输出，重新计算报表；不会重新运行大模型。改动生成/检索协议时使用新运行目录。人工原始标签保留，`human_review_issues.csv` / `query_quality_audit_issues.csv` 列出需研究者确认的项目。运行前重启内核，使用 Legal FAQ (local GPU) 内核。"))
        notebook.metadata['maintenance_replay_version'] = 1
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                cell.outputs = []
                cell.execution_count = None
        nbformat.validate(notebook)
        nbformat.write(notebook, path)


def review_instructions(mode, directory):
    return (f"## 人工复核与最新校准\n主表：`{directory}/human_review_blinded.csv`。已完成的80条也在同目录 `researcher_review_sample_80.csv` 中；只选一种表修改，避免维护两个不同版本。\n"
            f"若修改80条样本，在新单元运行：\n```python\nfrom review_tools import import_review_sample\ndisplay(import_review_sample('{mode}'))\n```\n"
            "若直接改主表，只需重新运行最后的 import_human_review 单元。导入样本会备份主表，并允许有依据地把旧分数改为空白。\n"
            "自动评分分歧不等于人工错误；请按既定rubric决定。`human_review_issues.csv` 和 `provenance/RESEARCH_PENDING.md` 是待核查清单，不能当成人工已完成的结论。")


def revise_retrieval_notebook(notebook):
    cells = notebook.cells
    cells[2].source = "import importlib.util\nmissing = [name for name in ['datasets', 'rank_bm25', 'sentence_transformers', 'torch', 'numpy', 'pandas', 'matplotlib'] if importlib.util.find_spec(name) is None]\nif missing:\n    raise ImportError('Select the Legal FAQ kernel or see README.md for package versions; missing: ' + ', '.join(missing))\nprint('Dependencies available; no automatic package upgrades.')"
    cells[4].source = cells[4].source.replace('DATASET_REVISION = "main"', 'DATASET_REVISION = "1e6d5276557ee6c05e554290e5270ef59d140111"').replace('MODEL_REVISION = "main"', 'MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"')
    cells[4].source = cells[4].source.replace('# On a repeat study, replace \'main\' with the commit hashes saved in run_metadata.json.', '# 固定数据和模型版本，避免重跑时下载变化后的基准。')
    cells[4].source = cells[4].source.replace('OUTPUT_DIR = Path.cwd() / "legal_faq_results"', "RECOMPUTE_RETRIEVAL = False\nROOT = Path.cwd()\nOUTPUT_DIR = ROOT / ('legal_faq_rebuild' if RECOMPUTE_RETRIEVAL else 'legal_faq_results')\nif RECOMPUTE_RETRIEVAL and (OUTPUT_DIR / 'run_metadata.json').exists():\n    raise ValueError('Choose a new output directory for recomputation.')")
    cells[4].source += "\nfrom project_support import sha256\nif not RECOMPUTE_RETRIEVAL:\n    saved_metadata = json.loads((OUTPUT_DIR / 'run_metadata.json').read_text(encoding='utf-8'))\n    saved_hashes = json.loads((ROOT / 'rag_formal' / 'manifest.json').read_text(encoding='utf-8'))['hashes']\n    for filename, expected in {**saved_metadata['sha256'], **saved_hashes}.items():\n        assert sha256(OUTPUT_DIR / filename) == expected, filename + ' integrity failure'\nprint('Recompute retrieval:', RECOMPUTE_RETRIEVAL)"
    original = cells[6].source
    cells[6].source = "if RECOMPUTE_RETRIEVAL:\n" + indent(original) + "\nelse:\n    dataset_commit = saved_metadata['dataset_revision']\n    model_commit = saved_metadata['model_revision']\n    raw_df = pd.read_csv(OUTPUT_DIR / 'raw.csv')\n    print('Pinned local raw data:', raw_df.shape)\n    display(raw_df.head(3))\n    display(raw_df.dtypes)"
    for index, fragments in [(8, ['duplicate_answers.to_csv', 'corpus.to_csv']), (10, ['queries.to_csv', 'duplicate_queries.to_csv'])]:
        lines = cells[index].source.splitlines()
        cells[index].source = '\n'.join(('if RECOMPUTE_RETRIEVAL:\n    ' + line) if any(line.startswith(fragment) for fragment in fragments) else line for line in lines)
    cells[8].source += "\nif not RECOMPUTE_RETRIEVAL:\n    pd.testing.assert_frame_equal(corpus, pd.read_csv(OUTPUT_DIR / 'corpus.csv', dtype={'doc_id': str}))"
    cells[10].source += "\nif not RECOMPUTE_RETRIEVAL:\n    pd.testing.assert_frame_equal(queries, pd.read_csv(OUTPUT_DIR / 'queries.csv', dtype={'doc_id': str}))"
    for index, method in [(14, 'bm25'), (16, 'dense'), (18, 'hybrid')]:
        original = cells[index].source
        if index == 18:
            split = original.index('started = time.perf_counter()')
            setup, original = original[:split], original[split:]
        else:
            setup = ''
        replay = f"{method}_rankings = np.load(OUTPUT_DIR / '{method}_full_rankings.npy', mmap_mode='r')\n{method}_detail = evaluate_rankings({method}_rankings, '{'Hybrid' if method == 'hybrid' else method.title()}')\ndisplay(summarize({method}_detail).round(4))"
        if index == 16:
            replay += "\ndisplay(pd.read_csv(OUTPUT_DIR / 'truncation_report.csv'))"
        cells[index].source = setup + "if RECOMPUTE_RETRIEVAL:\n" + indent(original) + '\nelse:\n' + indent(replay)
    text = cells[24].source
    start = text.index('pd.DataFrame(timings).to_csv')
    end = text.index('for metric, table in comparison_tables.items():')
    text = text[:start] + 'if RECOMPUTE_RETRIEVAL:\n' + indent(text[start:end]) + '\n' + text[end:]
    start = text.index('rankings_by_method =')
    end = text.index('assert len(overall_results)')
    text = text[:start] + 'if RECOMPUTE_RETRIEVAL:\n' + indent(text[start:end]) + '\n' + text[end:]
    cells[24].source = text
    for cell in cells:
        if cell.cell_type == 'markdown':
            cell.source = cell.source.replace('RQ1:', 'Retrieval subquestion 1:').replace('RQ2:', 'Retrieval subquestion 2:').replace('RQ3:', 'Retrieval subquestion 3:')


def indent(text):
    return '\n'.join('    ' + line if line else '' for line in text.splitlines())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'receipts', 'notebooks'])
    parser.add_argument('--backup')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.backup)
    elif args.stage == 'receipts':
        receipts()
    else:
        notebooks()
