"""Integrity and statistical checks for the canonical saved thesis experiments."""

import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from project_support import ROOT, sha256
from judge_parsing import parse_judgment, validate_judgment


def check_run(name):
    directory = ROOT / name
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    sample = pd.read_csv(directory / 'sample_queries.csv', dtype={'query_id': str})
    assert sha256(directory / 'sample_queries.csv') == manifest['sample_sha256']
    expected = {(query_id, system) for query_id in sample.query_id for system in ['No-RAG', 'BM25-RAG', 'Dense-RAG', 'RRF-RAG']}
    answers = [json.loads(path.read_text(encoding='utf-8')) for path in (directory / 'answers').glob('*.json')]
    assert len(answers) == len(expected) == manifest['n_queries'] * 4
    assert {(item['query_id'], item['system']) for item in answers} == expected
    judgments = list((directory / 'judgments').glob('*.json'))
    assert len(judgments) == len(expected)
    for answer in answers:
        path = directory / 'judgments' / f"{answer['query_id']}_{answer['system']}.json"
        judgment = json.loads(path.read_text(encoding='utf-8'))
        assert judgment['status'] == 'ok' and judgment['scores']
        assert validate_judgment(dict(judgment['scores']), bool(answer['context'])) == judgment['scores']
        assert (answer['system'] != 'No-RAG') == bool(answer['context'])
        assert isinstance(answer['answer'], str) and answer['answer'].strip()
    frame = pd.read_csv(directory / 'answer_evaluation.csv', dtype={'query_id': str})
    assert len(frame) == len(expected) and frame.judge_status.eq('ok').all()
    assert not frame.duplicated(['query_id', 'system']).any()
    summary = pd.read_csv(directory / 'answer_quality_summary.csv').set_index('system')
    intervals = pd.read_csv(directory / 'answer_bootstrap_ci.csv')
    for row in intervals[intervals.query_type.eq('overall')].itertuples():
        actual = summary.loc[row.method, [row.metric, row.metric + '_ci_low', row.metric + '_ci_high']].astype(float)
        np.testing.assert_allclose(actual, [row.estimate, row.ci_low, row.ci_high])
        assert int(summary.loc[row.method, 'n_faq_groups_' + row.metric]) == row.n_groups
    review = pd.read_csv(directory / 'human_review_blinded.csv', keep_default_na=False, dtype=str)
    keys = pd.read_csv(directory / 'human_review_key.csv', dtype=str)
    assert len(review) == len(expected) and review.review_id.is_unique
    joined = review.merge(keys, on='review_id', validate='one_to_one').merge(
        frame, on=['query_id', 'system'], validate='one_to_one', suffixes=('', '_automatic'))
    for field in ['query', 'context', 'answer', 'reference_answer']:
        assert joined[field].fillna('').eq(joined[field + '_automatic'].fillna('')).all()
    calibration = pd.read_csv(directory / 'judge_human_calibration.csv').set_index('metric')
    for metric in ['correctness', 'relevance', 'faithfulness']:
        human = pd.to_numeric(joined['human_' + metric].replace('', np.nan), errors='raise')
        valid = human.notna() & joined[metric].notna()
        if metric == 'faithfulness':
            valid &= joined.system.ne('No-RAG')
        assert int(calibration.loc[metric, 'n_paired_reviews']) == valid.sum()
        if valid.any():
            np.testing.assert_allclose(calibration.loc[metric, 'exact_agreement'], (human[valid] == joined.loc[valid, metric]).mean())
            np.testing.assert_allclose(calibration.loc[metric, 'mean_absolute_difference'], (human[valid] - joined.loc[valid, metric]).abs().mean())
    return {'n_queries': len(sample), 'n_answers': len(frame), 'n_valid_judgments': len(frame),
            'n_reviewed_correctness': int(review.human_correctness.ne('').sum())}


def check_rankings(folder, hashes):
    directory = ROOT / folder
    corpus = pd.read_csv(directory / 'corpus.csv', dtype={'doc_id': str})
    queries = pd.read_csv(directory / 'queries.csv', dtype={'doc_id': str})
    for name, digest in hashes.items():
        assert sha256(directory / name) == digest, f'{folder}/{name} hash changed'
    rankings = {}
    for method in ['bm25', 'dense', 'hybrid']:
        values = np.load(directory / (method + '_full_rankings.npy'), mmap_mode='r')
        assert values.shape == (len(queries), len(corpus))
        for start in range(0, len(values), 128):
            assert np.all(np.sort(values[start:start + 128], axis=1) == np.arange(len(corpus)))
        rankings[method] = values
    for index in np.linspace(0, len(queries) - 1, 25, dtype=int):
        weights = 1 / (60 + np.arange(1, len(corpus) + 1))
        scores = np.zeros(len(corpus))
        scores[rankings['bm25'][index]] += weights
        scores[rankings['dense'][index]] += weights
        assert np.array_equal(np.argsort(-scores, kind='stable'), rankings['hybrid'][index])
    return {'documents': len(corpus), 'queries': len(queries), 'all_rankings_valid_permutations': True}


def main():
    from verify_study import run_checks
    run_checks()
    valid = '{"correctness":2,"relevance":2,"faithfulness":2,"reason":"supported"}'
    assert parse_judgment(valid[:-1] + ',}', True)['correctness'] == 2
    for text in [valid.replace('"correctness":2', '"correctness":0,"correctness":2'),
                 valid + valid.replace('"correctness":2', '"correctness":0')]:
        try:
            parse_judgment(text, True)
        except ValueError:
            pass
        else:
            raise AssertionError('Ambiguous judge output accepted.')
    primary = json.loads((ROOT / 'rag_formal' / 'manifest.json').read_text(encoding='utf-8'))
    secondary = json.loads((ROOT / 'rag_policybench_formal_v3' / 'manifest.json').read_text(encoding='utf-8'))
    result = {'retrieval_legal': check_rankings('legal_faq_results', primary['hashes']),
              'retrieval_policybench': check_rankings('policybench_data', secondary['data_hashes']),
              'formal_legal': check_run('rag_formal'), 'formal_policybench': check_run('rag_policybench_formal_v3')}
    notebooks = ['Legal_FAQ_Retrieval_Experiment.ipynb', 'Experiment_1_Retrieval_Supplement.ipynb',
                 'Experiment_2_RAG_Generation_Evaluation.ipynb', 'Experiment_3_PolicyBench_Validation.ipynb']
    for name in notebooks:
        notebook = nbformat.read(ROOT / name, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == 'code' and cell.source.strip():
                assert cell.execution_count is not None, name + ' has unexecuted cells'
                assert not any(output.output_type == 'error' for output in cell.outputs)
        assert (ROOT / 'html' / name.replace('.ipynb', '.html')).exists()
    result['notebooks_executed_without_errors'] = notebooks
    result['scope'] = 'Computational integrity only. Researcher judgment, source verification and thesis claims remain subject to review.'
    (ROOT / 'project_validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
