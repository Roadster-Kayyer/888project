"""Validate saved researcher annotations without inventing or changing human judgments."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
FIELDS = ['meaning_preserved_yes_no_uncertain', 'semantic_drift', 'invalid']


def audit_reformulations(output_dir=None, data_dir=None):
    output_dir = Path(output_dir or ROOT / 'retrieval_supplement')
    data_dir = Path(data_dir or ROOT / 'legal_faq_results')
    frame = pd.read_csv(output_dir / 'query_quality_audit_100.csv', keep_default_na=False, dtype=str)
    queries = pd.read_csv(data_dir / 'queries.csv', keep_default_na=False, dtype=str)
    corpus = pd.read_csv(data_dir / 'corpus.csv', keep_default_na=False, dtype=str)
    expected = pd.concat([queries[queries.query_type.eq(style)].sample(n=25, random_state=42)
                          for style in ['formal', 'casual', 'search', 'hard']])
    expected['original_question'] = expected.doc_id.map(
        queries[queries.query_type.eq('original')].set_index('doc_id')['query'])
    expected['reference_answer'] = expected.doc_id.map(corpus.set_index('doc_id').text)
    if not frame.query_id.is_unique or set(frame.query_id) != set(expected.query_id):
        raise ValueError('Reformulation audit sample IDs changed.')
    aligned = frame.set_index('query_id').loc[expected.query_id]
    for column in ['doc_id', 'query_type', 'query', 'original_question', 'reference_answer']:
        if not aligned[column].tolist() == expected[column].tolist():
            raise ValueError('Reformulation audit source changed: ' + column)
    allowed = ['', 'yes', 'no', 'uncertain']
    if not frame[FIELDS].isin(allowed).all().all():
        raise ValueError('Audit labels must be yes/no/uncertain or blank.')
    reviewed = frame[FIELDS].ne('').all(axis=1)
    issues = []
    for row in frame.to_dict('records'):
        reasons = []
        if row[FIELDS[0]] == 'yes' and row['semantic_drift'] == 'yes':
            reasons.append('preserved=yes conflicts with substantive drift=yes; researcher confirmation needed')
        if row[FIELDS[0]] == 'no' and row['semantic_drift'] == 'no':
            reasons.append('meaning not preserved but no drift recorded; explain the distinction')
        exceptional = (row[FIELDS[0]] in ['no', 'uncertain'] or
                       any(row[name] in ['yes', 'uncertain'] for name in FIELDS[1:]))
        if exceptional and not row['notes'].strip():
            reasons.append('exceptional label needs a reason')
        for reason in reasons:
            issues.append({'query_id': row['query_id'], 'query_type': row['query_type'],
                           'issue': reason, 'status': 'researcher_confirmation_required'})
    pd.DataFrame(issues, columns=['query_id', 'query_type', 'issue', 'status']).to_csv(
        output_dir / 'query_quality_audit_issues.csv', index=False, encoding='utf-8-sig')
    summary = frame.melt(id_vars=['query_type'], value_vars=FIELDS, var_name='field', value_name='label')
    summary['label'] = summary.label.replace('', 'unreviewed')
    summary = summary.groupby(['query_type', 'field', 'label']).size().rename('count').reset_index()
    summary.to_csv(output_dir / 'query_quality_audit_summary.csv', index=False, encoding='utf-8-sig')
    status = {'n_sample': len(frame), 'n_completed_labels': int(reviewed.sum()), 'n_issues': len(issues),
              'status': ('labels_complete_pending_consistency_review' if issues else 'labels_complete')
              if reviewed.all() else 'partially_reviewed' if reviewed.any() else 'awaiting_human_review',
              'annotations_changed': False, 'sample_seed': 42, 'per_type': 25}
    (output_dir / 'query_quality_audit_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    protocol_path = output_dir / 'protocol.json'
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
        protocol['audit_review_status'] = status['status']
        protocol['audit_review_counts'] = {key: status[key] for key in ['n_sample', 'n_completed_labels', 'n_issues']}
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    return summary, status


def review_flags(output_dir, merged):
    issues = []
    for row in merged.to_dict('records'):
        reviewed = any(pd.notna(row['human_' + name]) for name in ['correctness', 'relevance', 'faithfulness'])
        if not reviewed:
            continue
        note = row.get('human_notes', '')
        if pd.isna(note) or not str(note).strip():
            issues.append({'review_id': row['review_id'], 'issue': 'reviewed_without_notes',
                           'recommendation': 'Record reasons for disagreements and N/A judgments; routine agreements may stay blank.'})
        for metric in ['correctness', 'relevance', 'faithfulness']:
            human = row['human_' + metric]
            auto = row[metric]
            if pd.notna(human) and pd.notna(auto) and abs(human - auto) >= 1:
                issues.append({'review_id': row['review_id'], 'issue': metric + '_disagreement',
                               'recommendation': 'Inspect against rubric; do not change the human score just to match the model.'})
        if pd.notna(row['human_faithfulness']) and pd.isna(row['faithfulness']) and row['system'] != 'No-RAG':
            issues.append({'review_id': row['review_id'], 'issue': 'faithfulness_applicability_disagreement',
                           'recommendation': 'Check whether this is a pure abstention or includes factual claims. Neither label is automatically accepted.'})
    flags = pd.DataFrame(issues, columns=['review_id', 'issue', 'recommendation'])
    flags.to_csv(Path(output_dir) / 'human_review_issues.csv', index=False, encoding='utf-8-sig')
    return flags
