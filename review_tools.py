"""Import an explicitly edited sample into its matching master review sheet."""

from datetime import datetime, timezone
from pathlib import Path
import shutil

import pandas as pd

import rag_pipeline as rag


def import_review_sample(mode='formal', sample_path=None):
    output = rag.study_dir(mode)
    sample_path = Path(sample_path or output / 'researcher_review_sample_80.csv')
    sample = pd.read_csv(sample_path, keep_default_na=False, dtype=str)
    master_path = output / 'human_review_blinded.csv'
    master = pd.read_csv(master_path, keep_default_na=False, dtype=str).set_index('review_id')
    if not sample.review_id.is_unique or not sample.review_id.isin(master.index).all():
        raise ValueError('Sample review IDs do not belong to this run.')
    sample = sample.set_index('review_id')
    for field in ['query', 'reference_answer', 'context', 'answer']:
        if not sample[field].eq(master.loc[sample.index, field]).all():
            raise ValueError('Sample source text differs from this run: ' + field)
    scores = ['human_correctness', 'human_relevance', 'human_faithfulness']
    if not sample[scores].isin(['', '0', '1', '2']).all().all():
        raise ValueError('Scores must be 0, 1, 2, or blank.')
    selected = sample[scores].ne('').any(axis=1)
    sample = sample.loc[selected]
    if sample.empty:
        raise ValueError('There are no reviewed rows to import.')
    if (sample.context.eq('') & sample.human_faithfulness.ne('')).any():
        raise ValueError('Without evidence, human faithfulness must be blank.')
    columns = scores + ['human_notes']
    changed = master.loc[sample.index, columns].ne(sample[columns]).any(axis=1)
    if changed.any():
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        backup = rag.ROOT / 'provenance' / 'human_review_versions' / output.name / (stamp + '.csv')
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(master_path, backup)
        master.loc[sample.index, columns] = sample[columns]
        temporary = master_path.with_suffix('.tmp')
        master.reset_index().to_csv(temporary, index=False, encoding='utf-8-sig')
        temporary.replace(master_path)
    return rag.import_human_review(mode)
