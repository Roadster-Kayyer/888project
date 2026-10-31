"""Explicit compatibility checks for preserved experiments after reporting maintenance."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def compatible_manifest(previous, current, kind):
    if previous == current:
        return previous
    field = 'code_sha256' if kind == 'retrieval' else 'code_hashes'
    if {key: value for key, value in previous.items() if key != field} != {
            key: value for key, value in current.items() if key != field}:
        raise ValueError('Experiment configuration changed. Use a new run label/data directory.')
    receipt_path = ROOT / 'provenance' / 'source_compatibility.json'
    if not receipt_path.exists():
        raise ValueError('Source code changed without an explicit compatibility receipt.')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    for name, expected in receipt['current_source_hashes'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError('Source compatibility receipt is stale: ' + name)
    original = receipt['original_source_hashes']
    for name, expected in original.items():
        if sha256(ROOT / 'provenance' / 'original_code' / name) != expected:
            raise ValueError('Original source snapshot failed verification: ' + name)
    if kind == 'retrieval':
        valid = (previous[field] == original['policybench_validation.py'] and
                 current[field] == receipt['current_source_hashes']['policybench_validation.py'])
    else:
        valid = all(previous[field].get(name) == original.get(name) and
                    current[field].get(name) == receipt['current_source_hashes'].get(name)
                    for name in previous[field]) and set(previous[field]) == set(current[field])
    if not valid:
        raise ValueError('Source transition is not covered by the compatibility receipt.')
    print('Reusing original experiment under documented reporting/parser maintenance compatibility.')
    return previous
