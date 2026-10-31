# Provenance and historical evidence

- `backup_location.json`: external full-project snapshot, verified before maintenance; its `backup_manifest.json` contains original file hashes.
- `original_code`: original source snapshots required by historical experiment manifests.
- `source_compatibility.json`: explicit source transition for reporting/parser maintenance. It does not permit changed dataset/model/retrieval/generation parameters.
- `consolidation.json`: latest E2 researcher labels merged into `rag_formal`, after exact identity checks.
- `alternate_run_v2`: all v2 answer and judgment records packed as JSONL, preserving differing timing/memory measurements and original human sheets. They are not extra independent query samples and do not enter canonical statistical averages.
- `legacy_tools`: earlier postprocessing implementations retained for historical parser checksums. Run current root tools, not these historical copies.
- `researcher_confirmation_needed.csv`: assistant review suggestions. Current source-verification gaps are recorded in the root `PROJECT_STATUS.md`; neither is fabricated human annotation or verified government evidence.
- `deployment_validation_pilot_snapshot.json`: historical deployment and pilot check. Its old formal-run status is not the current experiment status; use the root `project_validation.json` and completed run records.

Original raw model attempts remain in every judgment record. Historical parser-recovery receipts may refer to earlier source hashes; use the corresponding archived implementation and the current verification report to distinguish versions.
