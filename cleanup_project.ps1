$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\')
$prefix = $projectRoot + '\'
if (Test-Path -LiteralPath (Join-Path $projectRoot 'cleanup_manifest.json')) {
    throw 'Cleanup has already been completed. Existing cleanup_manifest.json is the historical receipt; do not run again.'
}
$backup = (Get-Content -LiteralPath (Join-Path $projectRoot 'provenance\backup_location.json') -Raw | ConvertFrom-Json).path
if ([System.IO.Path]::GetFullPath($backup).StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Backup must be outside project.' }
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'project_validation.json'))) { throw 'Validate the project before cleanup.' }
$manifest = Get-Content -LiteralPath (Join-Path $backup 'backup_manifest.json') -Raw | ConvertFrom-Json
$hashes = @{}
foreach ($entry in $manifest) { $hashes[$entry.path] = $entry.sha256 }
$names = @('__pycache__', '.ipynb_checkpoints', '.virtual_documents', 'anaconda_projects',
    'rag_formal_v2', 'rag_policybench_pilot_v1',
    'Experiment_2_RAG_Generation_Evaluation_Final.ipynb', 'Experiment_2_RAG_Generation_Evaluation_Final.html',
    'Experiment_3_PolicyBench_Validation_Final.ipynb', 'Experiment_3_PolicyBench_Validation_Final.html',
    'closeout_experiment2.py', 'repair_experiment3_judgments.py',
    'rag_formal\EXPERIMENT_2_CLOSEOUT.md', 'rag_formal\error_analysis_current.csv',
    'rag_policybench_formal_v3\error_analysis_current.csv')
$duplicates = Join-Path $projectRoot 'rag_formal_v2'
if (Test-Path -LiteralPath $duplicates) {
    $source = @(Import-Csv -LiteralPath (Join-Path $duplicates 'human_review_blinded.csv'))
    $canonical = @(Import-Csv -LiteralPath (Join-Path $projectRoot 'rag_formal\human_review_blinded.csv'))
    $difference = Compare-Object -ReferenceObject $source -DifferenceObject $canonical -Property review_id,human_correctness,human_relevance,human_faithfulness,human_notes
    if ($difference) { throw 'Human annotations have not been fully consolidated.' }
    foreach ($kind in @('answers', 'judgments')) {
        $archive = Join-Path $projectRoot "provenance\alternate_run_v2\$kind.jsonl"
        if (-not (Test-Path -LiteralPath $archive)) { throw 'Alternate run records not archived.' }
        if (@(Get-Content -LiteralPath $archive).Count -ne 1600) { throw 'Alternate run archive is incomplete.' }
    }
}
$targets = @()
$records = @()
foreach ($name in $names) {
    $candidate = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $name))
    if (-not $candidate.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe cleanup target: $candidate" }
    if (-not (Test-Path -LiteralPath $candidate)) { continue }
    $resolved = (Resolve-Path -LiteralPath $candidate).Path
    if (-not $resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Resolved target outside project: $resolved" }
    $item = Get-Item -LiteralPath $resolved -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Refusing to traverse a reparse point.' }
    $children = if ($item.PSIsContainer) { @(Get-ChildItem -LiteralPath $resolved -Recurse -Force) } else { @($item) }
    if (@($children | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) { throw 'Reparse point inside cleanup target.' }
    $files = @($children | Where-Object { -not $_.PSIsContainer })
    $cache = $name -in @('__pycache__', '.ipynb_checkpoints', '.virtual_documents')
    foreach ($file in $files) {
        $relative = $file.FullName.Substring($prefix.Length)
        $digest = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        if (-not $cache) {
            if (-not $hashes.ContainsKey($relative) -or $hashes[$relative] -ne $digest) { throw "File changed since backup: $relative" }
            $copy = Join-Path (Join-Path $backup 'project') $relative
            if ((Get-FileHash -LiteralPath $copy -Algorithm SHA256).Hash.ToLowerInvariant() -ne $digest) { throw "Backup verification failed: $relative" }
        }
        $records += [pscustomobject]@{path=$relative; bytes=$file.Length; sha256=$digest; reason= $(if($cache){'regenerable cache'}else{'superseded; verified backup and canonical/archive retained'})}
    }
    $targets += $resolved
}
$receipt = [pscustomobject]@{backup=$backup; files_removed=$records.Count; bytes_removed=($records|Measure-Object bytes -Sum).Sum; targets=$targets; files=$records}
$receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $projectRoot 'cleanup_manifest.json') -Encoding UTF8
foreach ($target in $targets) { Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction Stop }
Write-Output "Removed $($records.Count) verified unnecessary files from $($targets.Count) targets. Backup: $backup"
