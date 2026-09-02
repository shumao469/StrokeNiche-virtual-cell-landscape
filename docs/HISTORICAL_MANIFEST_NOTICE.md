# Historical manifest notice

`code_manifest.tsv`, `figure_inventory.tsv`, `figure_manifest.tsv`,
`input_data_manifest.tsv`, `table_manifest.tsv`, `step_ledger.tsv`, and
`release_summary.json` were inherited from the earlier repository skeleton. They
are historical snapshots and may reference absent, duplicated, zero-filled, or
superseded artifacts. They are not proof that the current branch is reconstructible.

For the selected scripts added in this branch, use `curated_code_manifest.tsv`,
which records both source and released SHA-256 values plus modification notes.
For local visualizer data, use the generated `visualizer_manifest.json` beside the
isolated data bundle. Recompute and verify hashes before any tagged release.
