# MaintQA Benchmark Splits

MaintQA is a question-answering benchmark for maintainability engineering
design handbooks, comprising **1,812 unique samples** (1,114 training + 698
evaluation) across eight splits. The paper refers to the splits with a
`test1_` prefix (e.g., `test1_mil470a_200`); the files here use the short
names shown below.

| File (this folder) | Paper name | n | Task |
|---|---|---|---|
| `train_mil470a_highconf.jsonl` | `train_mil470a_highconf` | 1,114 | Task 1 training |
| `test_mil470a_200.jsonl` | `test1_mil470a_200` | 200 | Task 1 primary (9-category balanced) |
| `test_mil470a_500.jsonl` | `test1_mil470a_500` | 500 | Task 1 auxiliary; fully contains the 200 split |
| `test_mil470a_88.jsonl` | `test1_mil470a_88` | 88 | Task 1 additional, numerically dense (12-category balanced) |
| `test_dod791_30.jsonl` | `test1_dod791_30` | 30 | Task 1 cross-handbook (DOD-HDBK-791) |
| `task2_category_identification.jsonl` | `task2_category_id` | 30 | Task 2 multi-label category identification |
| `task3_open_generation.jsonl` | `task3_open_gen` | 20 | Task 3 open-ended design recommendation |

The cross-language split `test1_gjb451b_30` (30 samples over GJB 451B) is not
included in this repository because redistribution of the source document is
subject to access restrictions; it is available from the authors on request
(see the Data Availability Statement of the paper).

## Task 1 schema (train / test splits)

| Field | Description |
|---|---|
| `id` | Unique sample identifier |
| `query` | Question text |
| `query_en` | English rendering of the query |
| `answer` | Reference answer (canonical clause sentence or figure) |
| `answer_source` | Provenance object: source `page`, section, and guideline identifiers |
| `source_id` | Identifier of the source guideline/clause in the handbook |
| `source_record_type` | Type of the source record (guideline, body clause, etc.) |
| `constraint_type` / `constraint_type_id` | Maintainability design-technique category label (12-class system of GJB 368B §4.6) |
| `difficulty` | Coarse difficulty tier assigned at construction |
| `equipment_types` | Equipment domain label(s) used for stratification |
| `keywords` | Terms used for category matching |
| `aliases`, `alias_count`, `mapping_candidates`, `mapping_method` | Category-mapping audit fields (present in the 88/500/train splits) |
| `related_constraints` | Related numerical constraints, where applicable (88/train splits) |

## Task 2 schema (`task2_category_identification.jsonl`)

| Field | Description |
|---|---|
| `id`, `query`, `task` | Identifier, concatenated multi-guideline design description, task tag |
| `gt_categories`, `gt_category_ids`, `n_gt` | Ground-truth multi-label category set and its size |
| `sources` | Source guideline identifiers combined into the sample |

## Task 3 schema (`task3_open_generation.jsonl`)

| Field | Description |
|---|---|
| `id`, `title`, `query`, `task` | Identifier, scenario title, open-ended design request, task tag |
| `scenario_desc`, `family`, `n_suggestions` | Scenario description, equipment family, requested number of suggestions |

## Overlap note

`test_mil470a_500.jsonl` fully contains all entries of
`test_mil470a_200.jsonl`; the 1,812 total counts each sample once.
`test_mil470a_88.jsonl` partially overlaps the training pool in source
guidelines and is used in the paper only for within-family analyses where
every compared configuration is evaluated on the identical split.
