# Description of Supplementary Files

## Supplementary File 1: Code (structstd_rag_code.zip)

The supplementary code archive contains the complete implementation of the StructStd-RAG framework described in the manuscript, organized as follows:

**Core framework (`src/`)**
- `src/chunk/std_chunk.py` — StdChunk: three-layer structure-aware semantic indexing (Section → Paragraph → Constraint blocks) with table context preservation, cross-reference graph construction, and domain taxonomy labeling.
- `src/retriever/cat_retriever.py` — CatRetriever: category-aware hybrid retrieval with Dense+BM25 recall and taxonomy-guided reranking with equipment-type domain adaptation.
- `src/generator/bound_grounder.py` — BoundGrounder: four-step generation verification (source tracing → physical plausibility checking against a 171-parameter bound table → multi-source consistency assessment → confidence synthesis with graceful fallback).
- `src/pipeline/structstd_rag.py` — StructStdRAGPipeline: end-to-end composition of the three modules.
- Supporting modules: chunking strategies, retrieval backends (BM25, Dense, Hybrid, HyDE, Multi-Query), generator variants (Self-RAG, CRAG, RankRAG), evaluation metrics (Tuple-F1, Hallucination Rate, Recall@K, MRR, NDCG), document loaders, and configuration.

**Revision additions (`experiments/`, `scripts/`)**
- `experiments/run_e5_ablation_v2.py` — expanded seven-configuration ablation with multi-metric evaluation and paired bootstrap significance; supersedes the original single-purpose ablation script (removed).
- `experiments/judge_e5v2.py` and `experiments/run_llm_judge_v2.py` — blind Qwen-Max LLM-as-Judge re-evaluation of the seven ablation configurations.
- `experiments/run_s7_scaling.py` — corpus-scaling measurements (25/50/75/100% of MIL-HDBK-470A).
- `scripts/extract_param_bounds.py` — semi-automatic extraction pipeline (LLM extraction → rule-based filtering → aggregation) that produced the complete 171-parameter bound table.
- `results/` — result artifacts backing the revised ablation and scaling tables.

**Experiment scripts (`experiments/`)**
Scripts for reproducing all tables and figures in the manuscript: main comparison (Table 2), LLM-as-Judge evaluation (Table 3), LLM-free retrieval (Table 4), cross-equipment/manual/language generalization (Tables 5–7), numerical reliability (Table 8), robustness analysis (Tables 9–11), ablation study (Table 12), retrieval mode analysis (Figure 4), hyperparameter sensitivity, efficiency comparison (Table 13), and task expansions (Tables 14–15).

**Dataset construction scripts (`scripts/`)**
Scripts for building the MaintQA benchmark from MIL-HDBK-470A structured data: PDF extraction, candidate generation, high-confidence filtering, deduplication, and stratified sampling.

The code is written in Python 3.12 with English comments throughout. Dependencies are listed in `requirements.txt`.

## Supplementary File 2: MaintQA Benchmark and Document Corpus (maintqa_data.zip)

The data archive contains the MaintQA benchmark dataset and the document corpus used in all experiments:

**MaintQA benchmark (`data/maintqa/`)**
- `train_mil470a_highconf.jsonl` (1,114 samples) — Training split with semi-automatically generated QA pairs from MIL-HDBK-470A, manually spot-checked.
- `test_mil470a_200.jsonl` (200 samples) — Primary in-domain test split, 9-category balanced sampling.
- `test_mil470a_88.jsonl` (88 samples) — Additional numerically intensive test split, 12-category balanced sampling.
- `test_mil470a_500.jsonl` (500 samples) — Auxiliary evaluation set (200 original + 300 supplementary) for LLM-as-Judge evaluation.
- `task2_category_identification.jsonl` (30 samples) — Multi-label category identification task.
- `task3_open_generation.jsonl` (20 samples) — Open-ended design recommendation generation task.

Each sample includes the query (Chinese and English), gold answer, constraint type, answer source with document/section/page/quote references, difficulty level, equipment type, and relevant chunk identifiers. The complete study uses 1,812 unique samples (1,114 training + 698 evaluation across eight splits); this revised bundle ships the MIL-HDBK-470A splits and corpora (MIL-HDBK-470A, MIL-STD-1472F). The cross-handbook and cross-language transfer splits (DOD-HDBK-791, GJB 451B) and their source corpora are not included in this bundle.

**Document corpus (`data/{mil470a,milstd1472f}/`)**
Extracted page-level text (JSONL format) from the MIL-family maintainability engineering standards used as the retrieval corpus: MIL-HDBK-470A (709 pp) and MIL-STD-1472F (219 pp). These documents were obtained through legally accessible U.S. Government sources and are in the public domain.

**Parameter-bound table (`data/param_bounds/param_bounds_table.csv`, 171 parameters)**
The complete 171-parameter physical plausibility bound table, extracted from MIL-HDBK-470A with the semi-automatic pipeline in `scripts/extract_param_bounds.py` (LLM extraction → rule-based filtering → aggregation; raw extraction records included for provenance, see `data/param_bounds/README.md`). Each entry specifies a parameter name, unit, observed bound interval, supporting-guideline counts, and page-level source references. The runtime verification module additionally ships a hand-verified core subset of bounds in `src/utils/constraint_taxonomy.py`.

---

These supplementary files contain all code, data, and configuration necessary to fully reproduce the experimental results reported in the manuscript. The raw numerical measurements supporting Tables 2–15 and Figures 3–6 are generated by running the experiment scripts against the provided datasets.
