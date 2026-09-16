# StructStd-RAG

A Structure-Aware Retrieval-Augmented Generation Framework for Normative Technical Documents.

This repository contains the official implementation of **StructStd-RAG**, comprising three core modules:

- **StdChunk**: Structure-aware three-layer semantic indexing (Section → Paragraph → Constraint blocks)
- **CatRetriever**: Category-aware hybrid retrieval with Dense+BM25 recall and taxonomy-guided reranking
- **BoundGrounder**: Parameter-bound-grounded generation verification via source tracing, physical plausibility checking, multi-source consistency assessment, and confidence synthesis

## Repository Structure

```
structstd_rag/
├── src/
│   ├── chunk/          # Document chunking strategies (StdChunk, recursive, fixed, semantic, etc.)
│   ├── retriever/      # Retrieval strategies (CatRetriever, BM25, Dense, Hybrid, HyDE, etc.)
│   ├── generator/      # Generation + verification (BoundGrounder, direct, Self-RAG, CRAG, etc.)
│   ├── pipeline/       # End-to-end RAG pipelines (StructStdRAG, Naive, Dense, DSR, etc.)
│   ├── evaluation/     # Metrics (Tuple-F1, hallucination rate, recall@K, MRR, NDCG, etc.)
│   ├── data/           # Document loaders, preprocessors, MaintQA dataset builder
│   └── utils/          # Configuration, constraint taxonomy, LLM client, logger
├── experiments/        # Experiment scripts for reproducing paper results
├── scripts/            # Dataset building scripts
├── data/
│   ├── maintqa/        # MaintQA benchmark dataset (MIL-HDBK-470A splits)
│   ├── mil470a/        # MIL-HDBK-470A page-level corpus (JSONL)
│   ├── milstd1472f/    # MIL-STD-1472F page-level corpus (JSONL)
│   └── param_bounds/   # 171-parameter bound table (final release version)
├── README.md
└── requirements.txt
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export DASHSCOPE_API_KEY="your-api-key"

# Run core module test
python test_core.py

# Run full pipeline demo
python quickstart.py
```

## MaintQA Benchmark

MaintQA is a question-answering benchmark for maintainability engineering design handbooks, comprising 1,812 samples (1,114 training + 698 evaluation) across eight splits. The files here cover the MIL-HDBK-470A splits (`train_mil470a_highconf`, `test_mil470a_88/200/500`, `task2`, `task3`) and the cross-handbook DOD-HDBK-791 split (`test_dod791_30`). See `data/maintqa/README.md` for the split inventory, field-level schema, and overlap notes; the cross-language GJB 451B split is excluded due to source-document redistribution restrictions.

## Cross-Lingual Support (CJK Regex Patterns)

All code, comments, and prompts in this repository are in English. However, a small number of regular expressions intentionally retain CJK (Chinese) character patterns. **These are functional, not leftovers**: they enable the zero-shot cross-lingual generalization experiment (S3b), which evaluates the pipeline on **GJB 451B** — a Chinese-language military standard for equipment quality characteristics terminology — without any retraining or code changes. They also keep the chunker, retriever, and metrics bilingual-robust for other Chinese normative documents.

| Location | Pattern purpose |
|---|---|
| `src/chunk/std_chunk.py` | Detect Chinese section headings (`第X章/节/条`), constraint sentences (`不大于/至少` + units such as `步`), and cross-references (`见表/见§`) so StdChunk can build the three-layer index over Chinese standards |
| `src/data/preprocessor.py` | Chinese chapter-heading, table-caption, and cross-reference detection during preprocessing |
| `src/retriever/bm25_retriever.py` | Character-level CJK tokenization (`[一-鿿]`) so BM25 can index and retrieve Chinese corpus pages |
| `src/evaluation/metrics.py` | Extract (value, unit) tuples with Chinese units (`步`, `小时`) and tokenize CJK gold answers for partial-match scoring |
| `src/generator/bound_grounder.py` | Constraint-value extraction supports Chinese units and full-width punctuation in answers |
| `experiments/build_gjb451b_qa.py` | Parse the Chinese GJB 451B text into term–definition QA pairs (`test_gjb451b_30.jsonl`) |
| `experiments/run_s3a_*.py` / `run_s3b_*.py` | `answer_coverage` tokenizer counts CJK tokens in gold answers (required for the GJB 451B split; a no-op for English splits) |

If you do not need the cross-lingual experiment, these patterns are inert on English-only corpora and can be left as-is. (Note: this bundle ships the MIL corpora only; the GJB 451B corpus used by the cross-lingual experiment is not included.) The constraint taxonomy additionally carries bilingual keyword lists (72 Chinese + 55 English) used by the rule-based category matcher.

## Revision Additions (v2)

The following artifacts were added in the revised version:

- `experiments/run_e5_ablation_v2.py` — expanded seven-configuration ablation (full + three single-module + three pairwise removals) with multi-metric evaluation and paired bootstrap tests; supersedes the original `run_e5_ablation.py`. Backs the revised ablation table and Fig. 6.
- `experiments/judge_e5v2.py` (+ `experiments/run_llm_judge_v2.py`) — blind Qwen-Max LLM-as-Judge re-evaluation of the seven ablation configurations (v2 rubric).
- `experiments/run_s7_scaling.py` — corpus-scaling measurements (25/50/75/100% of MIL-HDBK-470A: chunks, indexing time, retrieval latency, index memory). Backs the scaling table in Section 5.8.
- `scripts/extract_param_bounds.py` + `data/param_bounds/` — the complete **171-parameter bound table** (`param_bounds_table.csv`), extracted from MIL-HDBK-470A via the semi-automatic pipeline (LLM extraction → rule-based filtering → aggregation; see `data/param_bounds/README.md`).

## Citation

If you use this code or dataset, please cite our paper:

```
[Citation to be added upon publication]
```

## License

Code: MIT (see [LICENSE](LICENSE)). Data (MaintQA splits, page-level corpus extracts, and the parameter-bound table): CC-BY-4.0.
