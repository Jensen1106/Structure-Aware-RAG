"""Smoke test: verify all translated modules import, prompts format, and the
LLM-free pipeline components (StdChunk + BM25 + taxonomy + metrics) run
end-to-end on a synthetic English document.

Run: python test_smoke_translation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

FAILED = []


def check(name, fn):
    try:
        fn()
        print(f"  [PASS] {name}")
    except Exception as e:
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
        FAILED.append(name)


# ── 1. Config paths are project-relative ─────────────────────────────────
def t_config():
    from src.config import (PROJECT_ROOT, DATA_DIR, RAW_DIR, MAINTQA_DIR,
                            RESULTS_DIR, LLM_CONFIG, DEFAULT_CONFIG)
    assert PROJECT_ROOT == Path(__file__).parent.resolve() or \
           PROJECT_ROOT.resolve() == Path(__file__).parent.resolve(), PROJECT_ROOT
    assert DATA_DIR == PROJECT_ROOT / "data"
    assert RAW_DIR == DATA_DIR / "raw"
    assert DEFAULT_CONFIG.chunk.maint_paragraph_max == 512


# ── 2. Taxonomy: English-only classification works ───────────────────────
def t_taxonomy():
    from src.utils.constraint_taxonomy import (
        classify_query_constraint, CONSTRAINT_TYPES, EQUIPMENT_PROFILES,
        PHYSICAL_BOUNDS, LABEL_TO_TYPE)
    assert len(CONSTRAINT_TYPES) == 12
    hits = classify_query_constraint(
        "What is the minimum accessibility clearance and MTTR requirement?")
    labels = {ct.label for ct in hits}
    assert "Accessibility" in labels, labels
    assert "Time Parameters" in labels, labels
    # no Chinese-only fields remain
    ct = LABEL_TO_TYPE["Accessibility"]
    assert not hasattr(ct, "name_zh") and not hasattr(ct, "keywords_zh")
    assert EQUIPMENT_PROFILES["aircraft_engine"].name == "Aircraft Engine"
    assert all(b.parameter for b in PHYSICAL_BOUNDS)


# ── 3. StdChunker on a synthetic English standard document ───────────────
def t_chunker():
    from src.data.loader import Document
    from src.chunk.std_chunk import StdChunker
    doc = Document(
        doc_id="test_doc",
        title="Test Standard",
        content=(
            "5.1 Accessibility Design Requirements\n"
            "Maintenance passage width shall be no less than 500 mm. "
            "Hand-access clearance shall be >= 50 mm for finger operation.\n"
            "5.2 Diagnostics\n"
            "The equipment shall provide built-in test (BIT) capability with a "
            "fault detection rate >= 95 %. See Table 5-1 for test point layout.\n"
        ),
        source_path="synthetic",
        doc_type="txt",
    )
    chunker = StdChunker()
    chunks = chunker.chunk_documents([doc])
    assert len(chunks) > 0, "no chunks produced"
    t_chunker.chunks = chunks


# ── 4. BM25 retrieval over the chunks (dense model not required) ─────────
def t_bm25():
    from src.retriever.bm25_retriever import BM25Retriever
    r = BM25Retriever()
    r.index(t_chunker.chunks)
    res = r.retrieve("minimum maintenance passage width", top_k=3)
    assert len(res) > 0, "no retrieval results"
    top = res[0].chunk.content.lower()
    assert "passage" in top or "accessibility" in top, top[:120]


# ── 5. Metrics on English answers ────────────────────────────────────────
def t_metrics():
    from src.evaluation.metrics import (
        exact_match, partial_match, tuple_f1, extract_answer_tuples)
    gold = "Maintenance passage width shall be no less than 500 mm"
    pred = "The maintenance passage width must be at least 500 mm [Source 1]"
    assert partial_match(pred, gold) > 0
    tf = tuple_f1(
        extract_answer_tuples(pred, parameter="Accessibility"),
        extract_answer_tuples(gold, parameter="Accessibility"),
    )
    assert 0.0 <= tf <= 1.0
    assert exact_match(gold, gold) == 1


# ── 6. All translated prompts format without KeyError ────────────────────
def t_prompts():
    from src.retriever.hyde_retriever import HYDE_PROMPT_TEMPLATE
    from src.retriever.multi_query_retriever import MQ_PROMPT
    from src.chunk.hichunk_lite import PROMPT as HICHUNK_PROMPT
    from src.generator.rankrag_lite_gen import PROMPT as RANKRAG_PROMPT
    for tpl, kw in [
        (HYDE_PROMPT_TEMPLATE, dict(query="q")),
        (MQ_PROMPT, dict(query="q")),
    ]:
        out = tpl.format(**kw)
        assert "q" in out
    # prompts must be fully English now (no CJK outside regex modules)
    import re
    cjk = re.compile(r"[一-鿿]")
    for name, tpl in [("HYDE", HYDE_PROMPT_TEMPLATE), ("MQ", MQ_PROMPT),
                      ("HICHUNK", HICHUNK_PROMPT), ("RANKRAG", RANKRAG_PROMPT)]:
        assert not cjk.search(tpl), f"CJK left in {name} prompt"


# ── 7. Generators construct with mocked LLM and answer in English ────────
def t_generator():
    from src.generator.direct_gen import DirectGenerator
    from src.generator.bound_grounder import BoundGrounder

    class MockLLM:
        def generate(self, prompt):
            assert "Reference Materials" in prompt or "Question" in prompt
            return "Maintenance passage width shall be no less than 500 mm (source: chunk_0)"

    from src.retriever.base import RetrievalResult
    results = [RetrievalResult(chunk=c, score=1.0) for c in t_chunker.chunks[:3]]
    for gen_cls in (DirectGenerator, BoundGrounder):
        gen = gen_cls(llm_client=MockLLM())
        out = gen.generate("What is the minimum maintenance passage width?", results)
        assert out.answer, f"{gen_cls.__name__} empty answer"


# ── 8. Experiment scripts: importable & relative paths resolve ───────────
def t_experiment_paths():
    import re
    exp_dir = Path(__file__).parent / "experiments"
    pat = re.compile(r"Path\(\"(/root|[A-Za-z]:)")
    for py in exp_dir.glob("*.py"):
        text = py.read_text(encoding="utf-8")
        assert not pat.search(text), f"absolute path left in {py.name}"


def main():
    print("=== Translation smoke test ===")
    check("config relative paths", t_config)
    check("constraint taxonomy (EN)", t_taxonomy)
    check("StdChunker on EN doc", t_chunker)
    check("BM25 retrieval", t_bm25)
    check("metrics", t_metrics)
    check("prompt templates EN", t_prompts)
    check("generators w/ mock LLM", t_generator)
    check("no absolute paths in experiments", t_experiment_paths)
    print()
    if FAILED:
        print(f"FAILED: {len(FAILED)} → {FAILED}")
        sys.exit(1)
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
