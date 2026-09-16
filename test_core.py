"""Quick test: Verify StructStdRAG core modules

Test content:
1. Document loading
2. StdChunk chunking
3. CatRetriever retrieval
4. Basic query testing
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import RAW_DIR
from src.data.loader import load_directory
from src.chunk.std_chunk import StdChunker
from src.retriever.cat_retriever import CatRetriever
from src.utils.logger import get_logger

logger = get_logger("test_core")


def load_corpus():
    """Load the bundled MIL-HDBK-470A page corpus (mil-only release).

    Falls back to data/raw PDF/DOCX loading if users add their own documents.
    """
    import json
    from src.data.loader import Document
    pages = Path(__file__).parent / "data" / "mil470a" / "mil470a_pages_clean.jsonl"
    if pages.exists():
        docs = []
        with open(pages, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text", "")
                if len(text.strip()) > 100:
                    docs.append(Document(
                        doc_id=f"mil470a_page_{i + 1}",
                        title=f"MIL-HDBK-470A Page {i + 1}",
                        content=text,
                        source_path=str(pages),
                        doc_type="jsonl",
                    ))
        return docs
    return load_directory(RAW_DIR)




def main():
    logger.info("=== Testing StructStdRAG Core Modules ===\n")

    # 1. Load documents
    logger.info("Step 1: Loading documents...")
    documents = load_corpus()
    logger.info(f"✓ Loaded {len(documents)} documents\n")

    # 2. Chunking
    logger.info("Step 2: StdChunk chunking...")
    chunker = StdChunker()
    chunks = chunker.chunk_documents(documents)
    logger.info(f"✓ Generated {len(chunks)} chunks")

    # Statistics
    layer_dist = {}
    type_dist = {}
    constraint_dist = {}

    for c in chunks:
        layer_dist[c.layer] = layer_dist.get(c.layer, 0) + 1
        type_dist[c.semantic_type] = type_dist.get(c.semantic_type, 0) + 1
        for ct in c.constraint_types:
            constraint_dist[ct] = constraint_dist.get(ct, 0) + 1

    logger.info(f"  - Layer distribution: {layer_dist}")
    logger.info(f"  - Semantic types: {type_dist}")
    logger.info(f"  - Constraint types (top 5): {dict(list(sorted(constraint_dist.items(), key=lambda x: x[1], reverse=True))[:5])}\n")

    # 3. Build retrieval index
    logger.info("Step 3: CatRetriever building index...")
    retriever = CatRetriever(equipment_type="aircraft_engine")
    retriever.index(chunks)
    logger.info(f"✓ Index built successfully\n")

    # 4. Test queries
    test_queries = [
        "What is the minimum requirement for maintenance access width?",
        "What is the definition of MTTR?",
        "What are the requirements for accessibility design?",
    ]

    logger.info("Step 4: Testing queries...\n")
    for i, query in enumerate(test_queries, 1):
        logger.info(f"Query {i}: {query}")
        results = retriever.retrieve(query, top_k=3)

        logger.info(f"  Retrieved {len(results)} relevant chunks:")
        for j, r in enumerate(results, 1):
            logger.info(f"    [{j}] score={r.score:.3f}, layer={r.chunk.layer}, types={r.chunk.constraint_types}")
            logger.info(f"        {r.chunk.content[:80]}...")
        logger.info("")

    logger.info("=== Test completed ===")


if __name__ == "__main__":
    main()