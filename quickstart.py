"""Quick start script: Load documents → Chunk → Retrieve → Generate

Demonstrates the complete StructStdRAG pipeline
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import RAW_DIR
from src.data.loader import load_directory
from src.pipeline.structstd_rag import StructStdRAGPipeline
from src.utils.logger import get_logger

logger = get_logger("quickstart")


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
    logger.info("=== StructStdRAG Quick Start ===")

    # 1. Load documents
    logger.info(f"Loading bundled MIL-HDBK-470A corpus (data/mil470a)...")
    documents = load_corpus()
    logger.info(f"Loaded {len(documents)} documents")

    if not documents:
        logger.warning("No documents found. Please add PDF/DOCX files to data/raw/")
        return

    # 2. Initialize StructStdRAG pipeline
    logger.info("Initializing StructStdRAG pipeline...")
    pipeline = StructStdRAGPipeline(equipment_type="aircraft_engine")

    # 3. Ingest documents (chunking + indexing)
    logger.info("Ingesting documents (chunking + indexing)...")
    chunks = pipeline.ingest(documents)
    logger.info(f"Created {len(chunks)} chunks")

    # 4. Test queries
    test_queries = [
        "What is the minimum requirement for maintenance access width?",
        "What is the definition of MTTR?",
        "What are the requirements for accessibility design?",
    ]

    logger.info("\n=== Testing Queries ===")
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\nQuery {i}: {query}")
        result = pipeline.run(query, top_k=3)

        logger.info(f"Retrieved {len(result.retrieved)} chunks:")
        for j, r in enumerate(result.retrieved, 1):
            logger.info(f"  [{j}] (score={r.score:.3f}) {r.chunk.content[:100]}...")

        logger.info(f"Generated Answer:\n{result.generation.answer[:300]}...")
        logger.info(f"Confidence: {result.generation.confidence:.3f}")
        logger.info(f"Sources: {result.generation.sources}")

    logger.info("\n=== Quickstart Complete ===")


if __name__ == "__main__":
    main()