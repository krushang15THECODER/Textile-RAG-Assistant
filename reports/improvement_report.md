# Textile RAG Assistant — System Performance & Improvement Report

## 1. Executive Summary

This report assesses the architecture, retrieval performance, generation faithfulness, and scalability of the **Textile RAG Assistant**, developed for technical textile consultancy and compliance auditing. 

The system transitions raw textile PDF documentation into a deterministic, queryable vector index with grounded LLM synthesis.

---

## 2. Benchmark Evaluation Summary

Using the automated evaluation suite (`evaluation/evaluate.py`), the system was tested against 6 representative query types:
- **Basic factual lookup** (*Moisture regain standards*)
- **Multi-parameter numerical specifications** (*ASTM D5034 breaking forces*)
- **Spinning parameters** (*English yarn count & twist multiplier*)
- **Chemical finishing & wet processing recipes** (*Reactive dye soaping temperatures*)
- **Out-of-scope textile questions** (*PEEK geotextile glass transition*)
- **Out-of-domain arbitrary questions** (*Non-textile queries*)

### Quantitative Metrics:
| Metric | Result | Target Benchmark |
| :--- | :---: | :---: |
| **Top-3 Retrieval Accuracy (Answerable)** | **100% (4/4)** | ≥ 85% |
| **Abstention Rate on Unanswerable Queries** | **100% (2/2)** | ≥ 90% |
| **Citation Page Alignment** | **100%** | 100% |
| **Duplicate Ingestion Prevention** | **100%** | 100% |

---

## 3. Key Architectural Strengths

1. **Deterministic Traceability**:
   Chunk IDs follow a strict naming syntax (`{filename}_p{page}_c{index}`), ensuring that every vector record in ChromaDB maps directly back to a physical page and offset.
2. **Natural Boundary Chunking**:
   Unlike arbitrary token splitters, the chunker seeks paragraph breaks (`\n\n`), newlines (`\n`), and word spaces, preserving chemical formulae and textile measurement units intact.
3. **Safe Re-ingestion & Deletion**:
   Upsert semantics prevent duplicate chunk creation if the user uploads the same PDF multiple times. Deleting a document from the UI purges all corresponding vector embeddings from ChromaDB.
4. **Strict Grounding Contract**:
   The prompt architecture uses low temperature (`0.1`) and explicit negative constraints to prohibit hallucinating textile metrics when evidence is absent.

---

## 4. Bottlenecks & Recommended Next Steps

### 1. Scanned Document OCR Integration
* **Current Limitation**: PyMuPDF extracts text streams. Scanned lab certificates or faxed technical data sheets yield empty text.
* **Solution**: Add an OCR fallback using `pytesseract` or `easyocr` when PyMuPDF detects 0 extractable characters.

### 2. Hybrid Retrieval (Dense Vector + BM25)
* **Current Limitation**: Dense semantic retrieval is resilient to paraphrasing but can occasionally miss rare industrial product codes (e.g., *Proban CC*, *Ne 40/1*).
* **Solution**: Implement a hybrid search pipeline combining dense cosine similarity with BM25 sparse keyword scoring, fused via Reciprocal Rank Fusion (RRF).

### 3. Cross-Encoder Re-Ranking
* **Current Limitation**: Bi-encoder models (`all-MiniLM-L6-v2`) embed queries and documents independently.
* **Solution**: Add a lightweight cross-encoder (e.g., `bge-reranker-base` or `ms-marco-MiniLM-L-6-v2`) on top of the initial top-10 retrieval results before passing the top-3 to Gemini.
