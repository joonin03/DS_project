# 🏥 Medical Smart RAG: 당뇨병 임상진료지침 AI 닥터 (Core Engine)

**Advanced Hybrid RAG & Reranking System Backend**

Google의 의료 특화 LLM인 **TxGemma-9b**와 최신 **RAG(Retrieval-Augmented Generation)** 기술을 결합한 의료 질의응답 시스템의 **핵심 백엔드 엔진**입니다.

단독 실행 가능한 Python 모듈로 설계되었으며, **Hybrid Search (Vector + BM25)**와 **Reranking (Cross-Encoder)** 기술을 탑재하여 의료 문서 기반의 정밀한 답변을 생성합니다.

## 🚀 Key Features

* **🧠 Advanced Hybrid Retrieval:**
    * **Dense (Vector):** 문맥적 의미 파악 (`ko-sroberta-multitask`)
    * **Sparse (BM25):** 정확한 의학 키워드 매칭 (`rank_bm25`)
    * **RRF (Reciprocal Rank Fusion):** 두 검색 결과를 최적의 비율로 결합
* **⚖️ Precision Reranking:**
    * `BAAI/bge-reranker-v2-m3` 모델을 사용하여 검색된 후보 문서들을 정밀 검증.
    * 질문과 문서 간의 연관성을 채점하여 상위 3개의 **"진짜 정답"**만 LLM에게 전달.
* **🛡️ Robust PDF Processing:**
    * `pdfplumber`를 활용하여 텍스트뿐만 아니라 **표(Table)** 데이터를 Markdown으로 변환하여 학습.
    * LLM이 표 안의 수치(eGFR, 혈당 기준 등)를 정확히 비교 분석 가능.
* **🔒 Secure & Modular:**
    * 객체 지향(OOP) 설계로 확장이 용이하며, Hugging Face 토큰을 안전하게 관리.

## 🛠️ Tech Stack

| Category | Technology | Usage |
| :--- | :--- | :--- |
| **LLM** | [google/txgemma-9b-chat](https://huggingface.co/google/txgemma-9b-chat) | 답변 생성 (4-bit Quantization) |
| **Embedding** | jhgan/ko-sroberta-multitask | 한국어 문장 임베딩 |
| **Reranker** | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | 검색 결과 재순위화 (Cross-Encoder) |
| **Vector DB** | ChromaDB | 벡터 데이터 저장 및 검색 |
| **Search** | Rank-BM25 | 키워드 기반 검색 |
| **Tools** | pdfplumber, LangChain | 문서 전처리 및 파이프라인 관리 |

## ⚙️ Installation

이 프로젝트는 **Google Colab (GPU: T4 or A100)** 환경에 최적화되어 있습니다.

**1. 필수 라이브러리 설치**
```bash
pip install -U transformers langchain langchain-community langchain-huggingface datasets
pip install sentence-transformers chromadb accelerate bitsandbytes pdfplumber rank_bm25 python-dotenv
