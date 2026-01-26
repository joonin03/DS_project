# 🏥 Medical Smart RAG: 당뇨병 임상진료지침 AI 닥터

**Interactive Medical RAG Web Application with Advanced Hybrid Search**

> **"Evidence-Based Medicine for Everyone"** > Google의 의료 특화 LLM인 **TxGemma-9b**를 기반으로 구축된 **대화형 의료 검색 증강 생성(RAG) 웹 애플리케이션**입니다.

기존의 단순 검색 엔진을 넘어, **Gradio 기반의 직관적인 인터페이스**를 제공하며, **Hybrid Search (Vector + BM25)**와 **Reranking**, 그리고 **CoT(Chain of Thought)** 기술을 결합하여 의학적 근거에 기반한 정밀하고 안전한 답변을 제공합니다.

---

## 🚀 Key Features

### 1. 🧠 Advanced Core Intelligence (Backend)
* **Hybrid Retrieval System:**
    * **Dense (Vector):** `ko-sroberta-multitask` 모델로 질문의 문맥적 의미 파악.
    * **Sparse (BM25):** `rank_bm25`를 활용하여 의학 전문 용어 및 키워드 매칭 정확도 보장.
    * **RRF (Reciprocal Rank Fusion):** 두 검색 결과를 최적의 비율로 결합하여 Recall 성능 극대화.
* **Precision Reranking:**
    * `BAAI/bge-reranker-v2-m3` (Cross-Encoder)를 사용하여 1차 검색된 문서들을 정밀 재검증.
    * 질문과의 연관성을 채점하여 환각(Hallucination)을 최소화하고 상위 3개의 "진짜 정답"만 LLM에 전달.

### 2. 💻 Interactive Web UI (Frontend)
* **Smart Chat Interface:** 의료 상담에 최적화된 Clean & Minimal 디자인(Gradio) 적용.
* **Real-time PDF Ingestion:** 사용자가 업로드한 최신 의학 지침(PDF)을 즉시 분석하여 Knowledge Base 구축.
* **Response Cleaning:** LLM의 내부 추론 과정(CoT)을 자동으로 필터링하여, 사용자에게는 결론만 깔끔하게 전달하는 후처리 로직 적용.

### 3. 🛡️ Robust Engineering & Safety
* **Prompt Engineering (CoT):** "생각 후 답변(Reasoning First)" 메커니즘을 적용하여 논리적 비약을 방지.
* **Safety Guardrails:** 의료 답변의 특수성을 고려하여 문서 내 **금기사항(Contraindications)** 및 **주의사항**을 명확히 구분하도록 프롬프트 제어.
* **OOP Architecture:** `Facade` 패턴을 적용하여 UI와 비즈니스 로직을 완벽히 분리(Decoupling), 유지보수성과 확장성 확보.

---

## 🛠️ Tech Stack

| Category | Technology | Description |
| :--- | :--- | :--- |
| **LLM** | [google/txgemma-9b-chat](https://huggingface.co/google/txgemma-9b-chat) | 답변 생성 (4-bit Quantization applied) |
| **Frontend** | **Gradio** | 대화형 웹 인터페이스 및 파일 처리 |
| **Embedding** | jhgan/ko-sroberta-multitask | 한국어 문장 임베딩 생성 |
| **Reranker** | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | 검색 결과 재순위화 (Cross-Encoder) |
| **Vector DB** | ChromaDB | 고성능 벡터 데이터 저장 및 검색 |
| **Search** | Rank-BM25 | 키워드 기반 역색인 검색 |
| **Parser** | pdfplumber | PDF 텍스트 및 표(Table) 마크다운 변환 |

---

## ⚙️ Installation & Usage

이 프로젝트는 **Google Colab (GPU: T4 or A100)** 환경에서 즉시 실행 가능하도록 최적화되어 있습니다.

### 1. Prerequisites
필요한 라이브러리를 설치합니다.
```bash
pip install -U transformers langchain langchain-community langchain-huggingface datasets
pip install sentence-transformers chromadb accelerate bitsandbytes pdfplumber rank_bm25 gradio
