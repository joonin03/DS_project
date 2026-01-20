# 🏥 Medical Smart RAG: 당뇨병 임상진료지침 Q&A

Google의 의료 특화 LLM인 **TxGemma-9b**와 **RAG(Retrieval-Augmented Generation)** 기술을 활용하여 구축한 의료 질의응답 시스템입니다.

대한의학회의 [당뇨병 임상진료지침] PDF 문서를 기반으로 답변을 생성하며, 특히 **PDF 내의 표(Table) 데이터를 Markdown으로 변환**하여 RAG의 정확도를 높인 것이 특징입니다.

## 🚀 Key Features

* **Medical LLM 활용:** Google의 `TxGemma-9b-chat` 모델을 4-bit 양자화하여 사용하여 의료 도메인에 특화된 답변 생성.
* **Advanced PDF Processing:** `pdfplumber`를 사용하여 PDF 내의 텍스트뿐만 아니라 **표(Table) 구조를 인식하고 Markdown 형식으로 변환**하여 학습.
* **Korean Embedding:** 한국어 문장 의미 파악에 탁월한 `jhgan/ko-sroberta-multitask` 임베딩 모델 사용.
* **Efficient Retrieval:** `ChromaDB`와 `LangChain`을 활용한 벡터 검색 및 문맥(Context) 추출.

## 🛠️ Tech Stack

* **Model:** [google/txgemma-9b-chat](https://huggingface.co/google/txgemma-9b-chat)
* **Embedding:** jhgan/ko-sroberta-multitask
* **Framework:** LangChain, PyTorch
* **Vector DB:** ChromaDB
* **Tools:** pdfplumber (PDF & Table Parsing), BitsAndBytes (Quantization)

## ⚙️ Installation & Usage

이 프로젝트는 Google Colab(GPU 환경)에서 실행되도록 설계되었습니다.

### 1. Prerequisites
필요한 라이브러리를 설치합니다.
```bash
pip install -U transformers langchain langchain-community langchain-huggingface
pip install sentence-transformers chromadb accelerate bitsandbytes pdfplumber rank_bm25
