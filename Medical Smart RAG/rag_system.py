from huggingface_hub import login
login()

import os
import logging
import torch
import numpy as np
import pdfplumber
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

# 라이브러리 임포트
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings

# ==========================================
# 1. 설정 및 로깅 (Configuration & Logging)
# ==========================================

# 로깅 설정
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("MedicalRAG")

@dataclass
class AppConfig:
    """애플리케이션 전체 설정 관리"""
    # 모델 경로
    llm_model_id: str = "google/txgemma-9b-chat"
    embedding_model_id: str = "jhgan/ko-sroberta-multitask"
    reranker_model_id: str = "BAAI/bge-reranker-v2-m3"

    # 벡터 DB 설정
    collection_name: str = "medical_kb"
    chroma_path: str = "./chroma_db"  # 영구 저장 경로 (선택 사항)

    # 검색 파라미터
    initial_retrieval_k: int = 10  # 1차 검색 개수
    final_top_k: int = 3           # 최종 Reranking 후 개수
    rrf_k_constant: int = 60       # RRF 상수

    # 문서 처리 설정
    chunk_size: int = 800
    chunk_overlap: int = 350

    # 생성 설정
    max_new_tokens: int = 768
    temperature: float = 0.1
    top_p: float = 0.9

config = AppConfig()

# ==========================================
# 2. 데이터 구조 (Data Structures)
# ==========================================

@dataclass
class SearchResult:
    """검색된 문서 정보 구조체"""
    text: str
    source: str
    score: float = 0.0

@dataclass
class RAGResponse:
    """최종 답변 구조체"""
    query: str
    answer: str
    sources: List[str]
    processing_time: float = 0.0

# ==========================================
# 3. 문서 처리 모듈 (Document Processor)
# ==========================================

class DocumentProcessor:
    """PDF 로딩 및 청킹 담당"""

    @staticmethod
    def _table_to_markdown(table: List[List[str]]) -> str:
        """PDF 표를 마크다운 형식으로 변환"""
        if not table or len(table) < 2:
            return ""
        try:
            # None 값을 빈 문자열로 처리하고 줄바꿈 문자 제거
            table = [['' if cell is None else str(cell).replace('\n', ' ') for cell in row] for row in table]

            markdown = "| " + " | ".join(table[0]) + " |\n"
            markdown += "| " + " | ".join(["---"] * len(table[0])) + " |\n"
            for row in table[1:]:
                markdown += "| " + " | ".join(row) + " |\n"
            return markdown + "\n"
        except Exception as e:
            logger.warning(f"표 변환 중 오류 발생: {e}")
            return ""

    @classmethod
    def process_pdf(cls, file_path: str) -> List[Dict[str, str]]:
        """PDF 파일을 읽어 처리된 청크 리스트 반환"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        logger.info(f"📄 PDF 처리 시작: {file_path}")
        docs = []

        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    tables = page.extract_tables()
                    table_text = "".join([cls._table_to_markdown(t) for t in tables if t])

                    # 표 내용과 텍스트 결합
                    combined_content = f"{text}\n\n[표 데이터]\n{table_text}"
                    docs.append(Document(
                        page_content=combined_content,
                        metadata={"source": f"{os.path.basename(file_path)} (p.{i + 1})"}
                    ))
        except Exception as e:
            logger.error(f"PDF 읽기 실패: {e}")
            raise e

        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", "|", ".", " "]
        )
        splits = text_splitter.split_documents(docs)

        logger.info(f"✅ 문서 분할 완료: {len(splits)} 청크")
        return [{"text": doc.page_content, "source": doc.metadata['source']} for doc in splits]

# ==========================================
# 4. 검색 엔진 모듈 (Retrieval Engine)
# ==========================================

class HybridRetriever:
    """Vector DB + BM25 + Reranker 검색 엔진"""

    def __init__(self, cfg: AppConfig, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer # BM25 토큰화용

        # 모델 로드
        logger.info("SEARCH: 임베딩 및 리랭커 모델 로딩 중...")
        self.embed_model = SentenceTransformer(cfg.embedding_model_id)
        self.reranker = CrossEncoder(
            cfg.reranker_model_id,
            automodel_args={"torch_dtype": torch.float16},
            trust_remote_code=True
        )

        # 벡터 DB 초기화 (메모리 모드)
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.chroma_client.reset() # 초기화
        self.collection = self.chroma_client.create_collection(cfg.collection_name)

        # Sparse 검색(BM25)용 상태
        self.bm25 = None
        self.documents_map = {} # ID -> Document 매핑

    def index_documents(self, documents: List[Dict[str, str]]):
        """문서 인덱싱 (Vector DB + BM25)"""
        if not documents:
            logger.warning("인덱싱할 문서가 없습니다.")
            return

        logger.info(f"SEARCH: {len(documents)}개 문서 인덱싱 시작...")
        texts = [doc['text'] for doc in documents]
        ids = [f"doc_{i}" for i in range(len(texts))]

        # 1. BM25 인덱싱
        tokenized_corpus = [self.tokenizer.tokenize(doc) for doc in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 2. 문서 맵 저장
        for i, doc_id in enumerate(ids):
            self.documents_map[doc_id] = {
                "text": texts[i],
                "source": documents[i]['source']
            }

        # 3. 벡터 DB 저장
        embeddings = self.embed_model.encode(texts, convert_to_numpy=True)
        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            ids=ids
        )
        logger.info("✅ 인덱싱 완료.")

    def search(self, query: str) -> List[SearchResult]:
        """하이브리드 검색 + 리랭킹 실행"""
        if not self.bm25 or not self.documents_map:
            logger.warning("인덱싱된 문서가 없습니다.")
            return []

        k = self.cfg.initial_retrieval_k

        # 1. Dense Retrieval (벡터)
        query_vec = self.embed_model.encode(query).tolist()
        vec_res = self.collection.query(query_embeddings=[query_vec], n_results=k)
        vec_ids = vec_res['ids'][0] if vec_res['ids'] else []

        # 2. Sparse Retrieval (BM25)
        tokenized_query = self.tokenizer.tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_n_indices = np.argsort(bm25_scores)[::-1][:k]
        bm25_ids = [f"doc_{i}" for i in top_n_indices]

        # 3. RRF Fusion (순위 재조정)
        rrf_score = {}
        for rank, doc_id in enumerate(vec_ids):
            rrf_score[doc_id] = rrf_score.get(doc_id, 0) + 1 / (rank + self.cfg.rrf_k_constant)
        for rank, doc_id in enumerate(bm25_ids):
            rrf_score[doc_id] = rrf_score.get(doc_id, 0) + 1 / (rank + self.cfg.rrf_k_constant)

        # 상위 후보 추출
        sorted_candidates = sorted(rrf_score.items(), key=lambda item: item[1], reverse=True)
        candidate_ids = [doc_id for doc_id, _ in sorted_candidates[:k]]

        # 4. Reranking (정밀 검증)
        candidate_texts = [self.documents_map[doc_id]['text'] for doc_id in candidate_ids]
        pairs = [[query, text] for text in candidate_texts]
        rerank_scores = self.reranker.predict(pairs)

        # 최종 정렬 및 결과 반환
        final_results = []
        for doc_id, score in sorted(zip(candidate_ids, rerank_scores), key=lambda x: x[1], reverse=True)[:self.cfg.final_top_k]:
            doc_info = self.documents_map[doc_id]
            final_results.append(SearchResult(
                text=doc_info['text'],
                source=doc_info['source'],
                score=float(score)
            ))

        return final_results

# ==========================================
# 5. 생성 엔진 모듈 (Generation Engine)
# ==========================================

class LLMGenerator:
    """LLM 로드 및 답변 생성 담당"""

    PROMPT_TEMPLATE = """<start_of_turn>user
당신은 '근거 중심 의학(Evidence-Based Medicine)'을 준수하는 의료 AI 전문의입니다.
아래 [검색된 문서]를 정밀하게 분석하여 질문에 답변하세요.

[검색된 문서]
{context}

[답변 작성 원칙]
1. **단계별 추론**: 질문의 의도를 파악하고, 문서에서 관련된 팩트를 찾은 뒤 답변을 구성하세요.
2. **엄격한 구분**: 문서 내의 '금기사항(Contraindications)', '주의사항(Warnings)', '권고(Indications/Target)'를 명확히 구분하세요.
3. **오해 금지**: 표 작성 시 '위험(Risk)'이라는 단어가 오해를 사지 않도록 '임상적 영향' 등으로 명확히 하세요.
4. **사실 기반**: 문서에 없는 내용은 "제공된 문서에 정보가 없습니다"라고 답하세요.
5. **출처 인용**: 답변의 근거가 되는 내용이 문서 어디에 있는지 참고하세요.

질문: {query}<end_of_turn>
<start_of_turn>model
[분석 단계]
1. 질문의 핵심 키워드와 의도 파악: """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        logger.info(f"GEN: LLM 모델 로딩 중 ({cfg.llm_model_id})...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.llm_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("✅ LLM 로딩 완료.")

    def generate(self, query: str, context_docs: List[SearchResult]) -> str:
        """프롬프트 구성 및 답변 생성"""
        # 문맥 조합
        context_str = "\n\n".join([f"문서[{i+1}]: {doc.text}" for i, doc in enumerate(context_docs)])

        prompt = self.PROMPT_TEMPLATE.format(context=context_str, query=query)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.cfg.max_new_tokens,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    repetition_penalty=1.1,
                    do_sample=True
                )

            full_response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return full_response.strip()

        except Exception as e:
            logger.error(f"답변 생성 중 오류: {e}")
            return "죄송합니다. 답변을 생성하는 도중 오류가 발생했습니다."

# ==========================================
# 6. 메인 시스템 (Orchestrator)
# ==========================================

class MedicalRAGSystem:
    """전체 RAG 시스템을 관장하는 Facade 클래스"""

    def __init__(self, config: AppConfig = AppConfig()):
        self.cfg = config

        # 컴포넌트 초기화
        self.generator = LLMGenerator(config)
        self.retriever = HybridRetriever(config, self.generator.tokenizer)

    def ingest_file(self, file_path: str):
        """파일 업로드 및 처리"""
        try:
            processed_docs = DocumentProcessor.process_pdf(file_path)
            self.retriever.index_documents(processed_docs)
            return True
        except Exception as e:
            logger.error(f"파일 처리 실패: {e}")
            return False

    def ask(self, query: str) -> RAGResponse:
        """질문 처리 파이프라인"""
        logger.info(f"❓ 질문 수신: {query}")

        # 1. 검색
        search_results = self.retriever.search(query)

        if not search_results:
            return RAGResponse(query=query, answer="관련된 문서를 찾을 수 없습니다.", sources=[])

        # 2. 생성
        answer = self.generator.generate(query, search_results)

        # 3. 출처 정리
        sources = list(set([doc.source for doc in search_results]))

        return RAGResponse(query=query, answer=answer, sources=sources)

# ==========================================
# 7. 실행 예제 (Usage)
# ==========================================

if __name__ == "__main__":
    import time

    # 1. 시스템 초기화
    rag_system = MedicalRAGSystem()

    # 2. 데이터 주입 (경로를 실제 파일 경로로 수정하세요)
    # 업로드하신 파일명을 기반으로 예시 경로 설정
    pdf_path = "/content/drive/MyDrive/[대한의학회] 당뇨병 임상진료지침.pdf"

    if os.path.exists(pdf_path):
        rag_system.ingest_file(pdf_path)

        # 3. 질문 테스트
        questions = [
            "당뇨병 환자의 운동 부하 검사 금기 사항은 뭐야?",
            "SGLT2 억제제와 DPP-4 억제제의 체중 영향 차이를 표로 비교해줘."
        ]

        for q in questions:
            print(f"\n[질문] {q}")
            start_t = time.time()
            response = rag_system.ask(q)
            end_t = time.time()

            print(f"[답변]\n{response.answer}")
            print(f"[출처] {response.sources}")
            print(f"[소요시간] {end_t - start_t:.2f}초")
            print("-" * 50)
    else:
        logger.warning(f"PDF 파일을 찾을 수 없습니다. 경로를 확인해주세요: {pdf_path}")