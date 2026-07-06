"""POC 런타임 설정.

모든 값은 환경변수(.env)로 덮어쓸 수 있으며, 기본값은 사용자가 제공한
하네스-1 기반 RAG 구동 환경 정보를 그대로 반영한다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

# .env 로드 (python-dotenv가 있으면 사용, 없으면 무시)
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:  # pragma: no cover - dotenv는 선택적 의존성
    pass


def _get(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # --- 검색 모델 (harness-1) : OpenAI 호환 ---
    harness1_base_url: str = _get("HARNESS1_BASE_URL", "http://192.168.1.108:8000/v1")
    harness1_model: str = _get("HARNESS1_MODEL", "harness-1")
    harness1_api_key: str = _get("HARNESS1_API_KEY", "EMPTY")
    harness1_temperature: float = _get_float("HARNESS1_TEMPERATURE", 0.7)
    harness1_max_tokens: int = _get_int("HARNESS1_MAX_TOKENS", 4096)
    harness1_max_turns: int = _get_int("HARNESS1_MAX_TURNS", 8)

    # --- 임베딩 모델 ---
    # backend: "local_bge_m3"(로컬 BAAI/bge-m3) | "openai"(원격 OpenAI 호환 엔드포인트)
    embed_backend: str = _get("EMBED_BACKEND", "local_bge_m3")
    embed_dim: int = _get_int("EMBED_DIM", 1024)
    # 로컬 백엔드 설정
    embed_local_model: str = _get("EMBED_LOCAL_MODEL", "BAAI/bge-m3")
    embed_device: str = _get("EMBED_DEVICE", "cpu")  # "cpu" | "cuda"
    embed_normalize: bool = _get("EMBED_NORMALIZE", "1") == "1"
    # 원격(OpenAI 호환) 백엔드 설정
    embed_base_url: str = _get("EMBED_BASE_URL", "http://192.168.1.108:8081/v1")
    embed_model: str = _get("EMBED_MODEL", "qwen3-embedding")
    embed_api_key: str = _get("EMBED_API_KEY", "EMPTY")

    # --- 리랭커 모델 (Qwen3-Reranker-8B) : vLLM /score ---
    rerank_url: str = _get("RERANK_URL", "http://192.168.1.108:8003/score")
    rerank_model: str = _get("RERANK_MODEL", "Qwen/Qwen3-Reranker-8B")
    rerank_enabled: bool = _get("RERANK_ENABLED", "1") == "1"

    # --- 오케스트레이터 (exaone3.5:32b) : Ollama OpenAI 호환 ---
    orch_base_url: str = _get("ORCH_BASE_URL", "http://192.168.1.108:11434/v1")
    orch_model: str = _get("ORCH_MODEL", "exaone3.5:32b")
    orch_api_key: str = _get("ORCH_API_KEY", "ollama")
    orch_temperature: float = _get_float("ORCH_TEMPERATURE", 0.3)
    # 검색 전 질의를 영어로 번역할지 여부 (off면 원문 그대로 검색)
    translate_query_enabled: bool = _get("TRANSLATE_QUERY_TO_ENGLISH", "0") == "1"

    # --- 웹검색 : 혼합식 도구 (provider: serpapi | serper) ---
    web_search_enabled: bool = _get("WEB_SEARCH_ENABLED", "1") == "1"
    web_search_provider: str = _get("WEB_SEARCH_PROVIDER", "serpapi")
    # SerpApi (serpapi.com) — GET, api_key 파라미터, 응답 organic_results
    serpapi_api_key: str = _get("SERPAPI_API_KEY", "")
    serpapi_url: str = _get("SERPAPI_URL", "https://serpapi.com/search")
    # Serper (serper.dev) — POST, X-API-KEY 헤더, 응답 organic
    serper_api_key: str = _get("SERPER_API_KEY", "")
    serper_search_url: str = _get("SERPER_SEARCH_URL", "https://google.serper.dev/search")
    serper_scrape_url: str = _get("SERPER_SCRAPE_URL", "https://scrape.serper.dev")
    # 페이지 본문 읽기 (Jina Reader)
    jina_api_key: str = _get("JINA_API_KEY", "")
    web_search_limit: int = _get_int("WEB_SEARCH_LIMIT", 10)
    web_read_max_chars: int = _get_int("WEB_READ_MAX_CHARS", 6000)

    @property
    def web_api_key(self) -> str:
        """현재 provider에 맞는 API 키."""
        if self.web_search_provider == "serpapi":
            return self.serpapi_api_key
        return self.serper_api_key

    # --- 벡터스토어 (Qdrant) ---
    qdrant_url: str = _get("QDRANT_URL", "http://192.168.1.10:6333")
    qdrant_api_key: str = _get("QDRANT_API_KEY", "")
    qdrant_collection: str = _get("QDRANT_COLLECTION", "browsecompplus_qwen3_1024")
    # Qdrant payload 필드명 (실제 컬렉션 구조에서 확인된 값)
    #   document=본문, source=문서ID, chunk_id=청크ID
    qdrant_text_field: str = _get("QDRANT_TEXT_FIELD", "document")
    qdrant_docid_field: str = _get("QDRANT_DOCID_FIELD", "source")
    qdrant_chunkid_field: str = _get("QDRANT_CHUNKID_FIELD", "chunk_id")
    # named vector 이름 (이 컬렉션은 'dense' 사용. 빈 문자열이면 unnamed)
    qdrant_vector_name: str = _get("QDRANT_VECTOR_NAME", "dense")

    # --- 검색 파이프라인 파라미터 ---
    search_limit: int = _get_int("SEARCH_LIMIT", 50)      # Qdrant 1차 검색 후보 수
    rerank_top_k: int = _get_int("RERANK_TOP_K", 10)      # 리랭크 후 노출 수
    synth_top_k: int = _get_int("SYNTH_TOP_K", 6)         # 최종 합성에 넣을 문서 수
    request_timeout: int = _get_int("REQUEST_TIMEOUT", 120)

    # --- 하이브리드 검색 (dense + sparse, RRF 융합) ---
    # sparse_method: "bm25"(fastembed Qdrant/bm25) | "bge_m3"(FlagEmbedding bge-m3 lexical weights)
    # sparse_vector_name: Qdrant 컬렉션의 sparse 벡터 이름 (v1='bm25', v2='sparse')
    sparse_enabled: bool = _get("SPARSE_ENABLED", "0") == "1"
    sparse_vector_name: str = _get("SPARSE_VECTOR_NAME", "")
    sparse_method: str = _get("SPARSE_METHOD", "")
    sparse_model: str = _get("SPARSE_MODEL", "")
    hybrid_prefetch_limit: int = _get_int("HYBRID_PREFETCH_LIMIT", 50)  # 융합 전 각 분기 후보 수

    @property
    def qdrant_using_named_vector(self) -> bool:
        return bool(self.qdrant_vector_name)


settings = Settings()


# ---------------------------------------------------------------------------
# 버전 프로파일
# 공유 구성(harness-1·리랭커·오케스트레이터·웹검색·파이프라인 파라미터)은 settings(.env) 를 그대로
# 상속하고, 버전마다 다른 "임베딩 + 벡터스토어" 부분만 명시적으로 고정한다.
# → 전역 .env 의 QDRANT_*/EMBED_* 값에 휘둘리지 않고 v1/v2 가 각자 다른 백엔드를 가리킨다.
#   (각 값은 V1_* / V2_* 환경변수로 다시 덮어쓸 수 있다.)
#
# v1 프로파일
#   - 임베딩 모델: qwen3-embedding (원격, 1024차원)
#   - 벡터스토어 : 192.168.1.10:6333 / 컬렉션 browsecompplus_qwen3_1024 (dense)
# ---------------------------------------------------------------------------
def _build_settings_v1() -> Settings:
    return replace(
        settings,
        # 임베딩: 원격 qwen3-embedding (browsecompplus 컬렉션이 qwen3 임베딩이라 동일 모델 사용)
        embed_backend=_get("V1_EMBED_BACKEND", "openai"),
        embed_base_url=_get("V1_EMBED_BASE_URL", "http://192.168.1.108:8081/v1"),
        embed_model=_get("V1_EMBED_MODEL", "qwen3-embedding"),
        embed_api_key=_get("V1_EMBED_API_KEY", "EMPTY"),
        embed_dim=_get_int("V1_EMBED_DIM", 1024),
        # 벡터스토어: browsecompplus_qwen3_1024 (dense)
        qdrant_url=_get("V1_QDRANT_URL", "http://192.168.1.10:6333"),
        qdrant_api_key=_get("V1_QDRANT_API_KEY", ""),
        qdrant_collection=_get("V1_QDRANT_COLLECTION", "browsecompplus_qwen3_1024"),
        qdrant_vector_name=_get("V1_QDRANT_VECTOR_NAME", "dense"),
        qdrant_text_field=_get("V1_QDRANT_TEXT_FIELD", "document"),
        qdrant_docid_field=_get("V1_QDRANT_DOCID_FIELD", "source"),
        qdrant_chunkid_field=_get("V1_QDRANT_CHUNKID_FIELD", "chunk_id"),
        # 하이브리드: dense(qwen3) + sparse(bm25, modifier=IDF) — 쿼리는 fastembed Qdrant/bm25
        sparse_enabled=_get("V1_SPARSE_ENABLED", "1") == "1",
        sparse_vector_name=_get("V1_SPARSE_VECTOR_NAME", "bm25"),
        sparse_method=_get("V1_SPARSE_METHOD", "bm25"),
        sparse_model=_get("V1_SPARSE_MODEL", "Qdrant/bm25"),
    )


settings_v1 = _build_settings_v1()


# ---------------------------------------------------------------------------
# v2 프로파일
#   - 임베딩 모델: BAAI/bge-m3 (로컬, 1024차원)
#   - 벡터스토어 : 121.78.130.83:6333 / 컬렉션 kb_bsss (dense, 1024차원)
# 각 값은 V2_* 환경변수로 다시 덮어쓸 수 있다.
# ---------------------------------------------------------------------------
def _build_settings_v2() -> Settings:
    return replace(
        settings,
        # 임베딩: 로컬 BAAI/bge-m3 (1024차원)
        embed_backend=_get("V2_EMBED_BACKEND", "local_bge_m3"),
        embed_local_model=_get("V2_EMBED_LOCAL_MODEL", "BAAI/bge-m3"),
        embed_device=_get("V2_EMBED_DEVICE", settings.embed_device),
        embed_dim=_get_int("V2_EMBED_DIM", 1024),
        # (openai 백엔드로 전환할 경우에 대비한 원격 임베딩 설정)
        embed_base_url=_get("V2_EMBED_BASE_URL", settings.embed_base_url),
        embed_model=_get("V2_EMBED_MODEL", "bge-m3"),
        embed_api_key=_get("V2_EMBED_API_KEY", settings.embed_api_key),
        # 벡터스토어: kb_bsss (dense)
        qdrant_url=_get("V2_QDRANT_URL", "http://121.78.130.83:6333"),
        qdrant_api_key=_get("V2_QDRANT_API_KEY", ""),
        qdrant_collection=_get("V2_QDRANT_COLLECTION", "kb_bsss"),
        qdrant_vector_name=_get("V2_QDRANT_VECTOR_NAME", "dense"),
        qdrant_text_field=_get("V2_QDRANT_TEXT_FIELD", "content"),
        qdrant_docid_field=_get("V2_QDRANT_DOCID_FIELD", "doc_id"),
        qdrant_chunkid_field=_get("V2_QDRANT_CHUNKID_FIELD", "chunk_id"),
        # 하이브리드: dense(bge-m3) + sparse(bge-m3 lexical weights) — FlagEmbedding 한 번에 생성
        sparse_enabled=_get("V2_SPARSE_ENABLED", "1") == "1",
        sparse_vector_name=_get("V2_SPARSE_VECTOR_NAME", "sparse"),
        sparse_method=_get("V2_SPARSE_METHOD", "bge_m3"),
        sparse_model=_get("V2_SPARSE_MODEL", "BAAI/bge-m3"),
    )


settings_v2 = _build_settings_v2()
