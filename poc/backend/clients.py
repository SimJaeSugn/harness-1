"""외부 서비스 클라이언트 모음.

- 임베딩(qwen3-embedding) / harness-1 / 오케스트레이터(exaone3.5): OpenAI 호환 SDK
- 리랭커(Qwen3-Reranker-8B): vLLM /score REST
- 벡터스토어: Qdrant REST (qdrant-client)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests
from openai import OpenAI
from qdrant_client import QdrantClient

from .config import Settings, settings

logger = logging.getLogger("poc.clients")


# ---------------------------------------------------------------------------
# OpenAI 호환 클라이언트 (임베딩 / harness-1 / 오케스트레이터)
# ---------------------------------------------------------------------------

def make_embedding_client() -> OpenAI:
    return OpenAI(base_url=settings.embed_base_url, api_key=settings.embed_api_key)


def make_harness1_client() -> OpenAI:
    return OpenAI(base_url=settings.harness1_base_url, api_key=settings.harness1_api_key)


def make_orchestrator_client() -> OpenAI:
    return OpenAI(base_url=settings.orch_base_url, api_key=settings.orch_api_key)


# ---------------------------------------------------------------------------
# 임베딩
# ---------------------------------------------------------------------------

# 로컬 임베딩 모델은 무겁고 1회 로드로 충분하므로 프로세스 전역에 캐시한다.
# 버전별로 다른 모델(예: v2=BAAI/bge-m3)을 쓸 수 있으므로 (모델명, 디바이스)별로 캐시한다.
_local_embed_models: dict = {}


def _get_local_embed_model(model_name: str, device: str):
    key = (model_name, device)
    model = _local_embed_models.get(key)
    if model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("loading local embedding model: %s (%s)", model_name, device)
        model = SentenceTransformer(model_name, device=device)
        _local_embed_models[key] = model
    return model


class EmbeddingClient:
    """질의 임베딩. backend='local_bge_m3'면 로컬 BAAI/bge-m3, 'openai'면 원격 엔드포인트.

    cfg 로 버전별 설정(임베딩 모델/엔드포인트)을 주입할 수 있다(기본: 전역 settings).
    """

    def __init__(self, client: Optional[OpenAI] = None, cfg: Settings = settings) -> None:
        self._cfg = cfg
        self._backend = cfg.embed_backend
        self._client = client if self._backend == "openai" else None

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self._backend == "local_bge_m3":
            model = _get_local_embed_model(self._cfg.embed_local_model, self._cfg.embed_device)
            vecs = model.encode(
                texts,
                normalize_embeddings=self._cfg.embed_normalize,
                convert_to_numpy=True,
            )
            return [v.tolist() for v in vecs]
        client = self._client or OpenAI(
            base_url=self._cfg.embed_base_url, api_key=self._cfg.embed_api_key
        )
        resp = client.embeddings.create(
            model=self._cfg.embed_model,
            input=texts,
            encoding_format="float",
        )
        return [d.embedding for d in resp.data]

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


# ---------------------------------------------------------------------------
# 희소(sparse) 인코더 + 하이브리드 질의 인코더
#   - bm25   : fastembed Qdrant/bm25 (컬렉션 modifier=IDF 와 짝) — v1
#   - bge_m3 : FlagEmbedding bge-m3 lexical weights (dense 도 같은 모델로 한 번에) — v2
# 모델은 무거우므로 (메서드/모델명/디바이스)별로 프로세스 전역에 캐시한다.
# ---------------------------------------------------------------------------
_bm25_encoders: dict = {}
_bgem3_models: dict = {}

# 인코더가 반환하는 희소 벡터: (indices, values) 또는 없음(None)
Sparse = Optional[tuple]


def _get_bm25_encoder(model_name: str):
    enc = _bm25_encoders.get(model_name)
    if enc is None:
        from fastembed import SparseTextEmbedding

        logger.info("loading sparse(bm25) encoder: %s", model_name)
        enc = SparseTextEmbedding(model_name=model_name)
        _bm25_encoders[model_name] = enc
    return enc


def _get_bgem3_model(model_name: str, device: str):
    key = (model_name, device)
    model = _bgem3_models.get(key)
    if model is None:
        from FlagEmbedding import BGEM3FlagModel

        logger.info("loading bge-m3 (FlagEmbedding) for dense+sparse: %s (%s)", model_name, device)
        model = BGEM3FlagModel(model_name, use_fp16=(device != "cpu"))
        _bgem3_models[key] = model
    return model


class QueryEncoder:
    """질의를 (dense, sparse) 로 인코딩한다. sparse_enabled=False면 sparse=None.

    - bge_m3 : 하나의 bge-m3 모델로 dense + sparse 를 동시에 생성(모델 1회 로드).
    - 그 외   : dense 는 EmbeddingClient(원격/로컬), sparse 는 bm25(fastembed).
    sparse 라이브러리 미설치 등으로 실패하면 dense-only 로 안전하게 폴백한다.
    """

    def __init__(self, cfg: Settings = settings) -> None:
        self._cfg = cfg

    def encode(self, text: str) -> tuple:
        cfg = self._cfg

        # bge-m3: dense+sparse 동시 생성
        if cfg.sparse_enabled and cfg.sparse_method == "bge_m3":
            try:
                model = _get_bgem3_model(cfg.embed_local_model, cfg.embed_device)
                out = model.encode(
                    [text], return_dense=True, return_sparse=True, return_colbert_vecs=False
                )
                dense = out["dense_vecs"][0].tolist()
                lw = out["lexical_weights"][0]  # {token_id: weight}
                idx = [int(k) for k in lw.keys()]
                val = [float(v) for v in lw.values()]
                return dense, ((idx, val) if idx else None)
            except Exception as exc:  # noqa: BLE001 — sparse 실패 시 dense-only 폴백
                logger.warning("bge_m3 hybrid encode failed, dense-only 폴백: %s", exc)
                return EmbeddingClient(cfg=cfg).embed_one(text), None

        # 일반 dense
        dense = EmbeddingClient(cfg=cfg).embed_one(text)

        # bm25 sparse (옵션)
        if cfg.sparse_enabled and cfg.sparse_method == "bm25":
            try:
                enc = _get_bm25_encoder(cfg.sparse_model)
                se = list(enc.query_embed(text))[0]
                return dense, (se.indices.tolist(), se.values.tolist())
            except Exception as exc:  # noqa: BLE001
                logger.warning("bm25 sparse encode failed, dense-only 폴백: %s", exc)

        return dense, None


# ---------------------------------------------------------------------------
# 리랭커 (vLLM /score, Qwen3-Reranker-8B)
# 하네스 레포의 VLLMQwen3Reranker와 동일한 프롬프트 템플릿 사용 → 점수 호환.
# ---------------------------------------------------------------------------

@dataclass
class RerankResult:
    index: int      # 입력 documents 기준 원래 인덱스
    score: float


class RerankerClient:
    PREFIX = (
        '<|im_start|>system\nJudge whether the Document meets the requirements '
        'based on the Query and the Instruct provided. Note that the answer can '
        'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    DEFAULT_INSTRUCTION = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    def __init__(self, url: Optional[str] = None, batch_size: int = 32) -> None:
        self._url = (url or settings.rerank_url).rstrip("/")
        self._batch_size = batch_size

    def rerank(
        self, query: str, documents: List[str], instruction: Optional[str] = None
    ) -> List[RerankResult]:
        if not documents:
            return []
        instruction = instruction or self.DEFAULT_INSTRUCTION
        text_1 = f"{self.PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n"

        scores: List[float] = []
        for start in range(0, len(documents), self._batch_size):
            batch = documents[start : start + self._batch_size]
            payload = {
                "model": settings.rerank_model,
                "text_1": text_1,
                "text_2": [f"<Document>: {doc}{self.SUFFIX}" for doc in batch],
                "truncate_prompt_tokens": -1,
            }
            resp = requests.post(self._url, json=payload, timeout=settings.request_timeout)
            resp.raise_for_status()
            data = resp.json()["data"]
            scores.extend(float(item["score"]) for item in data)

        results = [RerankResult(index=i, score=s) for i, s in enumerate(scores)]
        results.sort(key=lambda r: r.score, reverse=True)
        return results


# ---------------------------------------------------------------------------
# 벡터스토어 (Qdrant)
# ---------------------------------------------------------------------------

@dataclass
class Hit:
    point_id: str       # Qdrant 포인트 UUID (검색 제외 필터에 사용)
    chunk_id: str       # 청크 ID (예: "26089_26") 또는 웹 ID("W1") — harness-1에 노출하는 DOCUMENT ID
    text: str           # 본문 (payload.document 또는 웹 스니펫/페이지)
    doc_id: str         # 문서 ID (corpus=payload.source, web=URL) — 인용/출처 표기에 사용
    score: float
    title: str = ""     # 웹 결과 제목 (코퍼스는 빈 문자열)
    file_name: str = "" # 원본 파일명 (kb_bsss 등에 존재; 없으면 "")
    page_no: str = ""   # 페이지 번호 (kb_bsss 등에 존재; 없으면 "")
    payload: Dict = field(default_factory=dict)  # Qdrant 원본 payload 전체(디버그/조회용)

    @property
    def source_label(self) -> str:
        """인용/표시용 출처 라벨.

        우선순위: 웹 결과는 title → 코퍼스는 file_name(+page_no) → 폴백 doc_id.
        kb_bsss(v2)처럼 file_name/page_no 가 있으면 이를 활용하고,
        해당 필드가 없는 코퍼스(예: v1 browsecompplus)는 기존대로 doc_id 로 폴백한다.
        """
        if self.title:
            return self.title
        if self.file_name:
            return f"{self.file_name} (p.{self.page_no})" if self.page_no else self.file_name
        return f"문서 {self.doc_id}"


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str


class WebSearchClient:
    """웹검색 + 페이지 읽기. provider: 'serpapi'(serpapi.com) 또는 'serper'(serper.dev).

    기존 eval_scripts/web_tools.py 의 WebSearchCorpusTool / WebReadDocumentTool 방식과 동일하며,
    SerpApi 백엔드를 추가로 지원한다. 페이지 본문은 Jina Reader(+serper scrape)로 가져온다.
    """

    def __init__(self) -> None:
        self._session = requests.Session()
        self._provider = settings.web_search_provider

    def search(self, query: str, num: Optional[int] = None) -> List[WebResult]:
        num = num or settings.web_search_limit
        if self._provider == "serpapi":
            return self._search_serpapi(query, num)
        return self._search_serper(query, num)

    def _search_serpapi(self, query: str, num: int) -> List[WebResult]:
        try:
            resp = self._session.get(
                settings.serpapi_url,
                params={"engine": "google", "q": query, "api_key": settings.serpapi_api_key, "num": num},
                timeout=settings.request_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("serpapi_search_error: %s", exc)
            return []
        if isinstance(data, dict) and data.get("error"):
            logger.warning("serpapi_error: %s", data["error"])
            return []
        results: List[WebResult] = []
        for item in data.get("organic_results", []):
            results.append(
                WebResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", "") or item.get("snippet_highlighted_words", ""),
                )
            )
        return results

    def _search_serper(self, query: str, num: int) -> List[WebResult]:
        headers = {"X-API-KEY": settings.serper_api_key, "Content-Type": "application/json"}
        try:
            resp = self._session.post(
                settings.serper_search_url,
                headers=headers,
                json={"q": query, "num": num},
                timeout=settings.request_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("serper_search_error: %s", exc)
            return []
        results: List[WebResult] = []
        for item in data.get("organic", []):
            results.append(
                WebResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                )
            )
        return results

    def read_page(self, url: str) -> str:
        # 1) Serper scrape (serper provider일 때만)
        if self._provider == "serper" and settings.serper_api_key:
            try:
                resp = self._session.post(
                    settings.serper_scrape_url,
                    headers={"X-API-KEY": settings.serper_api_key, "Content-Type": "application/json"},
                    json={"url": url},
                    timeout=settings.request_timeout,
                )
                resp.raise_for_status()
                text = resp.json().get("text")
                if text:
                    return text[: settings.web_read_max_chars]
            except (requests.RequestException, ValueError) as exc:
                logger.warning("serper_scrape_error %s: %s", url, exc)

        # 2) Jina Reader
        try:
            headers = {
                "X-Retain-Images": "none",
                "X-Remove-Selector": "header, nav, footer, aside, .sidebar, .comments, .social",
            }
            if settings.jina_api_key:
                headers["Authorization"] = f"Bearer {settings.jina_api_key}"
            resp = self._session.get(
                f"https://r.jina.ai/{url}", headers=headers, timeout=settings.request_timeout
            )
            resp.raise_for_status()
            text = resp.text
            marker = "Markdown Content:\n"
            if marker in text:
                text = text.split(marker, 1)[1]
            return text[: settings.web_read_max_chars]
        except requests.RequestException as exc:
            logger.warning("jina_fetch_error %s: %s", url, exc)
            return f"(웹페이지를 가져오지 못했습니다: {url})"


class VectorStore:
    def __init__(self, client: Optional[QdrantClient] = None, cfg: Settings = settings) -> None:
        self._cfg = cfg
        self._client = client or QdrantClient(
            url=cfg.qdrant_url,
            api_key=cfg.qdrant_api_key or None,
            timeout=cfg.request_timeout,
            check_compatibility=False,  # 클라이언트/서버 마이너 버전 차이 경고 억제
        )
        self._collection = cfg.qdrant_collection

    def _payload_to_hit(self, point_id, payload: dict, score: float) -> Hit:
        payload = payload or {}
        text = (
            payload.get(self._cfg.qdrant_text_field)
            or payload.get("document")
            or payload.get("text")
            or ""
        )
        doc_id = (
            payload.get(self._cfg.qdrant_docid_field)
            or payload.get("source")
            or payload.get("doc_id")
            or str(point_id)
        )
        chunk_id = (
            payload.get(self._cfg.qdrant_chunkid_field)
            or payload.get("chunk_id")
            or str(point_id)
        )
        return Hit(
            point_id=str(point_id),
            chunk_id=str(chunk_id),
            text=str(text),
            doc_id=str(doc_id),
            score=float(score),
            file_name=str(payload.get("file_name") or ""),
            page_no=str(payload.get("page_no") or ""),
            payload=dict(payload),
        )

    def search(
        self,
        vector: List[float],
        limit: int,
        exclude_ids: Optional[List[str]] = None,
        sparse: Optional[tuple] = None,
    ) -> List[Hit]:
        from qdrant_client import models as qm

        query_filter = None
        if exclude_ids:
            query_filter = qm.Filter(
                must_not=[qm.HasIdCondition(has_id=list(exclude_ids))]
            )

        # 하이브리드: dense + sparse 를 각각 prefetch 한 뒤 RRF 로 융합
        if sparse is not None and self._cfg.sparse_enabled and self._cfg.sparse_vector_name:
            indices, values = sparse
            sv = qm.SparseVector(indices=list(indices), values=list(values))
            pf = self._cfg.hybrid_prefetch_limit
            response = self._client.query_points(
                collection_name=self._collection,
                prefetch=[
                    qm.Prefetch(query=vector, using=self._cfg.qdrant_vector_name,
                                limit=pf, filter=query_filter),
                    qm.Prefetch(query=sv, using=self._cfg.sparse_vector_name,
                                limit=pf, filter=query_filter),
                ],
                query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
            return [self._payload_to_hit(p.id, p.payload, p.score) for p in response.points]

        # dense-only (qdrant-client 1.10+ query_points, named vector는 using= 으로 지정)
        kwargs = dict(
            collection_name=self._collection,
            query=vector,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        )
        if self._cfg.qdrant_using_named_vector:
            kwargs["using"] = self._cfg.qdrant_vector_name
        response = self._client.query_points(**kwargs)
        return [self._payload_to_hit(p.id, p.payload, p.score) for p in response.points]

    def fetch(self, point_ids: List[str]) -> List[Hit]:
        if not point_ids:
            return []
        records = self._client.retrieve(
            collection_name=self._collection,
            ids=point_ids,
            with_payload=True,
        )
        return [self._payload_to_hit(r.id, r.payload, 0.0) for r in records]

    def sample_payload_keys(self) -> dict:
        """컬렉션 구조 점검용: 샘플 1건의 payload 키와 벡터 설정을 반환."""
        info = self._client.get_collection(self._collection)
        sample_keys: List[str] = []
        try:
            recs, _ = self._client.scroll(
                collection_name=self._collection, limit=1, with_payload=True
            )
            if recs:
                sample_keys = list((recs[0].payload or {}).keys())
        except Exception as exc:  # pragma: no cover
            logger.warning("scroll failed: %s", exc)
        return {
            "collection": self._collection,
            "points_count": getattr(info, "points_count", None),
            "vectors_config": str(getattr(info.config.params, "vectors", None)),
            "sample_payload_keys": sample_keys,
        }
