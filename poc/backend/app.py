"""FastAPI 앱: 아리아 RAG POC.

- POST /v1/chat, /v2/chat : 질문을 받아 검색→합성 과정을 SSE로 스트리밍
- GET  /v1/health, /v2/health : 각 백엔드 서비스 연결 상태 점검
- GET  /v1/inspect, /v2/inspect : Qdrant 컬렉션 구조 점검(payload 필드명 확인용)
- GET/POST /v1/embed, /v2/embed : 벡터스토어 검색 UI(GET=페이지) + 데이터(POST JSON, GET ?format=json)
- GET/POST /v2/comp : 하이브리드 vs dense-only 비교 UI(GET=페이지) + 데이터(POST=JSON)
- GET  /, /v1, /v2 : 프론트엔드(아리아 ERP 디자인) 서빙

버전 차이(임베딩 모델 + 벡터스토어):
- v1: settings_v1 — 임베딩 qwen3-embedding(원격, 1024d) + 벡터스토어 192.168.1.10:6333/browsecompplus_qwen3_1024(dense)
- v2: settings_v2 — 임베딩 BAAI/bge-m3(로컬, 1024d) + 벡터스토어 121.78.130.83:6333/kb_bsss(dense)
harness-1·리랭커·오케스트레이터·웹검색은 두 버전이 공유한다(전역 settings/.env).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path
from typing import Iterator

import requests
from fastapi import APIRouter, FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dataclasses import replace

from .clients import QueryEncoder, RerankerClient, VectorStore
from .config import Settings, settings, settings_v1, settings_v2
from .orchestrator import synthesize_stream, translate_query_to_english
from .retrieval import run_harness1_retrieval

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("poc.app")

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

app = FastAPI(title="아리아 RAG POC", description="harness-1 기반 검색 RAG POC")

v1 = APIRouter(prefix="/v1", tags=["v1"])


class ChatRequest(BaseModel):
    question: str


class EmbedRequest(BaseModel):
    query: str
    limit: int = 10
    hybrid: bool = True     # True=하이브리드(dense+sparse), False=dense-only
    rerank: bool = False    # True면 Qwen3-Reranker로 재정렬


class CompareRequest(BaseModel):
    query: str
    k: int = 8
    rerank: bool = False
    version: str = "v2"     # "v1" | "v2" — 비교 대상 백엔드


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def event_stream(question: str, cfg: Settings = settings) -> Iterator[str]:
    q: "queue.Queue[tuple]" = queue.Queue()
    box: dict = {}

    def emit(ev: dict) -> None:
        q.put(("event", ev))

    def worker() -> None:
        try:
            # 1) (옵션) 검색 전 질의를 영어로 번역 — TRANSLATE_QUERY_TO_ENGLISH 로 제어
            if cfg.translate_query_enabled:
                emit({"type": "status", "phase": "translate", "message": "질의를 영어로 번역 중…"})
                search_q = translate_query_to_english(question)
                emit({"type": "translated", "original": question, "english": search_q})
            else:
                search_q = question

            # 2) harness-1 검색 (번역 비활성화 시 원문 그대로)
            result = run_harness1_retrieval(search_q, emit, cfg=cfg)
            box["result"] = result
            q.put(("retrieval_done", None))
        except Exception as exc:  # noqa: BLE001
            logger.exception("retrieval failed")
            q.put(("error", f"검색 단계 오류: {exc}"))

    threading.Thread(target=worker, daemon=True).start()

    # 1) 검색 단계 이벤트 스트리밍
    while True:
        kind, payload = q.get()
        if kind == "event":
            yield _sse(payload)
        elif kind == "error":
            yield _sse({"type": "error", "message": payload})
            return
        elif kind == "retrieval_done":
            break

    result = box.get("result")
    if result is None or not result.ranked_chunks:
        yield _sse({"type": "answer_start"})
        yield _sse({"type": "token", "text": "관련 문서를 찾지 못해 답변을 생성할 수 없습니다."})
        yield _sse({"type": "done", "citations": [], "n_searches": getattr(result, "n_searches", 0)})
        return

    # 2) 답변 합성 스트리밍
    yield _sse({"type": "answer_start"})
    try:
        for token in synthesize_stream(question, result.ranked_chunks):
            yield _sse({"type": "token", "text": token})
    except Exception as exc:  # noqa: BLE001
        logger.exception("synthesis failed")
        yield _sse({"type": "error", "message": f"답변 합성 오류: {exc}"})
        return

    citations = []
    for i, c in enumerate(result.ranked_chunks, 1):
        is_web = c.chunk_id.startswith("W")
        citations.append({
            "n": i,
            # 코퍼스는 file_name(+page_no) 우선, 없으면 doc_id 폴백 (Hit.source_label)
            "label": c.source_label,
            "url": c.doc_id if is_web else "",
            "is_web": is_web,
            "file_name": c.file_name,
            "page_no": c.page_no,
            "snippet": c.text[:300],
        })
    yield _sse({"type": "done", "citations": citations, "n_searches": result.n_searches})


# ---------------------------------------------------------------------------
# 버전 공유 핸들러 — cfg(Settings) 로 v1/v2 백엔드 설정을 주입한다.
#   v1: 전역 settings,  v2: settings_v2 (임베딩 bge-m3 + 벡터스토어 kb_bsss)
# ---------------------------------------------------------------------------

def _chat(req: ChatRequest, cfg: Settings) -> StreamingResponse:
    return StreamingResponse(
        event_stream(req.question, cfg=cfg),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _health(cfg: Settings) -> dict:
    """각 백엔드 서비스 연결을 짧은 타임아웃으로 점검한다(다운된 호스트에서 멈추지 않도록)."""
    from openai import OpenAI
    from qdrant_client import QdrantClient

    status: dict = {}
    T = 8  # 초

    def check(name: str, fn) -> None:
        try:
            fn()
            status[name] = {"ok": True}
        except Exception as exc:  # noqa: BLE001
            status[name] = {"ok": False, "error": str(exc)}

    if cfg.embed_backend == "local_bge_m3":
        # 로컬 임베딩: 2GB 모델을 health에서 다 로드하지 않고 라이브러리만 확인
        check("embedding", lambda: __import__("sentence_transformers"))
    else:
        check("embedding", lambda: OpenAI(base_url=cfg.embed_base_url,
              api_key=cfg.embed_api_key, timeout=T).models.list())
    check("harness1", lambda: OpenAI(base_url=cfg.harness1_base_url,
          api_key=cfg.harness1_api_key, timeout=T).models.list())
    check("orchestrator", lambda: OpenAI(base_url=cfg.orch_base_url,
          api_key=cfg.orch_api_key, timeout=T).models.list())
    check("qdrant", lambda: QdrantClient(url=cfg.qdrant_url,
          api_key=cfg.qdrant_api_key or None, timeout=T).get_collection(cfg.qdrant_collection))
    check(
        "reranker",
        lambda: requests.post(
            cfg.rerank_url,
            json={"model": cfg.rerank_model, "text_1": "ping", "text_2": ["pong"],
                  "truncate_prompt_tokens": -1},
            timeout=T,
        ).raise_for_status(),
    )
    status["all_ok"] = all(v.get("ok") for k, v in status.items() if isinstance(v, dict))
    embed_desc = (cfg.embed_local_model + " (local)") if cfg.embed_backend == "local_bge_m3" \
        else (cfg.embed_base_url + " (" + cfg.embed_model + ")")
    status["config"] = {
        "harness1": cfg.harness1_base_url + " (" + cfg.harness1_model + ")",
        "embedding": embed_desc,
        "reranker": cfg.rerank_url + " (" + cfg.rerank_model + ")",
        "orchestrator": cfg.orch_base_url + " (" + cfg.orch_model + ")",
        "qdrant": cfg.qdrant_url + " (" + cfg.qdrant_collection + ")",
    }
    return status


def _inspect(cfg: Settings) -> dict:
    """Qdrant 컬렉션의 payload 필드/벡터 구성을 확인한다."""
    return VectorStore(cfg=cfg).sample_payload_keys()


def _embed_search(cfg: Settings, req: EmbedRequest) -> dict:
    """질의를 임베딩해 벡터스토어를 직접 조회한다(harness/합성 없이 원시 검색 결과 반환).

    hybrid=False면 dense-only로 강제, rerank=True면 Qwen3-Reranker로 재정렬한다.
    하이브리드 vs dense-only 비교·디버깅 용도.
    """
    eff = cfg if req.hybrid else replace(cfg, sparse_enabled=False)
    dense, sparse = QueryEncoder(cfg=eff).encode(req.query)
    used_hybrid = sparse is not None and eff.sparse_enabled

    # rerank 시 더 넓게 후보를 뽑은 뒤 재정렬해 limit 으로 자른다.
    cand_n = max(req.limit, cfg.search_limit) if req.rerank else req.limit
    hits = VectorStore(cfg=eff).search(dense, limit=cand_n, exclude_ids=None, sparse=sparse)

    reranked = False
    if req.rerank and hits and cfg.rerank_enabled:
        order = RerankerClient().rerank(req.query, [h.text for h in hits])
        hits = [hits[r.index] for r in order]
        reranked = True
    hits = hits[: req.limit]

    return {
        "query": req.query,
        "collection": eff.qdrant_collection,
        "qdrant_url": eff.qdrant_url,
        "mode": "hybrid" if used_hybrid else "dense",
        "sparse_vector": eff.sparse_vector_name if used_hybrid else None,
        "reranked": reranked,
        "count": len(hits),
        "results": [
            {
                "rank": i,
                "chunk_id": h.chunk_id,
                "doc_id": h.doc_id,
                "file_name": h.file_name,
                "page_no": h.page_no,
                "score": round(h.score, 4),
                "snippet": h.text[:300],
                "payload": h.payload,
            }
            for i, h in enumerate(hits, 1)
        ],
    }


def _embed_get(cfg: Settings, query: str, limit: int, hybrid: bool, rerank: bool) -> dict:
    """GET(쿼리 파라미터) 접근용. query 가 비면 사용법 힌트를 반환한다."""
    if not query:
        return {
            "usage": "GET /v2/embed?query=질의&hybrid=true&limit=10&rerank=false (또는 POST JSON)",
            "collection": cfg.qdrant_collection,
            "qdrant_url": cfg.qdrant_url,
        }
    return _embed_search(cfg, EmbedRequest(query=query, limit=limit, hybrid=hybrid, rerank=rerank))


def _compare(cfg: Settings, req: CompareRequest) -> dict:
    """같은 질의로 dense-only 와 hybrid 결과를 함께 반환한다(/v2/comp UI 용)."""
    dense = _embed_search(cfg, EmbedRequest(query=req.query, limit=req.k, hybrid=False, rerank=req.rerank))
    hybrid = _embed_search(cfg, EmbedRequest(query=req.query, limit=req.k, hybrid=True, rerank=req.rerank))
    return {
        "query": req.query,
        "version": req.version,
        "collection": dense["collection"],
        "qdrant_url": dense["qdrant_url"],
        "rerank": req.rerank,
        "hybrid_available": hybrid["mode"] == "hybrid",
        "dense": dense,
        "hybrid": hybrid,
    }


# --- v1 라우터 (임베딩 qwen3 + 벡터스토어 browsecompplus_qwen3_1024) ---
@v1.post("/chat")
def chat_v1(req: ChatRequest) -> StreamingResponse:
    return _chat(req, settings_v1)


@v1.get("/health")
def health_v1() -> dict:
    return _health(settings_v1)


@v1.get("/inspect")
def inspect_v1() -> dict:
    return _inspect(settings_v1)


@v1.post("/embed")
def embed_v1(req: EmbedRequest) -> dict:
    return _embed_search(settings_v1, req)


@v1.get("/embed")
def embed_v1_get(query: str = "", limit: int = 10, hybrid: bool = True,
                 rerank: bool = False, format: str = ""):
    """GET=검색 UI 페이지. ?format=json 이면 JSON 결과를 반환(curl용)."""
    if format == "json":
        return JSONResponse(_embed_get(settings_v1, query, limit, hybrid, rerank))
    return FileResponse(str(FRONTEND_DIR / "embed.html"))


# --- v2 라우터 (임베딩 BAAI/bge-m3 + 벡터스토어 kb_bsss) ---
v2 = APIRouter(prefix="/v2", tags=["v2"])


@v2.post("/chat")
def chat_v2(req: ChatRequest) -> StreamingResponse:
    return _chat(req, settings_v2)


@v2.get("/health")
def health_v2() -> dict:
    return _health(settings_v2)


@v2.get("/inspect")
def inspect_v2() -> dict:
    return _inspect(settings_v2)


@v2.post("/embed")
def embed_v2(req: EmbedRequest) -> dict:
    """kb_bsss(121.78.130.83:6333) 벡터스토어를 직접 조회한다."""
    return _embed_search(settings_v2, req)


@v2.get("/embed")
def embed_v2_get(query: str = "", limit: int = 10, hybrid: bool = True,
                 rerank: bool = False, format: str = ""):
    """GET=kb_bsss 검색 UI 페이지. ?format=json 이면 JSON 결과를 반환(curl용)."""
    if format == "json":
        return JSONResponse(_embed_get(settings_v2, query, limit, hybrid, rerank))
    return FileResponse(str(FRONTEND_DIR / "embed.html"))


@v2.get("/comp")
def comp_page() -> FileResponse:
    """하이브리드 vs dense-only 비교 UI 페이지."""
    return FileResponse(str(FRONTEND_DIR / "comp.html"))


@v2.post("/comp")
def comp_data(req: CompareRequest) -> dict:
    """비교 데이터(dense/hybrid 두 결과)를 JSON으로 반환한다."""
    cfg = settings_v1 if req.version == "v1" else settings_v2
    return _compare(cfg, req)


app.include_router(v1)
app.include_router(v2)


@app.get("/")
@app.get("/v1")
@app.get("/v1/")
@app.get("/v2")
@app.get("/v2/")
def index() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
