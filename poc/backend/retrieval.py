"""harness-1 검색 서브에이전트 루프.

흐름: 질의 임베딩(qwen3) → Qdrant 벡터 검색 → 리랭크(Qwen3-Reranker-8B)
     → harness-1이 도구를 반복 호출하며 관련 문서를 랭킹.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .clients import (
    Hit,
    QueryEncoder,
    RerankerClient,
    VectorStore,
    WebSearchClient,
    make_harness1_client,
)
from .config import Settings, settings
from .prompts import retrieval_subagent_prompt

logger = logging.getLogger("poc.retrieval")

# 검색 결과 본문이 너무 길면 잘라서 모델 컨텍스트를 보호
SNIPPET_MAX_CHARS = 2000

Emit = Callable[[dict], None]


# Responses API용 도구 스키마 (flat 포맷).
# 기존 하네스 프로젝트의 OpenAIAgentInferenceModel(api_style="responses")와 동일한 방식.
SEARCH_CORPUS_TOOL = {
    "type": "function",
    "name": "search_corpus",
    "description": (
        "Searches the INTERNAL document corpus (private knowledge base) for relevant documents. "
        "Use this for domain/internal knowledge. Returns document sections, each with a DOCUMENT ID."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant documents in the internal corpus.",
            }
        },
        "required": ["query"],
    },
}

WEB_SEARCH_TOOL = {
    "type": "function",
    "name": "web_search",
    "description": (
        "Searches the PUBLIC WEB via Google. Use this for current events, general knowledge, "
        "or anything not likely to be in the internal corpus. Returns web results, each with a "
        "DOCUMENT ID like 'W1' plus title, URL and snippet."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The web search query.",
            }
        },
        "required": ["query"],
    },
}

READ_DOCUMENT_TOOL = {
    "type": "function",
    "name": "read_document",
    "description": (
        "Reads the full content of a document by its DOCUMENT ID. Works for both internal corpus "
        "chunk IDs (from search_corpus) and web result IDs like 'W1' (from web_search, fetches the page)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "doc_id": {
                "type": "string",
                "description": "The DOCUMENT ID from search_corpus or web_search.",
            }
        },
        "required": ["doc_id"],
    },
}


@dataclass
class RetrievalResult:
    ranked_chunks: List[Hit]          # 최종 랭킹된 청크 (합성에 사용)
    all_hits: Dict[str, Hit]          # 루프 동안 본 모든 청크 (point_id -> Hit)
    harness_output: str               # harness-1 최종 텍스트(랭킹 목록)
    n_searches: int = 0


class SearchEngine:
    """임베딩 + Qdrant + 리랭커를 묶은 검색 도구.

    cfg 로 버전별 임베딩/벡터스토어 설정을 주입한다(기본: 전역 settings).
    리랭커는 v1·v2 공유 자원이므로 전역 settings 를 그대로 사용한다.
    """

    def __init__(self, cfg: Settings = settings) -> None:
        self._cfg = cfg
        self.encoder = QueryEncoder(cfg=cfg)
        self.store = VectorStore(cfg=cfg)
        self.reranker = RerankerClient() if settings.rerank_enabled else None

    def search(
        self, query: str, ignore_ids: Optional[List[str]], emit: Emit
    ) -> Tuple[str, List[Hit]]:
        emit({"type": "status", "phase": "embedding", "message": f"질의 임베딩 생성: \"{query}\""})
        vector, sparse = self.encoder.encode(query)

        if sparse is not None and self._cfg.sparse_enabled:
            emit({"type": "status", "phase": "searching",
                  "message": f"Qdrant 하이브리드 검색 중… (dense+{self._cfg.sparse_vector_name}, RRF 융합)"})
        else:
            emit({"type": "status", "phase": "searching", "message": "Qdrant 벡터 검색 중…"})
        hits = self.store.search(vector, limit=self._cfg.search_limit,
                                 exclude_ids=ignore_ids, sparse=sparse)

        if self.reranker and hits:
            emit({"type": "status", "phase": "reranking", "message": f"리랭커 재정렬 중… (후보 {len(hits)}건)"})
            order = self.reranker.rerank(query, [h.text for h in hits])
            hits = [hits[r.index] for r in order]

        hits = hits[: self._cfg.rerank_top_k]

        emit({
            "type": "search_result",
            "query": query,
            "hits": [
                {"doc_id": h.doc_id, "chunk_id": h.chunk_id, "score": round(h.score, 4),
                 "snippet": h.text[:240]}
                for h in hits
            ],
        })

        formatted = self._format(hits)
        return formatted, hits

    @staticmethod
    def _format(hits: List[Hit]) -> str:
        if not hits:
            return "No results found"
        parts = [
            f"\n# DOCUMENT ID: {h.chunk_id}\n{h.text[:SNIPPET_MAX_CHARS]}"
            for h in hits
        ]
        return "\n".join(parts)


def _parse_ranked_ids(text: str) -> List[str]:
    """harness-1 최종 출력에서 <Document id=...> 의 id 목록을 추출."""
    ids = re.findall(r"<Document\s+id=([^\s>]+)\s*>", text)
    # id 에 따옴표가 붙어있을 수 있어 정리
    return [i.strip().strip('"').strip("'") for i in ids]


def _extract_message_text(output_item) -> str:
    """Responses API message 아이템에서 텍스트를 추출."""
    parts: List[str] = []
    for content in getattr(output_item, "content", []) or []:
        text = getattr(content, "text", None)
        if text:
            parts.append(text)
        else:
            refusal = getattr(content, "refusal", None)
            if refusal:
                parts.append(refusal)
    return "".join(parts).strip()


def run_harness1_retrieval(question: str, emit: Emit, cfg: Settings = settings) -> RetrievalResult:
    """harness-1을 도구와 함께 구동해 관련 문서를 랭킹한다.

    기존 하네스 프로젝트의 OpenAIAgentInferenceModel(api_style="responses")과 동일하게
    OpenAI **Responses API** + function tool 루프를 사용한다.
    (이 서버의 chat completions 도구 호출은 gpt-oss tool 파서 미설정으로 동작하지 않음)

    cfg 로 버전별 임베딩/벡터스토어 설정을 주입한다(기본: 전역 settings = v1).
    harness-1·웹검색은 v1·v2 공유 자원이라 전역 settings 를 그대로 사용한다.
    """
    engine = SearchEngine(cfg=cfg)
    client = make_harness1_client()

    web_enabled = settings.web_search_enabled and bool(settings.web_api_key)
    web_client = WebSearchClient() if web_enabled else None

    all_hits: Dict[str, Hit] = {}        # id(chunk_id 또는 'W{n}') -> Hit
    seen_point_ids: List[str] = []       # Qdrant 검색 제외용 point id 목록
    n_searches = 0
    web_counter = 0                      # 웹 결과 id 'W{n}' 발번용

    tools = [SEARCH_CORPUS_TOOL, READ_DOCUMENT_TOOL]
    if web_enabled:
        tools.insert(1, WEB_SEARCH_TOOL)  # search_corpus 다음에 web_search 노출
    # 누적 입력(full-resend): system/user + (function_call, function_call_output)* 를 매 턴 재전송
    input_items: List[dict] = [
        {"role": "system", "content": retrieval_subagent_prompt(question, web_enabled=web_enabled)},
        {"role": "user", "content": f"Find documents for: {question}"},
    ]

    emit({"type": "status", "phase": "agent", "message": "harness-1 검색 에이전트 시작 (Responses API)"})

    final_text = ""
    for turn in range(settings.harness1_max_turns):
        resp = client.responses.create(
            model=settings.harness1_model,
            input=input_items,
            tools=tools,
            parallel_tool_calls=True,
            temperature=settings.harness1_temperature,
            max_output_tokens=settings.harness1_max_tokens,
        )

        function_calls = [o for o in (resp.output or []) if getattr(o, "type", None) == "function_call"]

        if not function_calls:
            # 도구 호출이 없으면 최종 응답(message)에서 랭킹 텍스트 추출
            for o in resp.output or []:
                if getattr(o, "type", None) == "message":
                    final_text = _extract_message_text(o)
            break

        # 1) 모델이 낸 function_call 아이템들을 히스토리에 추가
        for fc in function_calls:
            input_items.append({
                "type": "function_call",
                "call_id": fc.call_id,
                "name": fc.name,
                "arguments": fc.arguments,
            })

        # 2) 각 도구를 실행하고 결과(function_call_output)를 히스토리에 추가
        for fc in function_calls:
            try:
                args = json.loads(fc.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if fc.name == "search_corpus":
                n_searches += 1
                query = args.get("query", question)
                emit({"type": "step", "tool": "search_corpus", "turn": turn + 1, "query": query})
                formatted, hits = engine.search(query, ignore_ids=seen_point_ids, emit=emit)
                for h in hits:
                    if h.chunk_id not in all_hits:
                        all_hits[h.chunk_id] = h
                    if h.point_id not in seen_point_ids:
                        seen_point_ids.append(h.point_id)
                tool_out = formatted
            elif fc.name == "web_search":
                n_searches += 1
                query = args.get("query", question)
                emit({"type": "step", "tool": "web_search", "turn": turn + 1, "query": query})
                if web_client is None:
                    tool_out = "web_search is not available."
                else:
                    results = web_client.search(query)
                    web_lines: List[str] = []
                    emitted_hits = []
                    for r in results:
                        web_counter += 1
                        wid = f"W{web_counter}"
                        # 웹 결과를 Hit 형태로 보관(text=snippet, doc_id=url)
                        all_hits[wid] = Hit(
                            point_id=wid, chunk_id=wid,
                            text=f"{r.title}\n{r.snippet}", doc_id=r.url, score=0.0,
                            title=r.title,
                        )
                        web_lines.append(
                            f"\n# DOCUMENT ID: {wid}\nTitle: {r.title}\nURL: {r.url}\n{r.snippet}"
                        )
                        emitted_hits.append({"doc_id": r.url, "chunk_id": wid,
                                             "score": 0.0, "snippet": r.snippet[:240]})
                    emit({"type": "search_result", "query": query, "source": "web", "hits": emitted_hits})
                    tool_out = "\n".join(web_lines) if web_lines else "No web results found."
            elif fc.name == "read_document":
                doc_id = str(args.get("doc_id", ""))
                emit({"type": "step", "tool": "read_document", "turn": turn + 1, "doc_id": doc_id})
                hit = all_hits.get(doc_id)
                if hit is None:
                    tool_out = f"Document {doc_id} not found."
                elif doc_id.startswith("W") and web_client is not None:
                    # 웹 결과는 실제 페이지 본문을 가져와 갱신
                    body = web_client.read_page(hit.doc_id)
                    hit.text = f"{hit.text}\n\n{body}"
                    tool_out = body
                else:
                    tool_out = hit.text
            else:
                tool_out = f"Unknown tool: {fc.name}"

            input_items.append({
                "type": "function_call_output",
                "call_id": fc.call_id,
                "output": tool_out,
            })
    else:
        emit({"type": "status", "phase": "agent", "message": "최대 턴 도달, 검색 종료"})

    # 최종 랭킹 해석
    ranked_ids = _parse_ranked_ids(final_text)
    ranked_chunks: List[Hit] = []
    for pid in ranked_ids:
        if pid in all_hits:
            ranked_chunks.append(all_hits[pid])

    # harness-1이 형식대로 출력하지 않은 경우: 본 순서대로 폴백
    if not ranked_chunks:
        ranked_chunks = list(all_hits.values())

    ranked_chunks = ranked_chunks[: cfg.synth_top_k]

    emit({
        "type": "ranked",
        "documents": [
            {"doc_id": h.doc_id, "chunk_id": h.chunk_id, "snippet": h.text[:240]}
            for h in ranked_chunks
        ],
    })

    return RetrievalResult(
        ranked_chunks=ranked_chunks,
        all_hits=all_hits,
        harness_output=final_text,
        n_searches=n_searches,
    )
