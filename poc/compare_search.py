"""하이브리드(dense+sparse) vs dense-only 검색 품질 비교 스크립트.

같은 질의로 두 방식을 돌려 상위 결과를 나란히 보여주고, 겹침/순위 변화/sparse 기여를 요약한다.

사용법 (poc/.venv 파이썬으로):
    .venv/Scripts/python.exe compare_search.py "학교 급식 보관 온도" --version v2 --k 8
    .venv/Scripts/python.exe compare_search.py "capital of China" --version v1 --k 8 --rerank

옵션:
    --version v1|v2   조회 대상 (기본 v2 = kb_bsss @ 121.78.130.83)
    --k N             상위 N건 비교 (기본 8)
    --rerank          두 방식 모두 Qwen3-Reranker 재정렬 후 비교
"""
from __future__ import annotations

import argparse
import io
import sys

# Windows(cp949)에서 한글/유니코드 출력 깨짐 방지
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from backend.app import EmbedRequest, _embed_search  # noqa: E402
from backend.config import settings_v1, settings_v2  # noqa: E402

GREEN = "\033[32m"
YELLOW = "\033[33m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _key(r: dict) -> str:
    """결과 동일성 판단 키 — chunk_id 우선, 없으면 doc_id."""
    return r.get("chunk_id") or r.get("doc_id") or ""


def _label(r: dict) -> str:
    """사람이 읽기 좋은 식별 라벨 (file_name 있으면 그걸로, 없으면 chunk_id)."""
    fn = r.get("file_name") or ""
    if fn:
        pg = r.get("page_no") or ""
        return f"{fn}" + (f" p{pg}" if pg else "")
    return r.get("chunk_id") or r.get("doc_id") or "?"


def _print_results(title: str, res: dict) -> None:
    mode = res["mode"]
    color = GREEN if mode == "hybrid" else YELLOW
    print(f"\n{BOLD}{color}■ {title}{RESET}  "
          f"{DIM}(mode={mode}, sparse={res.get('sparse_vector')}, reranked={res['reranked']}, "
          f"count={res['count']}){RESET}")
    for r in res["results"]:
        snip = (r["snippet"] or "").replace("\n", " ")[:70]
        print(f"  {r['rank']:>2}. score={r['score']:<8} {DIM}{_label(r)}{RESET}")
        print(f"      {DIM}{snip}{RESET}")


def compare(cfg, query: str, k: int, rerank: bool) -> None:
    dense = _embed_search(cfg, EmbedRequest(query=query, limit=k, hybrid=False, rerank=rerank))
    hybrid = _embed_search(cfg, EmbedRequest(query=query, limit=k, hybrid=True, rerank=rerank))

    print(f"{BOLD}질의:{RESET} {query}")
    print(f"{DIM}컬렉션: {dense['collection']} @ {dense['qdrant_url']}{RESET}")
    if hybrid["mode"] != "hybrid":
        print(f"{YELLOW}⚠ 이 버전은 sparse가 비활성(또는 인코더 로드 실패)이라 hybrid가 dense로 폴백됨.{RESET}")

    _print_results("dense-only", dense)
    _print_results("hybrid (dense+sparse, RRF)", hybrid)

    # ── 비교 요약 ──────────────────────────────────────────────
    d_keys = [_key(r) for r in dense["results"]]
    h_keys = [_key(r) for r in hybrid["results"]]
    d_set, h_set = set(d_keys), set(h_keys)
    common = d_set & h_set
    only_hybrid = [r for r in hybrid["results"] if _key(r) not in d_set]
    only_dense = [r for r in dense["results"] if _key(r) not in h_set]

    print(f"\n{BOLD}■ 비교 요약 (상위 {k}건){RESET}")
    print(f"  공통: {len(common)}건 / {k}  ·  하이브리드 단독: {len(only_hybrid)}건  ·  dense 단독: {len(only_dense)}건")

    if only_hybrid:
        print(f"\n  {GREEN}하이브리드에서 새로 올라온 문서 (sparse 기여 가능){RESET}")
        for r in only_hybrid:
            print(f"    +{r['rank']:>2}. {_label(r)}  {DIM}{(r['snippet'] or '')[:50]}{RESET}")
    if only_dense:
        print(f"\n  {YELLOW}하이브리드에서 빠진(밀려난) dense-only 문서{RESET}")
        for r in only_dense:
            print(f"    -{r['rank']:>2}. {_label(r)}  {DIM}{(r['snippet'] or '')[:50]}{RESET}")

    # 공통 문서의 순위 변화
    moves = []
    for kk in common:
        dr = d_keys.index(kk) + 1
        hr = h_keys.index(kk) + 1
        if dr != hr:
            moves.append((kk, dr, hr))
    if moves:
        print(f"\n  {DIM}공통 문서 순위 변화 (dense→hybrid):{RESET}")
        for kk, dr, hr in sorted(moves, key=lambda x: x[2]):
            arrow = "▲" if hr < dr else "▼"
            lbl = next((_label(r) for r in hybrid["results"] if _key(r) == kk), kk)
            print(f"    {arrow} {dr}→{hr}  {lbl}")


def main() -> None:
    ap = argparse.ArgumentParser(description="하이브리드 vs dense-only 검색 비교")
    ap.add_argument("query", help="검색 질의")
    ap.add_argument("--version", choices=["v1", "v2"], default="v2",
                    help="조회 대상 (기본 v2=kb_bsss)")
    ap.add_argument("--k", type=int, default=8, help="상위 N건 (기본 8)")
    ap.add_argument("--rerank", action="store_true", help="두 방식 모두 리랭커 적용")
    args = ap.parse_args()

    cfg = settings_v1 if args.version == "v1" else settings_v2
    compare(cfg, args.query, args.k, args.rerank)


if __name__ == "__main__":
    main()
