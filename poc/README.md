# 아리아 RAG POC — harness-1 검색 기반 RAG

harness-1(검색 서브에이전트)을 중심으로, 제공된 자체 호스팅 인프라(임베딩·리랭커·
오케스트레이터·Qdrant)를 묶은 검색 RAG POC입니다. UI는 "에이전틱 ERP(아리아)"
디자인 톤을 그대로 적용한 자연어 검색 채팅입니다.

## 아키텍처

```
사용자 질문 (웹 UI)
   │
   ▼
오케스트레이터 (exaone3.5:32b)        ── 최종 한국어 답변을 출처 기반으로 합성/스트리밍
   │
   ▼
harness-1 (검색 서브에이전트)          ── search_corpus/read_document 도구를 반복 호출해 관련 문서 랭킹
   ├── 임베딩 (qwen3-embedding, 1024d) ── 질의 벡터화
   ├── Qdrant (browsecompplus_qwen3_1024) ── 벡터 검색
   └── 리랭커 (Qwen3-Reranker-8B, /score) ── 후보 재정렬
```

- **harness-1**은 원본 레포 설계대로 *질문에 직접 답하지 않고* 관련 문서를 찾아
  `<Document id=...>` 형식으로 랭킹합니다(`harness/prompts.py`의 retrieval subagent 프롬프트와 동일).
- **오케스트레이터**가 랭킹된 문서 발췌만을 근거로 한국어 답변을 생성하고 `[doc_id]`로 인용합니다.

## 구성 요소(엔드포인트)

| 역할 | 모델 | URL |
|------|------|-----|
| 검색 모델 | harness-1 | http://192.168.1.108:8000/v1 |
| 임베딩 | qwen3-embedding (1024d) | http://192.168.1.108:8081/v1 |
| 리랭커 | Qwen/Qwen3-Reranker-8B | http://192.168.1.108:8003/score |
| 오케스트레이터 | exaone3.5:32b (Ollama) | http://192.168.1.108:11434/v1 |
| 벡터스토어 | Qdrant | http://192.168.1.10:6333 |

컬렉션: `browsecompplus_qwen3_1024`

## 실행

### Windows (PowerShell)
```powershell
.\poc\run.ps1
```

### 수동 실행
```bash
cd poc
cp .env.example .env          # 필요 시 값 수정
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # (Linux: .venv/bin/python)
.venv/Scripts/python -m uvicorn backend.app:app --host 0.0.0.0 --port 8080
```

브라우저에서 http://127.0.0.1:8080 접속.

## 점검 엔드포인트

- `GET /v1/health` — 5개 백엔드 서비스 연결 상태 + 현재 설정 확인
- `GET /v1/inspect` — Qdrant 컬렉션의 payload 필드명/벡터 구성 확인

> **확인된 컬렉션 스키마** (`browsecompplus_qwen3_1024`, 1,761,285 포인트):
> - named vector **`dense`** (1024차원, Cosine, on_disk)
> - payload: **`document`**(본문) · **`source`**(문서ID) · **`chunk_id`**(청크ID, 예 `26089_26`)
> - 이 값들은 `.env`에 이미 반영되어 있습니다.

## 설계 메모 / 가정

1. **harness-1 도구 호출 방식**: 기존 하네스 프로젝트의 `OpenAIAgentInferenceModel(api_style="responses")`와
   동일하게 OpenAI **Responses API**(`/v1/responses`) + function tool 루프를 사용합니다.
   `search_corpus`, `read_document`를 노출하며 도구 이름/스키마는 원본 harness와 동일합니다.
   > ⚠️ 이 서버는 `chat.completions` 경로의 tool calling이 동작하지 않습니다(gpt-oss tool-call 파서 미설정).
   > Responses API 경로는 정상 동작하므로 그쪽을 사용합니다. harness-1은 참고문서를 못 찾으면
   > 질의를 재구성해 **반복 검색**합니다(실측: 한 질문당 3~4회 검색 후 문서 랭킹).
2. **검색 방식**: 두 컬렉션 모두 sparse 벡터를 갖고 있어 **하이브리드(dense+sparse, RRF 융합) → 리랭커**로 구성했습니다.
   - **v1**(browsecompplus): dense=qwen3, sparse=`bm25`(modifier=IDF). 쿼리 sparse는 `fastembed`(Qdrant/bm25).
   - **v2**(kb_bsss): dense=bge-m3, sparse=`sparse`(bge-m3 lexical weights). `FlagEmbedding`이 dense+sparse를 한 번에 생성.
   - Qdrant Query API의 `prefetch`(dense/sparse 각각) + `FusionQuery(RRF)`로 융합 후 Qwen3-Reranker로 재정렬.
   - 끄려면 `V1_SPARSE_ENABLED=0` / `V2_SPARSE_ENABLED=0` (dense-only로 폴백). sparse 라이브러리 미설치 시에도 자동 dense-only 폴백.
3. **출처 ID**: 검색 결과의 `DOCUMENT ID`는 Qdrant point id를 사용하며, 답변 인용에는 payload의
   `doc_id`를 표시합니다.

## 파일 구조
```
poc/
├── backend/
│   ├── config.py        # 환경설정(엔드포인트/파라미터)
│   ├── clients.py       # 임베딩/리랭커/Qdrant/OpenAI 클라이언트
│   ├── prompts.py       # harness-1 검색 프롬프트 + 오케스트레이터 합성 프롬프트
│   ├── retrieval.py     # harness-1 검색 에이전트 루프 + 검색 도구
│   ├── orchestrator.py  # exaone3.5 답변 합성(스트리밍)
│   └── app.py           # FastAPI (SSE /v1/chat, /v1/health, /v1/inspect)
├── frontend/            # 아리아 디자인 RAG 채팅 UI (index.html/styles.css/app.js)
├── requirements.txt
├── .env.example
└── run.ps1
```
