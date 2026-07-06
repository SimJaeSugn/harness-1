"""오케스트레이터(exaone3.5:32b) 답변 합성.

harness-1이 랭킹한 문서 발췌를 근거로 최종 한국어 답변을 스트리밍 생성한다.
"""

from __future__ import annotations

import logging
import re
from typing import Iterator, List

from .clients import Hit, make_orchestrator_client
from .config import settings
from .prompts import orchestrator_system_prompt, orchestrator_user_prompt

logger = logging.getLogger("poc.orchestrator")

_HANGUL = re.compile(r"[가-힣]")


def _has_hangul(text: str) -> bool:
    return bool(_HANGUL.search(text))


def translate_query_to_english(question: str) -> str:
    """검색 품질을 위해 질문을 간결한 영어 검색 질의로 번역한다.

    이미 영어(한글 없음)이면 번역을 생략하고 원문을 그대로 반환한다.
    """
    if not _has_hangul(question):
        return question.strip()

    client = make_orchestrator_client()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a translator that converts a user's question into a concise "
                "English search query for retrieving documents from an English corpus. "
                "Output ONLY the English query text — no quotes, no explanation, no prefix."
            ),
        },
        {"role": "user", "content": question},
    ]
    resp = client.chat.completions.create(
        model=settings.orch_model,
        messages=messages,
        temperature=0.0,
    )
    english = (resp.choices[0].message.content or "").strip()
    # 모델이 따옴표/접두어를 붙이는 경우 정리
    english = english.strip('"').strip("'").strip()
    return english or question.strip()


def synthesize_stream(question: str, chunks: List[Hit]) -> Iterator[str]:
    """exaone3.5로 최종 답변 토큰을 스트리밍한다."""
    client = make_orchestrator_client()
    context = [
        {
            "n": i,
            # 코퍼스는 file_name(+page_no) 우선, 없으면 doc_id 폴백 (Hit.source_label)
            "label": c.source_label,
            "text": c.text[:2000],
        }
        for i, c in enumerate(chunks, 1)
    ]

    messages = [
        {"role": "system", "content": orchestrator_system_prompt()},
        {"role": "user", "content": orchestrator_user_prompt(question, context)},
    ]

    stream = client.chat.completions.create(
        model=settings.orch_model,
        messages=messages,
        temperature=settings.orch_temperature,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content
