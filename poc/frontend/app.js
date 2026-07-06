"use strict";

const EXAMPLES = [
  { label: "What is the capital of China?", hint: "단일 사실 · 빠른 검색" },
  { label: "여러 출처를 종합해야 하는 복합 질문을 입력해 보세요", hint: "멀티홉 · 반복 검색" },
  { label: "특정 인물·사건의 세부 정보를 찾아줘", hint: "엔티티 검색" },
  { label: "두 개념을 비교/대조해 설명해줘", hint: "비교 분석" },
];

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, html) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  return e;
};
const esc = (s) =>
  String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

// ---- 경량 마크다운 렌더러 ----
// LLM(exaone3.5) 답변의 제목/목록/굵게/코드/링크와 [n] 인용 강조를 렌더한다.
// 외부 라이브러리 없이 순수 문자열 변환으로 구현(폐쇄망 대비). 입력은 항상 esc()로
// 먼저 HTML 이스케이프하므로 XSS 안전하다. 스트리밍 중 매 토큰 전체 재렌더된다.

// 인라인 서식: `코드`, [텍스트](url), **굵게**, *기울임*, [n] 인용
function mdInline(escaped) {
  let s = escaped;
  // 1) 인라인 코드 보호(다른 변환에서 제외)
  const codes = [];
  s = s.replace(/`([^`\n]+?)`/g, (m, c) => {
    codes.push(c);
    return `${codes.length - 1}`;
  });
  // 2) 링크 [text](http…)  — esc 후 URL 의 &는 &amp; 이므로 http 로 시작만 확인
  s = s.replace(
    /\[([^\]\n]+?)\]\((https?:[^)\s]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
  );
  // 3) 굵게 **x** → 기울임 *x* (밑줄 _ 은 파일명 오인 방지를 위해 미지원)
  s = s.replace(/\*\*([^\n]+?)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/\*([^*\n]+?)\*/g, "<em>$1</em>");
  // 4) 인용 [n] — 링크로 소비되지 않은 대괄호만 남는다
  s = s.replace(/\[([^\]\n]{1,80}?)\]/g, '<span class="cite">[$1]</span>');
  // 5) 인라인 코드 복원
  s = s.replace(/(\d+)/g, (m, i) => `<code>${codes[+i]}</code>`);
  return s;
}

// 블록 단위: 제목 / 목록 / 인용문 / 코드펜스 / 수평선 / 문단
function renderMarkdown(src) {
  const lines = esc(src).split("\n");
  const out = [];
  let listType = null; // 'ul' | 'ol' | null
  let para = [];
  let i = 0;

  const flushPara = () => {
    if (para.length) {
      out.push("<p>" + mdInline(para.join(" ")) + "</p>");
      para = [];
    }
  };
  const closeList = () => {
    if (listType) {
      out.push(`</${listType}>`);
      listType = null;
    }
  };

  while (i < lines.length) {
    const line = lines[i];

    // 코드펜스 ```
    if (/^\s*```/.test(line)) {
      flushPara();
      closeList();
      const buf = [];
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) buf.push(lines[i++]);
      i++; // 닫는 펜스 소비(스트리밍 중 없으면 끝까지)
      out.push("<pre><code>" + buf.join("\n") + "</code></pre>");
      continue;
    }
    // 빈 줄 → 문단/리스트 종료
    if (/^\s*$/.test(line)) {
      flushPara();
      closeList();
      i++;
      continue;
    }
    // 제목 #~######
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) {
      flushPara();
      closeList();
      const lvl = h[1].length;
      out.push(`<h${lvl}>` + mdInline(h[2].trim()) + `</h${lvl}>`);
      i++;
      continue;
    }
    // 수평선
    if (/^\s*(---+|\*\*\*+|___+)\s*$/.test(line)) {
      flushPara();
      closeList();
      out.push("<hr>");
      i++;
      continue;
    }
    // 인용문 >  (esc 후 '>' 는 '&gt;')
    if (/^\s*&gt;\s?/.test(line)) {
      flushPara();
      closeList();
      const buf = [];
      let m;
      while (i < lines.length && (m = lines[i].match(/^\s*&gt;\s?(.*)$/))) {
        buf.push(m[1]);
        i++;
      }
      out.push("<blockquote>" + mdInline(buf.join(" ")) + "</blockquote>");
      continue;
    }
    // 순서 없는 목록  -, *, +
    const ul = line.match(/^\s*[-*+]\s+(.*)$/);
    if (ul) {
      flushPara();
      if (listType !== "ul") {
        closeList();
        out.push("<ul>");
        listType = "ul";
      }
      out.push("<li>" + mdInline(ul[1]) + "</li>");
      i++;
      continue;
    }
    // 순서 있는 목록  1. 2. …
    const ol = line.match(/^\s*\d+\.\s+(.*)$/);
    if (ol) {
      flushPara();
      if (listType !== "ol") {
        closeList();
        out.push("<ol>");
        listType = "ol";
      }
      out.push("<li>" + mdInline(ol[1]) + "</li>");
      i++;
      continue;
    }
    // 일반 문단 텍스트(연속 줄은 한 문단으로 합침)
    closeList();
    para.push(line.trim());
    i++;
  }
  flushPara();
  closeList();
  return out.join("\n");
}

const state = { recents: [], busy: false };

// 접속 경로에 따라 API 버전을 결정한다. (/v2 로 열면 /v2/chat, 그 외 /v1/chat)
const API_BASE = /\/v2\/?$/.test(location.pathname) ? "/v2" : "/v1";

// ---- layout helpers ----
function scrollDown() {
  const sc = $("#scroll");
  requestAnimationFrame(() => (sc.scrollTop = sc.scrollHeight));
  setTimeout(() => (sc.scrollTop = sc.scrollHeight), 120);
}
function setBusy(b) {
  state.busy = b;
  $("#send").disabled = b;
  const st = $("#agentStatus");
  st.classList.toggle("busy", b);
  $("#agentStatusText").textContent = b ? "검색·분석 중…" : "에이전트 대기 중";
}
function showConversation() {
  $("#empty").classList.add("hidden");
  $("#conv").classList.remove("hidden");
}

// ---- rendering ----
function addUser(text) {
  const m = el("div", "msg-user");
  m.appendChild(el("div", "bubble", esc(text)));
  $("#conv").appendChild(m);
  scrollDown();
}

function addAgentShell() {
  const wrap = el("div", "msg-agent");
  wrap.appendChild(el("div", "agent-ico", '<div class="d"></div>'));
  const body = el("div", "agent-body");

  body.appendChild(
    el("div", "gen-badge", '<span class="diamond-sm"></span><span class="t">harness-1 검색 에이전트</span>')
  );

  // 진행 카드
  const procCard = el("div", "card");
  const procPad = el("div", "card-pad");
  procPad.appendChild(el("div", "card-title", "검색 진행"));
  const thinking = el(
    "div",
    "thinking",
    '<div class="dots"><span></span><span></span><span></span></div><span class="ttext">시작하는 중…</span>'
  );
  thinking.style.marginTop = "12px";
  procPad.appendChild(thinking);
  const steps = el("div", "steps");
  steps.style.display = "none";
  procPad.appendChild(steps);
  procCard.appendChild(procPad);
  body.appendChild(procCard);

  wrap.appendChild(body);
  $("#conv").appendChild(wrap);
  scrollDown();

  return {
    body,
    thinkingText: thinking.querySelector(".ttext"),
    thinking,
    steps,
    procPad,
    answerCard: null,
    answerEl: null,
    answerText: "",
  };
}

function setStatus(ctx, text) {
  ctx.thinkingText.textContent = text;
}

function addStep(ctx, icon, title, meta) {
  ctx.steps.style.display = "";
  const step = el("div", "step");
  step.appendChild(el("div", "step-ico", icon));
  const main = el("div", "step-main");
  main.appendChild(el("div", "step-q", esc(title)));
  if (meta) main.appendChild(el("div", "step-meta", esc(meta)));
  step.appendChild(main);
  ctx.steps.appendChild(step);
  scrollDown();
}

function startAnswer(ctx) {
  ctx.thinking.remove();
  setStatus; // noop
  const card = el("div", "card");
  const head = el("div", "card-pad");
  head.style.paddingBottom = "0";
  head.appendChild(el("div", "card-title", "답변"));
  head.appendChild(el("div", "card-sub", "exaone3.5 · 검색 출처 기반 합성"));
  card.appendChild(head);
  const ans = el("div", "answer");
  card.appendChild(ans);
  ctx.body.appendChild(card);
  ctx.answerCard = card;
  ctx.answerEl = ans;
  scrollDown();
}

function appendToken(ctx, t) {
  ctx.answerText += t;
  ctx.answerEl.innerHTML = renderMarkdown(ctx.answerText);
  scrollDown();
}

function addCitations(ctx, cites) {
  if (!cites || !cites.length) return;
  const box = el("div", "cites");
  box.appendChild(el("div", "cites-label", `출처 ${cites.length}건`));
  cites.forEach((c, i) => {
    const n = c.n || i + 1;
    const item = el("div", "cite-item");
    const head = el("div", "cite-id");
    head.innerHTML = `<span class="cite-num">${n}</span> ${esc(c.label || "")}` +
      (c.is_web ? ' <span class="cite-tag">웹</span>' : ' <span class="cite-tag cor">코퍼스</span>');
    item.appendChild(head);
    if (c.url) {
      const a = el("a", "cite-url");
      a.href = c.url;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = c.url;
      item.appendChild(a);
    }
    item.appendChild(el("div", "cite-snip", esc(c.snippet || "")));
    box.appendChild(item);
  });
  ctx.answerCard.appendChild(box);
  scrollDown();
}

function showError(ctx, msg) {
  if (ctx.thinking && ctx.thinking.parentNode) ctx.thinking.remove();
  ctx.body.appendChild(el("div", "err", "⚠ " + esc(msg)));
  scrollDown();
}

// ---- event handling ----
function handleEvent(ctx, ev) {
  switch (ev.type) {
    case "status":
      setStatus(ctx, ev.message);
      break;
    case "translated":
      if (ev.english && ev.english !== ev.original)
        addStep(ctx, "🌐", `영어 검색 질의: ${ev.english}`, `원문: ${ev.original}`);
      break;
    case "step":
      if (ev.tool === "search_corpus") addStep(ctx, "🔍", `코퍼스 검색: ${ev.query}`, `턴 ${ev.turn}`);
      else if (ev.tool === "web_search") addStep(ctx, "🌐", `웹 검색: ${ev.query}`, `턴 ${ev.turn}`);
      else if (ev.tool === "read_document") addStep(ctx, "📄", `문서 열람: ${ev.doc_id}`, `턴 ${ev.turn}`);
      break;
    case "search_result": {
      const n = (ev.hits || []).length;
      const isWeb = ev.source === "web";
      const top = ev.hits && ev.hits[0] ? ` · 최상위 ${ev.hits[0].chunk_id || ev.hits[0].doc_id}` : "";
      addStep(ctx, "✓", `${isWeb ? "웹" : "코퍼스"} ${n}건 검색됨${top}`, ev.query ? `질의: ${ev.query}` : "");
      break;
    }
    case "ranked":
      addStep(ctx, "★", `관련 문서 ${ (ev.documents||[]).length }건 랭킹 완료`, "");
      setStatus(ctx, "답변을 생성하는 중…");
      break;
    case "answer_start":
      startAnswer(ctx);
      break;
    case "token":
      if (!ctx.answerEl) startAnswer(ctx);
      appendToken(ctx, ev.text);
      break;
    case "done":
      addCitations(ctx, ev.citations);
      break;
    case "error":
      showError(ctx, ev.message);
      break;
  }
}

async function ask(question) {
  const q = question.trim();
  if (!q || state.busy) return;
  showConversation();
  addUser(q);
  addRecent(q);
  const ctx = addAgentShell();
  setBusy(true);

  try {
    const resp = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
    });
    if (!resp.ok || !resp.body) throw new Error("서버 응답 오류 (" + resp.status + ")");

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const raw = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const line = raw.trim();
        if (!line.startsWith("data:")) continue;
        const json = line.slice(5).trim();
        if (!json) continue;
        try {
          handleEvent(ctx, JSON.parse(json));
        } catch (e) {
          /* ignore malformed */
        }
      }
    }
  } catch (e) {
    showError(ctx, e.message || String(e));
  } finally {
    setBusy(false);
  }
}

// ---- sidebar / recents ----
function addRecent(q) {
  state.recents = [q, ...state.recents.filter((r) => r !== q)].slice(0, 8);
  renderRecents();
}
function renderRecents() {
  const c = $("#recents");
  c.innerHTML = "";
  state.recents.forEach((r) => {
    const b = el("button", "recent-item");
    b.appendChild(el("span", "rdot"));
    b.appendChild(el("span", "rlabel", esc(r)));
    b.onclick = () => ask(r);
    c.appendChild(b);
  });
}

function renderExamples() {
  const g = $("#exampleChips");
  EXAMPLES.forEach((c) => {
    const b = el("button", "chip-card");
    b.appendChild(el("div", "chip-label", esc(c.label)));
    b.appendChild(el("div", "chip-hint", esc(c.hint)));
    b.onclick = () => ask(c.label);
    g.appendChild(b);
  });
}

// ---- init ----
function init() {
  const d = new Date();
  const days = ["일", "월", "화", "수", "목", "금", "토"];
  $("#greetDate").textContent = `${d.getMonth() + 1}월 ${d.getDate()}일 ${days[d.getDay()]}요일`;

  renderExamples();

  const input = $("#input");
  const autosize = () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 120) + "px";
  };
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const v = input.value;
      input.value = "";
      autosize();
      ask(v);
    }
  });
  $("#send").onclick = () => {
    const v = input.value;
    input.value = "";
    autosize();
    ask(v);
  };
  $("#toggleRail").onclick = () => $("#rail").classList.toggle("closed");
  $("#newSession").onclick = () => {
    $("#conv").innerHTML = "";
    $("#conv").classList.add("hidden");
    $("#empty").classList.remove("hidden");
    input.value = "";
    autosize();
  };
}

document.addEventListener("DOMContentLoaded", init);
