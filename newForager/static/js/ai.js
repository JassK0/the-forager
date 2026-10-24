// The AI analyst's plumbing: streaming answers from /api/ai and turning their
// markdown into DOM. Text only ever goes in through text nodes, never innerHTML.

function node(tag, cls, children = []) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  for (const c of children) n.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return n;
}

// **bold**, *italic*, _italic_ and `code` inside a line.
function inline(text) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*\s][^*]*\*|_[^_\s][^_]*_)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const t = m[0];
    if (t.startsWith("**")) out.push(node("strong", null, [t.slice(2, -2)]));
    else if (t.startsWith("`")) out.push(node("code", null, [t.slice(1, -1)]));
    else out.push(node("em", null, [t.slice(1, -1)]));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

// Headings, bullet and numbered lists, paragraphs. Enough for the analyst's answers.
export function renderMarkdown(text) {
  const frag = document.createDocumentFragment();
  let para = null, list = null;
  const close = () => { para = null; list = null; };
  for (const raw of text.split("\n")) {
    const line = raw.trimEnd();
    let m;
    if (!line.trim()) { close(); continue; }
    if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) { close(); continue; }
    if ((m = /^\s*(#{1,6})\s+(.*)$/.exec(line))) {
      close();
      frag.append(node(m[1].length <= 2 ? "h3" : "h4", null, inline(m[2].replace(/\s*#+\s*$/, ""))));
    } else if ((m = /^\s*[-*•]\s+(.*)$/.exec(line))) {
      if (!list || list.tagName !== "UL") { para = null; list = node("ul"); frag.append(list); }
      list.append(node("li", null, inline(m[1])));
    } else if ((m = /^\s*\d+[.)]\s+(.*)$/.exec(line))) {
      if (!list || list.tagName !== "OL") { para = null; list = node("ol"); frag.append(list); }
      list.append(node("li", null, inline(m[1])));
    } else if (list && /^\s{2,}\S/.test(raw)) {
      list.lastElementChild?.append(" ", ...inline(line.trim())); // a wrapped list item
    } else {
      list = null;
      if (!para) { para = node("p"); frag.append(para); } else para.append(" ");
      para.append(...inline(line.trim()));
    }
  }
  return frag;
}

// POST a request and call onEvent for each line of the streamed reply:
// {meta}, then {text} pieces, then {done} or {error}.
export async function streamAI(body, onEvent) {
  const res = await fetch("/api/ai", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok || !res.body) {
    const data = await res.json().catch(() => null);
    const err = new Error(data?.error || `The AI request failed (${res.status})`);
    err.setup = !!data?.setup;
    throw err;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 1);
      if (line) onEvent(JSON.parse(line));
    }
  }
  if (buf.trim()) onEvent(JSON.parse(buf));
}

export async function getJSON(path) {
  const res = await fetch(path);
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new Error(data?.error || `${path} failed (${res.status})`);
  return data;
}
