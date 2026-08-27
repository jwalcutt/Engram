import { createElement, Fragment, useMemo, type ReactNode } from "react";
import { marked } from "marked";
import DOMPurify from "dompurify";

/**
 * Markdown rendering for LLM-synthesized answers.
 *
 * Pipeline: marked (markdown -> HTML) -> DOMPurify (strict tag/attr allowlist;
 * LLM output is untrusted even locally, so raw HTML in the answer is stripped,
 * never rendered) -> a small DOM->React walker that re-applies the same tag
 * allowlist (defense in depth) and turns `[E#]` citation tokens in text nodes
 * into clickable chips — after sanitization, so a chip can only ever come from
 * our own transform, never from crafted answer HTML.
 */

const ALLOWED_TAGS = [
  "p", "h1", "h2", "h3", "h4", "h5", "h6",
  "ul", "ol", "li",
  "strong", "em", "del",
  "code", "pre",
  "blockquote", "hr", "br", "a",
  "table", "thead", "tbody", "tr", "th", "td",
];
const ALLOWED_ATTR = ["href", "start"];

marked.setOptions({ gfm: true, breaks: false, async: false });

const CITE_SPLIT_RE = /(\[E\d+\])/g;
const CITE_RE = /^\[E(\d+)\]$/;

/** Split a text node on `[E#]` tokens, rendering each as a citation chip. */
function citeNodes(text: string, onCite: (id: string) => void): ReactNode[] {
  return text.split(CITE_SPLIT_RE).map((part, i) => {
    const m = CITE_RE.exec(part);
    if (!m) return part;
    const id = `E${m[1]}`;
    return (
      <button key={i} type="button" className="cite" onClick={() => onCite(id)} translate="no">
        {id}
      </button>
    );
  });
}

function domToReact(
  node: ChildNode,
  onCite: (id: string) => void,
  inCode: boolean,
  key: number,
): ReactNode {
  if (node.nodeType === Node.TEXT_NODE) {
    const text = node.textContent ?? "";
    // A literal [E1] inside a code span/block is code, not a citation.
    return inCode ? text : createElement(Fragment, { key }, ...citeNodes(text, onCite));
  }
  if (node.nodeType !== Node.ELEMENT_NODE) return null;
  const el = node as Element;
  const tag = el.tagName.toLowerCase();
  if (!ALLOWED_TAGS.includes(tag)) return null; // DOMPurify already stripped these; belt and braces

  const props: Record<string, unknown> = { key };
  if (tag === "a") {
    const href = el.getAttribute("href") ?? "";
    // Only real web links survive as anchors; anything else unwraps to its text.
    if (!/^https?:\/\//i.test(href)) {
      return createElement(
        Fragment,
        { key },
        ...Array.from(el.childNodes).map((c, i) => domToReact(c, onCite, inCode, i)),
      );
    }
    props.href = href;
    props.target = "_blank";
    props.rel = "noopener noreferrer";
  }
  if (tag === "ol") {
    const start = el.getAttribute("start");
    if (start) props.start = start;
  }
  if (tag === "br" || tag === "hr") return createElement(tag, props);

  const children = Array.from(el.childNodes).map((c, i) =>
    domToReact(c, onCite, inCode || tag === "code" || tag === "pre", i),
  );
  return createElement(tag, props, ...children);
}

export function AnswerMarkdown({ text, onCite }: { text: string; onCite: (id: string) => void }) {
  const nodes = useMemo(() => {
    const raw = marked.parse(text) as string;
    // Two allowlists and nothing else, on purpose. Most published DOMPurify
    // bypasses need an option that is absent here: IN_PLACE returns a detached
    // subtree that stays executable, CUSTOM_ELEMENT_HANDLING lets custom
    // elements skip afterSanitizeElements, and hooks can hand back nodes the
    // sanitizer already cleared. Adding any of them, or widening ALLOWED_TAGS
    // to a script-bearing or resource-loading tag, turns advisories that do not
    // currently apply to this call site into live vulnerabilities.
    // `markdown.test.tsx` pins this config; change it there first.
    const clean = DOMPurify.sanitize(raw, { ALLOWED_TAGS, ALLOWED_ATTR });
    const doc = new DOMParser().parseFromString(clean, "text/html");
    return Array.from(doc.body.childNodes).map((n, i) => domToReact(n, onCite, false, i));
  }, [text, onCite]);
  return <>{nodes}</>;
}
