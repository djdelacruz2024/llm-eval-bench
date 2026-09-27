"""A deliberately small Markdown renderer for model responses shown in the
dashboard. Models answer in light Markdown (bold, lists, headings); showing
the raw `**` and `##` is hard to read, but pulling in a full Markdown
library for that is more than this needs.

The input is HTML-escaped *before* any formatting is applied, so a model
response can never inject markup into the page.
"""

from __future__ import annotations

import html
import re

from markupsafe import Markup

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET = re.compile(r"^\s*[-*+]\s+(.*)$")
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_HRULE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")


def _inline(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"__(.+?)__", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", text)
    return text


def render_markdown(text: str | None) -> Markup:
    if not text:
        return Markup("")

    out: list[str] = []
    paragraph: list[str] = []
    list_tag: str | None = None

    def flush_paragraph() -> None:
        if paragraph:
            out.append("<p>" + "<br>".join(_inline(line) for line in paragraph) + "</p>")
            paragraph.clear()

    def close_list() -> None:
        nonlocal list_tag
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = None

    for raw_line in html.escape(text, quote=False).splitlines():
        line = raw_line.rstrip()

        if not line.strip():
            flush_paragraph()
            close_list()
            continue

        if _HRULE.match(line):
            flush_paragraph()
            close_list()
            continue

        heading = _HEADING.match(line)
        if heading:
            flush_paragraph()
            close_list()
            out.append(f"<h4>{_inline(heading.group(2))}</h4>")
            continue

        bullet = _BULLET.match(line)
        numbered = None if bullet else _NUMBERED.match(line)
        if bullet or numbered:
            flush_paragraph()
            tag = "ul" if bullet else "ol"
            if list_tag != tag:
                close_list()
                out.append(f"<{tag}>")
                list_tag = tag
            out.append(f"<li>{_inline((bullet or numbered).group(1))}</li>")
            continue

        close_list()
        paragraph.append(line.strip())

    flush_paragraph()
    close_list()
    return Markup("\n".join(out))
