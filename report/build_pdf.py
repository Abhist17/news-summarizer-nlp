"""Build report/Text_Summarization_Prototype_Report.pdf from report/REPORT.md (pure pip: reportlab).

Supports the Markdown subset used in REPORT.md: #/##/### headings, paragraphs,
**bold**, *italic*, `code`, [links](url), bullet / numbered lists, pipe tables,
images ![caption](path), fenced code blocks (a ```mermaid block is replaced by
figures/pipeline.png), and <!-- pagebreak --> markers. The title block between
<!-- title-start --> and <!-- title-end --> becomes a dedicated title page.

Usage:  python report/build_pdf.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = Path(__file__).resolve().parent
SRC = HERE / "REPORT.md"
OUT = HERE / "Text_Summarization_Prototype_Report.pdf"

# ---------------------------------------------------------------- fonts (DejaVu ships with matplotlib)
import matplotlib  # noqa: E402

_FD = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(_FD / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", str(_FD / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", str(_FD / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DV-BI", str(_FD / "DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFont(TTFont("DVM", str(_FD / "DejaVuSansMono.ttf")))
from reportlab.pdfbase.pdfmetrics import registerFontFamily  # noqa: E402

registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

INK = colors.HexColor("#1a1a19")
MUTED = colors.HexColor("#52514e")
ACCENT = colors.HexColor("#2a78d6")
RULE = colors.HexColor("#d9d8d3")
HEAD_BG = colors.HexColor("#eef3fb")

ss = getSampleStyleSheet()
S = {
    "body": ParagraphStyle("body", parent=ss["BodyText"], fontName="DV", fontSize=9.6, leading=13.6,
                           textColor=INK, spaceAfter=5.5),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=16, leading=20, textColor=INK, spaceBefore=6,
                         spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=13.2, leading=17, textColor=INK, spaceBefore=12,
                         spaceAfter=6),
    "h3": ParagraphStyle("h3", fontName="DV-B", fontSize=10.8, leading=14, textColor=INK, spaceBefore=8,
                         spaceAfter=4),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=7.9, leading=10, textColor=INK),
    "cellh": ParagraphStyle("cellh", fontName="DV-B", fontSize=7.9, leading=10, textColor=INK),
    "cap": ParagraphStyle("cap", fontName="DV-I", fontSize=8.3, leading=11, textColor=MUTED,
                          alignment=TA_CENTER, spaceBefore=3, spaceAfter=10),
    "code": ParagraphStyle("code", fontName="DVM", fontSize=7.8, leading=10, textColor=INK,
                           backColor=colors.HexColor("#f4f3ef"), borderPadding=5, spaceBefore=4,
                           spaceAfter=8),
    "bullet": ParagraphStyle("bullet", fontName="DV", fontSize=9.6, leading=13.4, textColor=INK,
                             leftIndent=14, bulletIndent=4, spaceAfter=2.5),
    "t_title": ParagraphStyle("t_title", fontName="DV-B", fontSize=24, leading=31, textColor=INK,
                              alignment=TA_CENTER),
    "t_sub": ParagraphStyle("t_sub", fontName="DV", fontSize=12.5, leading=18, textColor=MUTED,
                            alignment=TA_CENTER),
}
PAGE_W, PAGE_H = A4
MARGIN = 2.0 * cm
CONTENT_W = PAGE_W - 2 * MARGIN


def inline(text: str) -> str:
    """Markdown inline -> reportlab mini-HTML."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    codes: list[str] = []

    def _code(m):
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", _code, text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2" color="#2a78d6"><u>\1</u></link>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<i>\1</i>", text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: f'<font face="DVM" size="8.4">{codes[int(m.group(1))]}</font>', text)
    return text


def image(path: Path, caption: str | None, max_w: float = CONTENT_W, max_h: float = 11.5 * cm):
    iw, ih = ImageReader(str(path)).getSize()
    scale = min(max_w / iw, max_h / ih)
    parts = [Image(str(path), width=iw * scale, height=ih * scale)]
    if caption:
        parts.append(Paragraph(inline(caption), S["cap"]))
    return KeepTogether(parts)


def _min_cell_width(text: str, header: bool) -> float:
    """Width of the longest unbreakable word in a cell, plus the 4 pt left/right padding."""
    font, size = ("DVM", 8.4) if "`" in text else ("DV-B" if header or "**" in text else "DV", 7.9)
    words = re.sub(r"[*`]", "", text).split() or [""]
    return max(pdfmetrics.stringWidth(w, font, size) for w in words) + 9


def table(lines: list[str]):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines]
    rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    lens = [max(len(re.sub(r"[*`]", "", r[j])) for r in rows) for j in range(ncol)]
    weights = [min(max(l, 4), 48) for l in lens]
    widths = [CONTENT_W * w / sum(weights) for w in weights]
    # every column at least as wide as its longest word (no "Accurac/y" breaks); the slack of the
    # wider columns pays for it
    mins = [max(_min_cell_width(r[j], i == 0) for i, r in enumerate(rows)) for j in range(ncol)]
    widths = [max(w, m) for w, m in zip(widths, mins)]
    excess = sum(widths) - CONTENT_W
    slack = [w - m for w, m in zip(widths, mins)]
    if excess > 0 and sum(slack) > 0:
        widths = [w - excess * s / sum(slack) for w, s in zip(widths, slack)]
    data = [[Paragraph(inline(c), S["cellh"] if i == 0 else S["cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return [t, Spacer(1, 8)]


def title_page(block: list[str]) -> list:
    meta = {}
    for l in block:
        m = re.match(r"\s*[-*]?\s*\*\*(.+?):\*\*\s*(.+)", l)
        if m:
            meta[m.group(1).lower()] = m.group(2).strip()
    title = next((l[2:].strip() for l in block if l.startswith("# ")), "Report")
    out = [Spacer(1, 5.5 * cm), Paragraph(inline(title), S["t_title"]), Spacer(1, 1.0 * cm)]
    for key in ("authors", "course", "date", "code"):
        if key in meta:
            out += [Paragraph(inline(meta[key]), S["t_sub"]), Spacer(1, 0.15 * cm)]
    out.append(PageBreak())
    return out


def parse(md: str) -> list:
    lines = md.splitlines()
    flow: list = []
    i = 0
    para: list[str] = []

    def flush():
        if para:
            flow.append(Paragraph(inline(" ".join(s.strip() for s in para)), S["body"]))
            para.clear()

    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s == "<!-- title-start -->":
            j = lines.index("<!-- title-end -->", i)
            flow += title_page(lines[i + 1:j])
            i = j + 1
            continue
        if s == "<!-- pagebreak -->":
            flush()
            flow.append(PageBreak())
        elif s.startswith("<!--"):
            flush()
            while "-->" not in lines[i]:
                i += 1
        elif s.startswith("```"):
            flush()
            lang = s[3:].strip()
            j = i + 1
            while not lines[j].strip().startswith("```"):
                j += 1
            body = "\n".join(lines[i + 1:j])
            if lang == "mermaid" and (HERE / "figures" / "pipeline.png").exists():
                flow.append(image(HERE / "figures" / "pipeline.png", None, max_h=13 * cm))
            else:
                flow.append(Preformatted(body, S["code"]))
            i = j
        elif s.startswith("### "):
            flush()
            flow.append(Paragraph(inline(s[4:]), S["h3"]))
        elif s.startswith("## "):
            flush()
            flow.append(Paragraph(inline(s[3:]), S["h2"]))
        elif s.startswith("# "):
            flush()
            flow.append(Paragraph(inline(s[2:]), S["h1"]))
        elif s.startswith("|"):
            flush()
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            flow += table(lines[i:j])
            i = j - 1
        elif re.match(r"!\[.*\]\(.*\)", s):
            flush()
            cap, path = re.match(r"!\[(.*)\]\((.*)\)", s).groups()
            flow.append(image(HERE / path, cap))
        elif re.match(r"^[-*] ", s) or re.match(r"^\d+\. ", s):
            flush()
            num = re.match(r"^(\d+)\. ", s)
            text = s[num.end():] if num else s[2:]
            # continuation lines (indented)
            while i + 1 < len(lines) and lines[i + 1].startswith("  ") and lines[i + 1].strip() \
                    and not re.match(r"^\s*([-*]|\d+\.) ", lines[i + 1]):
                i += 1
                text += " " + lines[i].strip()
            indent = 14 + (12 if line.startswith("  ") else 0)
            st = ParagraphStyle("b", parent=S["bullet"], leftIndent=indent, bulletIndent=indent - 10)
            flow.append(Paragraph(inline(text), st, bulletText=f"{num.group(1)}." if num else "•"))
        elif s.startswith(">"):
            flush()
            flow.append(Paragraph(inline(s.lstrip("> ")), ParagraphStyle(
                "q", parent=S["body"], leftIndent=10, textColor=MUTED, borderColor=ACCENT)))
        elif not s:
            flush()
        else:
            para.append(s)
        i += 1
    flush()
    return flow


def _sticky(f) -> bool:
    """Headings and table captions must stay on the same page as what follows them."""
    return isinstance(f, Paragraph) and (f.style.name in ("h1", "h2", "h3") or f.text.startswith("<i>Table "))


def keep_with_next(flow: list) -> list:
    out, i = [], 0
    while i < len(flow):
        group = []
        while i < len(flow) and _sticky(flow[i]):
            group.append(flow[i])
            i += 1
        if not group:
            out.append(flow[i])
            i += 1
            continue
        if i < len(flow) and not isinstance(flow[i], PageBreak):
            group.append(flow[i])
            i += 1
        out.append(KeepTogether(group))
    return out


def on_page(canvas, doc):
    if doc.page == 1:
        return
    canvas.saveState()
    canvas.setFont("DV", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, PAGE_H - 1.2 * cm, "Neural Text Summarization: A News-Article Summarizer Prototype")
    canvas.drawRightString(PAGE_W - MARGIN, 1.1 * cm, f"{doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.line(MARGIN, PAGE_H - 1.35 * cm, PAGE_W - MARGIN, PAGE_H - 1.35 * cm)
    canvas.restoreState()


def build(src: Path = SRC, out: Path = OUT) -> Path:
    flow = keep_with_next(parse(src.read_text(encoding="utf-8")))
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=MARGIN,
                            title="Neural Text Summarization: A News-Article Summarizer Prototype",
                            author="Vedansh Shukla & Abhist Kamle")
    doc.build(flow, onFirstPage=on_page, onLaterPages=on_page)
    return out


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else (OUT if src == SRC else src.with_suffix(".pdf"))
    print("Wrote", build(src, out))
