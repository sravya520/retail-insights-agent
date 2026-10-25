"""Render the Markdown policy documents in data/docs to PDF with reportlab.

Run: python -m ria.make_policy_pdfs
"""
from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from ria.config import DOCS_DIR

FOOTER_TEMPLATE = "Northwind Gifts (synthetic document) | {title} | Page {page}"


def inline_markup(text: str) -> str:
    """Escape XML and turn **bold** into <b>bold</b>."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escape(text))


def render_pdf(md_path: Path, pdf_path: Path) -> None:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    title = lines[0].removeprefix("# ").strip()
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10.5, leading=15, spaceAfter=7)
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=20, spaceAfter=10)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=6)

    story = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline_markup(line[2:]), h1))
        elif line.startswith("## "):
            story.append(Paragraph(inline_markup(line[3:]), h2))
        else:
            story.append(Paragraph(inline_markup(line), body))
    story.append(Spacer(1, 4 * mm))

    def footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(20 * mm, 10 * mm, FOOTER_TEMPLATE.format(title=title, page=doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4, title=title, author="Northwind Gifts (synthetic)",
        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=20 * mm,
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    for md_path in sorted(DOCS_DIR.glob("*.md")):
        pdf_path = md_path.with_suffix(".pdf")
        render_pdf(md_path, pdf_path)
        print(f"wrote {pdf_path.name}")


if __name__ == "__main__":
    main()
