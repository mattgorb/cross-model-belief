#!/usr/bin/env python3
"""Remove the teaser's two header lines and crop its PDF page to the panels.

Usage:
    python3 scripts/crop_teaser_pdf.py input.pdf output.pdf

The page artwork remains vector based. PyMuPDF renders a temporary bitmap only
to find the tight page bounds; it does not replace or rasterize the artwork.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz
from PIL import Image


HEADER_LINES = (
    "When two probes agree, a false claim can pass unnoticed",
    "Mass-mean probe results · false claims and silent agreements",
)


def crop_page(input_path: Path, output_path: Path) -> None:
    doc = fitz.open(input_path)
    if len(doc) != 1:
        raise ValueError(f"Expected a one-page PDF, found {len(doc)} pages")
    page = doc[0]

    for line in HEADER_LINES:
        for rect in page.search_for(line):
            # A small white redaction removes only the requested text glyphs.
            page.add_redact_annot(rect + (-0.4, -0.4, 0.4, 0.4), fill=(1, 1, 1))
    page.apply_redactions(images=0, graphics=0, text=0)

    # Detect visible ink on a temporary render, then convert its bounds back
    # to PDF points. This changes only the page boxes; the original vectors,
    # fonts, colors, and panel geometry remain intact.
    scale = 3
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    mask = image.convert("L").point(lambda value: 255 if value < 248 else 0)
    bounds = mask.getbbox()
    if bounds is None:
        raise ValueError("No visible figure content found")

    x0, y0, x1, y1 = bounds
    pad_pt = 2.0
    sx, sy = page.rect.width / pix.width, page.rect.height / pix.height
    crop_page = fitz.Rect(
        max(page.rect.x0, x0 * sx - pad_pt),
        max(page.rect.y0, y0 * sy - pad_pt),
        min(page.rect.x1, x1 * sx + pad_pt),
        min(page.rect.y1, y1 * sy + pad_pt),
    )
    # Convert from PyMuPDF's top-left page coordinates to PDF user space.
    crop_pdf = crop_page * ~page.transformation_matrix
    page.set_mediabox(crop_pdf)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path, garbage=0, deflate=False)
    doc.close()
    print(f"Saved {output_path} with page size {crop_page.width:.1f} × {crop_page.height:.1f} pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_pdf", type=Path)
    parser.add_argument("output_pdf", type=Path)
    args = parser.parse_args()
    crop_page(args.input_pdf, args.output_pdf)


if __name__ == "__main__":
    main()
