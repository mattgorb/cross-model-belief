#!/usr/bin/env python3
"""Extend teaser panel C's bars left while keeping the 0--100% scale."""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz


ROWS = (
    ("87.6%", 87.6, 20.9, 58.82, 62.10),
    ("65.8%", 65.8, 30.7, 96.18, 99.47),
    ("53.4%", 53.4, 40.4, 133.55, 136.84),
    ("46.0%", 46.0, 28.1, 170.92, 174.20),
    ("24.2%", 24.2, 17.4, 208.28, 211.57),
)


def extend_bars(source: Path, output: Path) -> None:
    doc = fitz.open(source)
    if len(doc) != 1:
        raise ValueError("Expected a one-page teaser PDF")
    page = doc[0]

    # Reuse the figure's embedded medium-weight font for the bar values.
    value_font = next(font[0] for font in page.get_fonts(full=True) if font[4] == "TT9")
    font_bytes = doc.extract_font(value_font)[3]

    # Remove only the old plot artwork and value text. Dataset names and the
    # panel heading sit outside this rectangle and remain untouched.
    plot_area = fitz.Rect(640, 48, page.rect.width, 239.5)
    page.add_redact_annot(plot_area, fill=(1, 1, 1))
    page.apply_redactions(images=0, graphics=1, text=0)
    page.insert_font(fontname="teaser_values", fontbuffer=font_bytes)

    x_left, x_right = 651.5, 958.11
    width = x_right - x_left
    page.draw_line((x_left, 49.23), (x_left, 238.94),
                   color=(0.811765, 0.831373, 0.854902), width=0.958)

    muted = (0.874510, 0.890196, 0.909804)
    blue = (0.203922, 0.470588, 0.784314)
    red = (0.788235, 0.207843, 0.207843)
    ink = (0.125490, 0.129412, 0.141176)
    value_font_size = 12.455625

    for label, precision, random_rate, y_top, text_top in ROWS:
        y_bottom = y_top + 21.08
        page.draw_rect(
            fitz.Rect(x_left, y_top, x_right, y_bottom),
            color=None, fill=muted, radius=0.1, overlay=True,
        )
        end = x_left + width * precision / 100
        page.draw_rect(
            fitz.Rect(x_left, y_top, end, y_bottom),
            color=None, fill=blue, radius=0.1, overlay=True,
        )

        tick_x = x_left + width * random_rate / 100
        page.draw_line((tick_x, y_top - 2.87), (tick_x, y_bottom + 2.87),
                       color=red, width=1.916)

        # Match the old value-label spacing after each resized bar.
        text_x = end + 8.62
        baseline_y = text_top + 12.04459
        page.insert_text(
            (text_x, baseline_y), label,
            fontname="teaser_values", fontsize=value_font_size,
            color=ink, overlay=True,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output, garbage=0, deflate=False)
    doc.close()
    print(f"Saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_pdf", type=Path)
    parser.add_argument("output_pdf", type=Path)
    args = parser.parse_args()
    extend_bars(args.input_pdf, args.output_pdf)


if __name__ == "__main__":
    main()
