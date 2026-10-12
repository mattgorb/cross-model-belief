# Teaser 1

`teaser1.pdf` is the cropped, vector-based paper figure. `teaser1.html` contains the panel markup and exact labels.

To reproduce the current PDF bar layout from its preserved vector source:

```bash
python3 extend_bars.py teaser1-base.pdf teaser1.pdf
```

`extend_bars.py` keeps the panel text, colors, 0--100% scale, and embedded font, and redraws only Panel C's bars and random-baseline ticks with a shorter gap after the dataset names. Edit the `ROWS` values in the script and the matching SVG values in `teaser1.html` when updating the data.
