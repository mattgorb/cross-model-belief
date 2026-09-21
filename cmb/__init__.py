"""cmb — cross-model belief probes and the false-agreement (Row 2) failure mode.

See DESIGN.md for the spec this package implements. Module map:

    config    paths, seeds, model registry, layer selection
    prompts   claim text + CCS contrast pairs (P / not-P invariant)
    data      dataset loaders -> Item(id, claim, label, ...)
    models    HF wrapper: multi-layer hidden states + option logprobs
    cache     activation cache keyed (model, dataset, item, layer, pos|neg)
    extract   run a (model, dataset) pass, cached
    probes    CCS probe, sign resolution, supervised direction for Exp 3
    align     ridge map A->B, linear CKA
    metrics   sign-resolved AUROC, 8-cell table, Row-2 rate, error correlation
    synthetic activation backend for dry runs and tests (no GPU, no downloads)
"""

__version__ = "0.1.0"
