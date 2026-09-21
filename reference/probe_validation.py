"""
probe_validation.py
-------------------
Does an unsupervised belief probe read TRUTH (not just prominence/confidence),
and does it SURVIVE cross-model transfer via a linear alignment map?

Two questions, one run:
  Q1 (native):     Fit CCS on model B's own activations. Does its belief score
                   separate correct vs wrong answers -- ESPECIALLY on the
                   confident-wrong slice (high output-confidence natural errors)?
                   If it only works where output confidence already works, it's
                   measuring confidence, not truth. That slice is the whole test.
  Q2 (cross-model): Fit CCS on model A. Fit a linear map A->B on paired
                   activations. Transfer A's probe to B through the map. Does it
                   still separate on B? Compare transfer-AUROC to native-B AUROC.
                   If transfer ~= native and both beat the confidence baseline,
                   the alignment map is doing real cross-model work. If native
                   works but transfer doesn't, the map is dead weight.

Decision rule after running:
  - Native probe AUROC on confident-wrong slice <= confidence baseline  -> probe
    is not reading truth. Stop. No enclave/standard/nonprofit saves this.
  - Native beats baseline but transfer collapses -> probe is real, cross-model
    story is not. Build single-model audits, drop the alignment framing.
  - Both beat baseline and transfer ~= native -> you have the foundable thing.
    Write the concept doc.

Open-weight only (needs activations + logprobs). ~1-2 days on one GPU.

Requires: torch, transformers, datasets, scikit-learn, numpy
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# ----------------------------------------------------------------------------
# CONFIG -- edit these
# ----------------------------------------------------------------------------
MODEL_A = "Qwen/Qwen2.5-7B-Instruct"     # source model (fit probe + map here)
MODEL_B = "meta-llama/Llama-3.1-8B-Instruct"  # target model (transfer to here)
LAYER_FRAC = 0.6      # which hidden layer to probe, as fraction of depth
N_ITEMS = 800         # benchmark items; want a real natural error rate
CONF_SLICE_Q = 0.5    # "confident" = top this fraction by output confidence
SEED = 0

# Benchmark: needs (question, options, gold) with a NON-trivial error rate for a
# 7-8B model. MMLU hard subjects are fine to start; GPQA / MATH give more errors
# and are the harder, more honest test. Swap in build_dataset().
# ----------------------------------------------------------------------------


def build_dataset(n):
    """Return list of dicts: {question, options:[..], gold_idx}.
    Default: MMLU college-level subjects (swap for GPQA/MATH for a harder test)."""
    from datasets import load_dataset
    subjects = ["college_physics", "college_chemistry", "college_mathematics",
                "professional_law", "college_biology"]
    items = []
    for s in subjects:
        ds = load_dataset("cais/mmlu", s, split="test")
        for r in ds:
            items.append({"question": r["question"],
                          "options": r["choices"],
                          "gold_idx": r["answer"]})
    rng = np.random.default_rng(SEED)
    rng.shuffle(items)
    return items[:n]


def claim_text(question, option):
    """Declarative claim: 'Q? The answer is <option>.' Used both to elicit the
    model's pick (via option logprobs) and to build CCS contrast pairs."""
    return f"Question: {question}\nAnswer: {option}"


def contrast_pair(question, option):
    """CCS needs a +/- pair over the SAME claim (asserted true vs false).
    Negating the *claim* preserves the P/not-P invariant CCS relies on."""
    base = claim_text(question, option)
    pos = f"{base}\nIs this answer correct? Yes."
    neg = f"{base}\nIs this answer correct? No."
    return pos, neg


class Model:
    def __init__(self, name, layer_frac):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.lm = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.float16, device_map="auto",
            output_hidden_states=True)
        self.lm.eval()
        self.layer = int(self.lm.config.num_hidden_layers * layer_frac)

    @torch.no_grad()
    def hidden(self, text):
        """Mean-pooled hidden state at self.layer for a text (the probe input)."""
        ids = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=512).to(self.lm.device)
        out = self.lm(**ids)
        h = out.hidden_states[self.layer][0]      # [seq, d]
        return h.mean(0).float().cpu().numpy()

    @torch.no_grad()
    def pick(self, question, options):
        """Model's chosen answer + output confidence, from per-option avg
        token logprob of the claim. Confidence = softmax over options."""
        lps = []
        for opt in options:
            text = claim_text(question, opt)
            ids = self.tok(text, return_tensors="pt").to(self.lm.device)
            out = self.lm(**ids)
            logits = out.logits[0, :-1]
            tgt = ids["input_ids"][0, 1:]
            lp = torch.log_softmax(logits, -1)[range(len(tgt)), tgt].mean().item()
            lps.append(lp)
        lps = np.array(lps)
        probs = np.exp(lps - lps.max()); probs /= probs.sum()
        return int(lps.argmax()), float(probs.max())  # picked_idx, confidence


class CCS(nn.Module):
    """Contrast-Consistent Search probe (Burns et al. 2022). Unsupervised:
    loss = consistency [p+ - (1-p-)]^2 + confidence min(p+,p-)^2. No labels."""
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1)

    def forward(self, xp, xn):
        pp = torch.sigmoid(self.w(xp))
        pn = torch.sigmoid(self.w(xn))
        return pp, pn

    def fit(self, Xp, Xn, epochs=1000, lr=1e-3, ntries=10):
        Xp = torch.tensor(Xp, dtype=torch.float32)
        Xn = torch.tensor(Xn, dtype=torch.float32)
        best, best_loss = None, 1e9
        for _ in range(ntries):               # CCS is seed-sensitive; restart
            self.w.reset_parameters()
            opt = torch.optim.Adam(self.parameters(), lr=lr)
            for _ in range(epochs):
                opt.zero_grad()
                pp, pn = self(Xp, Xn)
                loss = ((pp - (1 - pn))**2).mean() + torch.min(pp, pn).pow(2).mean()
                loss.backward(); opt.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best = {k: v.clone() for k, v in self.state_dict().items()}
        self.load_state_dict(best)
        # Resolve CCS sign ambiguity later, against correctness, outside fit.

    @torch.no_grad()
    def belief(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        return torch.sigmoid(self.w(X)).squeeze(-1).numpy()


def fit_map(Xa, Xb):
    """Least-squares linear map A->B (ridge). Xa,Xb are paired activations
    (same inputs through both models). This is the alignment result in one line."""
    d = Xa.shape[1]
    A = np.linalg.solve(Xa.T @ Xa + 1e-2 * np.eye(d), Xa.T @ Xb)
    return A  # apply as Xa @ A  -> B-space


def auroc_signed(score, correct):
    """AUROC of belief-score predicting correctness, sign-resolved (CCS has an
    arbitrary sign; take the orientation that predicts better)."""
    a = roc_auc_score(correct, score)
    return max(a, 1 - a)


def run_model(m, items):
    """For each item: pick answer + confidence, correctness, and probe features
    (pos/neg contrast activations on the PICKED answer)."""
    Xp, Xn, correct, conf = [], [], [], []
    for it in items:
        pk, c = m.pick(it["question"], it["options"])
        pos, neg = contrast_pair(it["question"], it["options"][pk])
        Xp.append(m.hidden(pos)); Xn.append(m.hidden(neg))
        correct.append(int(pk == it["gold_idx"])); conf.append(c)
    return (np.array(Xp), np.array(Xn),
            np.array(correct), np.array(conf))


def report(tag, belief, correct, conf):
    """Headline metrics: probe AUROC vs correctness overall AND on the
    confident-wrong slice, next to the raw-confidence baseline."""
    base = auroc_signed(conf, correct)
    full = auroc_signed(belief, correct)
    # confident slice = top CONF_SLICE_Q by output confidence
    thr = np.quantile(conf, 1 - CONF_SLICE_Q)
    m = conf >= thr
    slice_base = auroc_signed(conf[m], correct[m]) if m.sum() > 10 else float("nan")
    slice_full = auroc_signed(belief[m], correct[m]) if m.sum() > 10 else float("nan")
    err = 1 - correct.mean()
    print(f"\n=== {tag} ===")
    print(f"  natural error rate:        {err:.2f}  (n={len(correct)})")
    print(f"  confidence baseline AUROC: {base:.3f}   (does 'unsure' predict wrong?)")
    print(f"  PROBE AUROC (all items):   {full:.3f}")
    print(f"  -- confident slice (n={m.sum()}) --")
    print(f"  confidence baseline AUROC: {slice_base:.3f}")
    print(f"  PROBE AUROC:               {slice_full:.3f}  <-- THE NUMBER THAT MATTERS")
    print(f"  probe edge over confidence on slice: {slice_full - slice_base:+.3f}")


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    items = build_dataset(N_ITEMS)

    print("Loading model B (native test)...")
    B = Model(MODEL_B, LAYER_FRAC)
    Xp_b, Xn_b, corr_b, conf_b = run_model(B, items)

    # Q1: native probe on B
    ccs_b = CCS(Xp_b.shape[1]); ccs_b.fit(Xp_b, Xn_b)
    belief_b_native = ccs_b.belief(Xp_b)
    report("Q1  NATIVE  probe fit on B, tested on B", belief_b_native, corr_b, conf_b)

    print("\nLoading model A (source for transfer)...")
    A = Model(MODEL_A, LAYER_FRAC)
    Xp_a, Xn_a, corr_a, conf_a = run_model(A, items)

    # Q2: fit probe on A, fit map A->B on paired pos-activations, transfer to B
    ccs_a = CCS(Xp_a.shape[1]); ccs_a.fit(Xp_a, Xn_a)
    report("Q2a NATIVE  probe fit on A, tested on A", ccs_a.belief(Xp_a), corr_a, conf_a)

    M = fit_map(Xp_a, Xp_b)                       # A-space -> B-space
    Xp_b_from_a = Xp_b @ np.linalg.pinv(M)        # map B's activations into A-space
    belief_b_transfer = ccs_a.belief(Xp_b_from_a) # apply A's probe
    report("Q2b TRANSFER probe fit on A, tested on B via map",
           belief_b_transfer, corr_b, conf_b)

    print("\n---------------- READ THIS ----------------")
    print("Compare Q1 (native-B) vs Q2b (transfer-to-B) on the confident slice.")
    print("  transfer ~= native, both > baseline  -> map does real work. Foundable.")
    print("  native > baseline, transfer collapses -> drop the cross-model framing.")
    print("  native <= baseline on the slice       -> probe reads confidence, not")
    print("                                            truth. The whole idea stops here.")


if __name__ == "__main__":
    main()
