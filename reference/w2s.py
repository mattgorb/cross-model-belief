"""
w2s.py -- weak-to-strong oversight via linear transfer. One train/test split.

Question: can a WEAK model's probe, transferred through a linear map, catch a
STRONG model's errors better than the strong model's own confidence?
That's scalable oversight: weak overseer, strong overseen.

  train split -> fit CCS probe on weak model A, fit map A->B
  test split  -> transfer probe to strong model B, predict B's correctness
  WIN if transfer AUROC > B confidence baseline on the test split.

Run on: A=Qwen2.5-1.5B-Instruct (weak), B=Qwen2.5-7B-Instruct (strong).
Requires: torch, transformers, datasets, scikit-learn, numpy
"""
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

A_NAME = "Qwen/Qwen2.5-7B-Instruct"     # peer overseer
B_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # peer, overseen (same capability, diff family)
LAYER_FRAC, N_TRAIN, N_TEST, SEED = 0.6, 600, 600, 0


def data(n_train, n_test):
    from datasets import load_dataset
    items = []
    for s in ["college_physics", "college_mathematics", "professional_law",
              "college_chemistry", "college_biology"]:
        for r in load_dataset("cais/mmlu", s, split="test"):
            items.append((r["question"], r["choices"], r["answer"]))
    np.random.default_rng(SEED).shuffle(items)
    return items[:n_train], items[n_train:n_train + n_test]


def claim(q, opt): return f"Question: {q}\nAnswer: {opt}"
def pair(q, opt):
    b = claim(q, opt)
    return f"{b}\nIs this answer correct? Yes.", f"{b}\nIs this answer correct? No."


class Model:
    def __init__(self, name):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.lm = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.float16, device_map="auto",
            output_hidden_states=True).eval()
        self.layer = int(self.lm.config.num_hidden_layers * LAYER_FRAC)

    @torch.no_grad()
    def hid(self, t):
        ids = self.tok(t, return_tensors="pt", truncation=True,
                       max_length=512).to(self.lm.device)
        return self.lm(**ids).hidden_states[self.layer][0].mean(0).float().cpu().numpy()

    @torch.no_grad()
    def pick(self, q, opts):
        lps = []
        for o in opts:
            ids = self.tok(claim(q, o), return_tensors="pt").to(self.lm.device)
            lg = self.lm(**ids).logits[0, :-1]
            tg = ids["input_ids"][0, 1:]
            lps.append(torch.log_softmax(lg, -1)[range(len(tg)), tg].mean().item())
        lps = np.array(lps); p = np.exp(lps - lps.max()); p /= p.sum()
        return int(lps.argmax()), float(p.max())


class CCS(nn.Module):
    def __init__(self, d): super().__init__(); self.w = nn.Linear(d, 1)
    def fit(self, Xp, Xn, epochs=1000, tries=10):
        Xp, Xn = torch.tensor(Xp).float(), torch.tensor(Xn).float()
        best, bl = None, 1e9
        for _ in range(tries):
            self.w.reset_parameters(); opt = torch.optim.Adam(self.parameters(), 1e-3)
            for _ in range(epochs):
                opt.zero_grad()
                pp, pn = torch.sigmoid(self.w(Xp)), torch.sigmoid(self.w(Xn))
                loss = ((pp - (1 - pn))**2).mean() + torch.min(pp, pn).pow(2).mean()
                loss.backward(); opt.step()
            if loss.item() < bl: bl = loss.item(); best = {k: v.clone() for k, v in self.state_dict().items()}
        self.load_state_dict(best)
    @torch.no_grad()
    def belief(self, X): return torch.sigmoid(self.w(torch.tensor(X).float())).squeeze(-1).numpy()


def collect(m, items):
    Xp, corr, conf = [], [], []
    for q, opts, gold in items:
        pk, c = m.pick(q, opts)
        p, _ = pair(q, opts[pk]); Xp.append(m.hid(p))
        corr.append(int(pk == gold)); conf.append(c)
    return np.array(Xp), np.array(corr), np.array(conf)


def auroc(s, y): a = roc_auc_score(y, s); return max(a, 1 - a)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    tr, te = data(N_TRAIN, N_TEST)
    A, B = Model(A_NAME), Model(B_NAME)

    # TRAIN: weak probe + map, on train split
    Xp_a_tr, _, _ = collect(A, tr)
    Xp_b_tr, _, _ = collect(B, tr)
    _, Xn_a_tr, _ = None, None, None
    # need neg activations too for CCS fit on A:
    Xn_a_tr = np.array([A.hid(pair(q, opts[A.pick(q, opts)[0]])[1]) for q, opts, _ in tr])
    probe = CCS(Xp_a_tr.shape[1]); probe.fit(Xp_a_tr, Xn_a_tr)
    d = Xp_a_tr.shape[1]
    M = np.linalg.solve(Xp_a_tr.T @ Xp_a_tr + 1e-2*np.eye(d), Xp_a_tr.T @ Xp_b_tr)  # A->B

    # TEST: transfer weak probe onto strong B, test split
    Xp_b_te, corr_b_te, conf_b_te = collect(B, te)
    belief_transfer = probe.belief(Xp_b_te @ np.linalg.pinv(M))  # B-space -> A-space -> probe

    base = auroc(conf_b_te, corr_b_te)
    trans = auroc(belief_transfer, corr_b_te)
    print(f"\nStrong model B error rate (test): {1-corr_b_te.mean():.2f}")
    print(f"B confidence baseline AUROC:      {base:.3f}")
    print(f"WEAK-PROBE-TRANSFER AUROC:        {trans:.3f}")
    print(f"oversight edge:                   {trans-base:+.3f}")
    print("  > 0  -> weak overseer beats strong self-report. Scalable oversight signal.")
    print("  <=0  -> transfer adds nothing over confidence. Idea stops here.")


if __name__ == "__main__":
    main()
