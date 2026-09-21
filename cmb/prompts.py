"""Claim text and CCS contrast pairs.

CCS (Burns et al. 2022) needs a +/- pair over the SAME claim, asserted true vs
false. DESIGN.md §5: negate the *claim*, not the question, so the P / not-P
invariant the consistency loss relies on actually holds.
"""

from __future__ import annotations


def claim_text(claim: str) -> str:
    """A bare declarative claim, used for both elicitation and probing."""
    return f"Claim: {claim.strip()}"


def qa_claim(question: str, option: str) -> str:
    """Multiple-choice item rendered as a single declarative claim."""
    return f"Question: {question.strip()}\nAnswer: {option.strip()}"


def contrast_pair(claim: str) -> tuple[str, str]:
    """(positive, negative) contrast pair for a claim.

    Both halves are identical up to the Yes/No token, which is what keeps the
    difference between them a statement about the claim rather than about the
    surface form.
    """
    base = claim_text(claim)
    return (
        f"{base}\nIs this claim true? Yes.",
        f"{base}\nIs this claim true? No.",
    )
