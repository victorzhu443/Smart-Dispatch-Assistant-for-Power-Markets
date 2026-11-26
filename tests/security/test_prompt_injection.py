"""Prompt-injection robustness for Smart Dispatch's RAG `/query` path.

Deterministic and cost-free: no model is called. The live `/query` RAG service
was removed from the current tree (`tests/test_frontend_smoke.py` asserts the
`/chat` route is gone; `backend/deploy_k8s.*` still reference a `query-service`
and `gpt2_dispatch_model`). There is therefore NO model sink to wrap in the code
today — so, rather than fabricate one, this models the *documented* sink: the
RAG step that concatenated a retrieved market document and the user question into
a prompt for the explanation model.

Against that stand-in we prove two things, the measure->mitigate->verify loop
from security-agents/promptinject bound to this app's untrusted channel:

1. Undefended, the documented OWASP LLM01 corpus subverts the sink (risk is real).
2. Routed through `smart_dispatch_security.harden_untrusted` / `scrub_output`,
   the corpus is resisted and the benign control never false-positives.

When the `/query` path is reinstated, point `_sink` at the real retrieval+prompt
builder and this suite guards the live boundary unchanged.
"""
from promptinject.corpus import CANARY, corpus
from promptinject.harness import evaluate

from smart_dispatch_security import harden_untrusted, scrub_output


def _sink(document_text: str) -> str:
    """Stand-in for the RAG `/query` prompt sink. A vulnerable app effectively
    lets instructions embedded in a retrieved document through; we emulate that
    by surfacing the text, so a canary in the document reaches 'output'."""
    return (
        "Answer the operator's question about the ERCOT price forecast using "
        f"this retrieved market document:\n{document_text}"
    )


def _defended(document_text: str) -> str:
    return scrub_output(_sink(harden_untrusted(document_text)))


def test_corpus_lands_on_undefended_sink():
    # Sanity: the threat is real; the injection cases subvert the naive sink.
    report = evaluate(_sink, corpus())
    assert report.landed, "corpus should expose the undefended /query sink"


def test_hardened_query_path_resists_corpus():
    report = evaluate(_defended, corpus())
    assert report.ok, "\n" + report.render()


def test_control_document_is_not_flagged():
    # The benign control must pass through hardening without tripping the canary.
    assert CANARY not in _defended(
        "ERCOT day-ahead prices for HB_HOUSTON averaged $42/MWh last week."
    )
