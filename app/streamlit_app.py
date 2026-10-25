"""Streamlit chat UI for the Retail Insights Agent.

Every answer has a "How this was answered" panel: route, agent trajectory, SQL used, citations with the cited
passages, the verified flag, tokens, cost and latency. Live questions are limited (see the sidebar) to protect a free
quota. Pre-computed examples from real eval-v1 runs can be opened without using any quota.

Run: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import hashlib
import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ria import config, ui_backend  # noqa: E402

LIVE_EXAMPLES = [
    "What were the top 5 products by revenue in November 2011?",
    "Can a wholesale customer return a damaged item after 40 days?",
    "How many orders were returned in December 2010, and how long do retail customers have to return an item if they change their mind?",
    "What is the weather in London tomorrow?",
]
ROUTE_LABEL = {"data": "Sales data", "policy": "Policy documents", "both": "Data and policy", "out_of_scope": "Out of scope"}

st.set_page_config(page_title="Retail Insights Agent", page_icon="📊", layout="wide")


def visitor_id() -> str:
    """A stable anonymous id for the limits: a hash of the visitor's IP address when the host passes it on."""
    try:
        headers = st.context.headers
        ip = (headers.get("X-Forwarded-For") or headers.get("X-Real-Ip") or "").split(",")[0].strip()
    except Exception:  # noqa: BLE001
        ip = ""
    if ip:
        return "ip-" + hashlib.sha256(ip.encode()).hexdigest()[:16]
    if "visitor" not in st.session_state:
        st.session_state.visitor = "sess-" + uuid.uuid4().hex[:12]
    return st.session_state.visitor


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("thread_id", "t-" + uuid.uuid4().hex[:10])


def render_details(meta: dict, key: str) -> None:
    """The 'How this was answered' panel."""
    with st.expander("How this was answered", expanded=False):
        if meta.get("source") == "precomputed":
            st.info(f"**Pre-computed, not live.** {meta.get('source_note', '')} Opening it used no quota."
                    + {True: " That run's answer matched the eval gold answer.", False: " That run's answer did NOT match the eval gold answer."}.get(meta.get("matches_gold"), ""))
        c1, c2 = st.columns([1, 3])
        c1.markdown(f"**Route**  \n`{meta.get('route')}` ({ROUTE_LABEL.get(meta.get('route'), '')})")
        c2.markdown("**Agent trajectory**  \n" + " → ".join(f"`{n}`" for n in meta.get("trajectory") or []))
        sq = meta.get("standalone_question")
        if sq:
            st.caption(f"Question as understood by the router: {sq}")
        if meta.get("explanation_failed"):
            st.info("The written explanation could not be produced, so the answer lists the result rows instead.")
        if meta.get("sql"):
            st.markdown("**SQL used**" + (" (repaired once after an error)" if meta.get("sql_repaired") else ""))
            st.code(meta["sql"], language="sql", wrap_lines=True)
            if meta.get("sql_rows"):
                import pandas as pd
                st.dataframe(pd.DataFrame(meta["sql_rows"], columns=meta.get("sql_columns") or None), hide_index=True)
                total = meta.get("sql_row_count", len(meta["sql_rows"]))
                st.caption(f"Showing {len(meta['sql_rows'])} of {total} result rows.")
        if meta.get("citations"):
            st.markdown("**Citations**")
            for c in meta["citations"]:
                st.markdown(f"- **{c['doc']} §{c['clause_id']}, p.{c['page']}**" + (f"  \n  > {c['text']}" if c.get("text") else ""))
        verified = meta.get("verified")
        if verified is True:
            st.success("Verified: every number and citation in this answer was checked by code against the SQL rows and the retrieved passages. Nothing was removed.")
        elif verified is False:
            removed = meta.get("removed") or []
            st.warning(f"Not fully verified: {len(removed)} unsupported sentence(s) were removed from the answer.")
            for r in removed:
                st.caption(f"Removed ({r.get('reason')}: {r.get('detail')}): {r.get('sentence')}")
        else:
            st.caption("Verifier: not applicable (the question was refused, so there is no answer text to check).")
        t = meta.get("tokens") or {}
        st.markdown(
            "| Tokens in | Tokens out | Cached in | Cost (USD) | Latency | Model time | Model calls |\n|---|---|---|---|---|---|---|\n"
            f"| {t.get('input', 0):,} | {t.get('output', 0):,} | {t.get('cached', 0):,} | ${meta.get('cost_usd', 0):.5f} | "
            f"{meta.get('latency_s', 0):.1f} s | {meta.get('model_latency_s', 0):.1f} s | {meta.get('llm_calls', 0)} |")
        st.caption(f"Cost is what this usage would cost at paid-tier list prices. The free tier bills nothing. Model: {meta.get('model', '')}. "
                   "Latency is wall time. Model time is the time spent inside model calls.")


def render_message(i: int, msg: dict) -> None:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
            return
        badge = "Pre-computed example" if msg["meta"].get("source") == "precomputed" else "Live answer"
        st.caption(badge)
        st.markdown(msg["content"])
        render_details(msg["meta"], key=f"details-{i}")


def ask_live(question: str, backend, visitor: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    try:
        with st.spinner("Working on it. Live questions take about 10 to 25 seconds, and the first one after a restart can take longer."):
            meta = backend.ask(question, st.session_state.thread_id, visitor)
        st.session_state.messages.append({"role": "assistant", "content": meta["answer"], "meta": meta})
    except ui_backend.BackendError as exc:
        st.session_state.notice = (exc.kind, exc.message)


def open_example(example: dict) -> None:
    st.session_state.messages.append({"role": "user", "content": example["question"]})
    st.session_state.messages.append({"role": "assistant", "content": example["answer"], "meta": example})


def sidebar(backend, visitor: str) -> None:
    sb = st.sidebar
    sb.title("Retail Insights Agent")
    sb.caption("Ask about the sales data of a UK online gift retailer (2009 to 2011) or about the policies of the fictional Northwind Gifts.")
    lim = backend.limits(visitor) or {}
    sb.subheader("Live questions")
    if lim:
        sb.progress(min(max(lim.get("session_remaining", 0) / max(lim.get("session_limit", 1), 1), 0.0), 1.0),
                    text=f"You: {lim.get('session_remaining', 0)} of {lim.get('session_limit', 0)} left. Demo today: {lim.get('daily_remaining', 0)} of {lim.get('daily_limit', 0)} left.")
        if not lim.get("live_available", True):
            sb.warning(lim.get("message", "Live questions are not available right now."))
    sb.caption("Live questions use a small free quota, so they are limited. The pre-computed examples below are free.")
    for q in LIVE_EXAMPLES:
        if sb.button(q, key=f"live-{q[:20]}", use_container_width=True, disabled=bool(lim) and not lim.get("live_available", True)):
            st.session_state.pending = q
    sb.divider()
    sb.subheader("Pre-computed examples")
    sb.caption("Real answers from the eval-v1 held-out questions. They use no quota.")
    examples = backend.examples()
    if examples:
        groups = {}
        for e in examples:
            groups.setdefault(e["route"], []).append(e)
        route = sb.selectbox("Route", list(groups), format_func=lambda r: ROUTE_LABEL.get(r, r), key="example_route")
        choice = sb.selectbox("Question", groups[route], format_func=lambda e: e["question"][:90], key="example_choice")
        if sb.button("Open this example (free)", use_container_width=True, key="open-example"):
            open_example(choice)
    else:
        sb.info("No pre-computed examples found.")
    sb.divider()
    if sb.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = "t-" + uuid.uuid4().hex[:10]
    sb.caption("**Limits.** The policy documents are synthetic. The sales data covers 2009 to 2011. The answers come from a small, free language model and can be wrong. See the README.")


def main() -> None:
    init_state()
    backend = ui_backend.get_backend()
    visitor = visitor_id()
    sidebar(backend, visitor)
    st.title("Retail Insights Agent")
    st.caption("A LangGraph multi-agent system. A router sends each question to a SQL agent, a policy RAG agent, both, or a refusal. "
               "A code-only verifier checks every number and citation before an answer is shown.")
    pending = st.session_state.pop("pending", None)
    if pending:
        ask_live(pending, backend, visitor)
        st.rerun()  # refresh the remaining-questions meter in the sidebar
    for i, msg in enumerate(st.session_state.messages):
        render_message(i, msg)
    notice = st.session_state.pop("notice", None)
    if notice:
        kind, message = notice
        (st.warning if kind in ("limit", "quota") else st.error)(message)
    if not st.session_state.messages:
        st.info("Pick a live question or a pre-computed example in the sidebar, or type your own below.")
    question = st.chat_input(f"Ask a question (max {config.MAX_QUESTION_CHARS} characters)")
    if question:
        ask_live(question, backend, visitor)
        st.rerun()


main()
