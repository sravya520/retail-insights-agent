"""LangGraph workflow: router -> sql_agent and/or rag_agent -> combiner -> verifier, or router -> refuse.

Design notes
  * The router returns a Pydantic object (route + a standalone version of the question), so follow-ups
    like "And in December?" work. The graph state is saved per thread_id with MemorySaver.
  * For route "both" the two agents run one after the other (sql_agent, then rag_agent). This keeps the
    trajectory order fixed and testable.
  * The combiner only calls the LLM for route "both". For one agent it passes that agent's answer through.
  * The verifier is plain code (see verifier.py). The refuse node returns a fixed message, no LLM.
  * Every node adds its name to state["trajectory"]. The router starts a fresh list for each request.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ria import config
from ria.llm import get_llm, invoke_structured, invoke_with_retry
from ria.prompting import build_messages
from ria.rag_agent import RagAnswer, answer_policy_question
from ria.sql_agent import SQLAgent
from ria.tracing import log_request_trace, request_scope, totals
from ria.verifier import verify_answer

Route = Literal["data", "policy", "both", "out_of_scope"]

REFUSAL_MESSAGE = ("I can only help with questions about Northwind Gifts sales data and its policy documents "
                   "(returns, warranty, shipping, wholesale terms, loyalty programme, data privacy). "
                   "I cannot help with that question.")
HISTORY_TURNS_FOR_ROUTER = 3


class RouterDecision(BaseModel):
    """What the router must return."""
    route: Route = Field(description="data, policy, both or out_of_scope")
    standalone_question: str = Field(description="The user's question rewritten so it makes sense without the chat history")
    data_question: str = Field(default="", description="Only the part answered from sales data (empty for policy and out_of_scope)")
    policy_question: str = Field(default="", description="Only the part answered from policy documents (empty for data and out_of_scope)")
    reason: str = Field(description="One short sentence explaining the route")


ROUTER_PROMPT = """You are the router of a retail question answering system. Choose one route for the user's new question.

Routes:
  data         The answer is a number or list from the sales database: revenue, orders, products, customers,
               countries, returns (cancelled orders). The data covers a UK online gift retailer, Dec 2009 to Dec 2011.
  policy       The answer is a rule in the company policy documents: returns and refunds, warranty, shipping and
               delivery, wholesale and B2B terms, loyalty programme, customer data and privacy.
  both         The question needs a figure from the sales data AND a rule from the policy documents.
  out_of_scope Anything else: weather, general knowledge, coding, opinions, other companies, chit-chat.

Words such as returns, refunds, points, shipping, discount and customers appear in both the sales data and the policy
documents. Decide by what is being asked:
  * how many, how much, which, or a total over a period  ->  data
  * what a rule says, how long, whether something is allowed, or whether a rule applies to some kind of customer  ->  policy
  * choose both only when the question asks for a figure from the data AND, separately, a rule from the policy documents.

Examples:
  "How many orders were returned last June?"                       -> data
  "Can I send back a pair of mittens I only wore once?"            -> policy
  "Do trade accounts earn points on what they buy?"                -> policy
  "What happens to points earned on something I then return?"      -> policy
  "How many customers ordered twice, and how long is the warranty?" -> both

Also write standalone_question: the new question rewritten so it can be understood without the chat history.
If the new question is a follow-up (for example "And in December?"), use the history to fill in what is missing.
If it is already complete, copy it unchanged.

Then split the work for the agents:
  data_question    the part that needs the sales data, as a complete question (empty if none)
  policy_question  the part that needs the policy documents, as a complete question (empty if none)
For route data, data_question is the standalone question. For route policy, policy_question is the standalone question.
For route both, give each agent only its own part. Do not put a policy rule in data_question or a number request in policy_question."""


class State(TypedDict, total=False):
    question: str
    route: str
    standalone_question: str
    data_question: str
    policy_question: str
    router_reason: str
    sql: dict[str, Any] | None
    rag: dict[str, Any] | None
    draft: str
    answer: str
    verifier: dict[str, Any] | None
    verified: bool | None
    trajectory: list[str]
    history: list[dict[str, str]]


@dataclass
class Services:
    """The parts the graph needs. Tests pass fakes here."""
    router_llm: Any
    sql_agent: Any                                    # has .run(question) -> SQLResult
    rag_answer: Callable[[str], RagAnswer]
    combiner_llm: Any


def default_services() -> Services:
    from ria.retrieval import Retriever
    retriever = Retriever()
    rag_llm = get_llm("rag")
    return Services(
        router_llm=get_llm("router"),
        sql_agent=SQLAgent(),
        rag_answer=lambda q: answer_policy_question(q, retriever=retriever, llm=rag_llm),
        combiner_llm=get_llm("rag"),
    )


def _history_text(history: list[dict[str, str]]) -> str:
    turns = history[-HISTORY_TURNS_FOR_ROUTER:]
    return "\n".join(f"User: {t['standalone']}\nAssistant ({t['route']}): {t['answer'][:300]}" for t in turns) or "(none)"


def _jsonable(v: Any) -> Any:
    return v if v is None or isinstance(v, (int, float, str, bool)) else str(v)


COMBINER_PROMPT = """Write one short answer to the question from the two partial answers below.
Put the data answer first, then the policy answer.
Keep every number exactly as written. Keep every citation in square brackets exactly as written.
Do not add facts, numbers or citations that are not in the partial answers.

Question: {question}

Data answer:
{sql_answer}

Policy answer:
{rag_answer}"""


def build_graph(services: Services, checkpointer: Any = None):
    def router(state: State) -> State:
        history = state.get("history", [])
        messages = build_messages(ROUTER_PROMPT, f"Chat history:\n{_history_text(history)}\n\nNew question: {state['question']}")
        decision = invoke_structured(services.router_llm, messages, RouterDecision, node="router")
        standalone = decision.standalone_question.strip() or state["question"]
        return {"route": decision.route, "standalone_question": standalone,
                "data_question": decision.data_question.strip() or standalone,
                "policy_question": decision.policy_question.strip() or standalone,
                "router_reason": decision.reason, "trajectory": ["router"], "sql": None, "rag": None,
                "draft": "", "answer": "", "verifier": None, "verified": None}

    def sql_agent(state: State) -> State:
        res = services.sql_agent.run(state["data_question"])
        sql = {"sql": res.sql, "columns": res.columns, "rows": [[_jsonable(v) for v in r] for r in res.rows],
               "answer": res.answer, "error": res.error, "repaired": res.repaired,
               "repair_log": res.repair_log, "explanation_failed": res.explanation_failed,
               "explanation_error": res.explanation_error}
        return {"sql": sql, "trajectory": state["trajectory"] + ["sql_agent"]}

    def rag_agent(state: State) -> State:
        res = services.rag_answer(state["policy_question"])
        rag = {"answer": res.answer,
               "chunks": [{"doc": h.chunk.doc, "clause_id": h.chunk.clause_id, "page": h.chunk.page,
                           "text": h.chunk.text} for h in res.hits],
               "citations": [vars(c) for c in res.citations]}
        return {"rag": rag, "trajectory": state["trajectory"] + ["rag_agent"]}

    def combiner(state: State) -> State:
        route = state["route"]
        if route == "data":
            draft = state["sql"]["answer"]
        elif route == "policy":
            draft = state["rag"]["answer"]
        else:
            prompt = COMBINER_PROMPT.format(question=state["standalone_question"],
                                            sql_answer=state["sql"]["answer"], rag_answer=state["rag"]["answer"])
            draft = invoke_with_retry(services.combiner_llm, [("human", prompt)], node="combiner").content.strip()
        return {"draft": draft, "trajectory": state["trajectory"] + ["combiner"]}

    def verifier(state: State) -> State:
        res = verify_answer(state["draft"], question=state["standalone_question"],
                            sql_rows=(state.get("sql") or {}).get("rows") or [],
                            chunks=(state.get("rag") or {}).get("chunks") or [])
        turn = {"question": state["question"], "standalone": state["standalone_question"],
                "route": state["route"], "answer": res.answer}
        return {"answer": res.answer, "verified": res.verified, "verifier": res.to_log(),
                "trajectory": state["trajectory"] + ["verifier"],
                "history": (state.get("history", []) + [turn])[-6:]}

    def refuse(state: State) -> State:
        turn = {"question": state["question"], "standalone": state["standalone_question"],
                "route": "out_of_scope", "answer": REFUSAL_MESSAGE}
        return {"answer": REFUSAL_MESSAGE, "trajectory": state["trajectory"] + ["refuse"],
                "history": (state.get("history", []) + [turn])[-6:]}

    def after_router(state: State) -> str:
        return {"data": "sql_agent", "both": "sql_agent", "policy": "rag_agent"}.get(state["route"], "refuse")

    def after_sql(state: State) -> str:
        return "rag_agent" if state["route"] == "both" else "combiner"

    graph = StateGraph(State)
    for name, fn in [("router", router), ("sql_agent", sql_agent), ("rag_agent", rag_agent),
                     ("combiner", combiner), ("verifier", verifier), ("refuse", refuse)]:
        graph.add_node(name, fn)
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", after_router, ["sql_agent", "rag_agent", "refuse"])
    graph.add_conditional_edges("sql_agent", after_sql, ["rag_agent", "combiner"])
    graph.add_edge("rag_agent", "combiner")
    graph.add_edge("combiner", "verifier")
    graph.add_edge("verifier", END)
    graph.add_edge("refuse", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())


def ask(app: Any, question: str, thread_id: str = "default", label: str | None = None) -> dict[str, Any]:
    """Run one question on a thread. Returns the final state plus latency and usage totals, and writes a trace line."""
    started = time.perf_counter()
    with request_scope() as calls:
        state = app.invoke({"question": question}, config={"configurable": {"thread_id": thread_id}})
    state = dict(state)
    state["latency_s"] = round(time.perf_counter() - started, 3)
    state["thread_id"] = thread_id
    state["usage"] = totals(calls)
    state["calls"] = [{k: c.get(k) for k in ("node", "model", "input_tokens", "output_tokens", "cached_tokens",
                                             "latency_s", "cost_usd", "cache_hit")} for c in calls]
    log_request_trace({
        "thread_id": thread_id, "question": question, "standalone_question": state.get("standalone_question"),
        "route": state.get("route"), "trajectory": state.get("trajectory"), "verified": state.get("verified"),
        "removed": (state.get("verifier") or {}).get("removed", []), "latency_s": state["latency_s"],
        "model": config.GEMINI_MODEL, "label": label, "layout": config.PROMPT_LAYOUT,
        # Error texts go to the trace log only. The answer never contains them.
        "sql_error": (state.get("sql") or {}).get("error"),
        "explanation_failed": bool((state.get("sql") or {}).get("explanation_failed")),
        "explanation_error": (state.get("sql") or {}).get("explanation_error"),
        **state["usage"],
        "calls": state["calls"]})
    return state
