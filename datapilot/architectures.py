"""Five architectures behind one interface: `.invoke({"messages":[...], "usage":[]}, config) -> state`.

single            one tool-using agent                                   (v1 baseline)
single_thinking   same, with model reasoning ON                          (budget-matched control, Tran & Kiela)
reflection        agent -> checker(with tools)                           (v1 self-reflection)
multi_agent       planner -> analyst <-> tools -> reporter               (v1 multi-agent)
self_consistency  k independent single-agent samples + majority vote     (compute-matched, no roles)
parallel          planner -> k parallel sub-analysts -> reporter         (breadth/parallelizable tasks)

Every LLM call appends {"node", "input", "output", "total", "missing"} to state["usage"], so cost can be
broken down per role. All LangChain/LangGraph imports are lazy; construction needs those packages.
"""
# NOTE: no `from __future__ import annotations` here on purpose. LangGraph resolves the state class
# annotations with get_type_hints(); string annotations that name function-local imports would fail.
import json, operator, os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any
from typing_extensions import TypedDict

from .config import ModelConfig, RunConfig
from .evaluators import parse_final_answer, normalize

CORE = ("You are an analytical assistant with calculator tools. Use the tools for every arithmetic step; do "
        "not do arithmetic in your head. Read the question carefully, distinguish relevant from irrelevant "
        "information, and if the information given cannot determine the answer, say so.")
CHECKER = ("You are the verification stage. Independently recompute the key quantities with the tools, compare "
           "with the draft, correct any error, and give the final answer. Do not mention this verification.")
PLANNER = ("You are the planner. Write a concise numbered plan (steps, order, which tools) for the analyst. "
           "Do NOT compute anything. You have no tools.")
ANALYST = ("You are the analyst. Follow the planner's plan when sensible, use the tools for all arithmetic, "
           "and state the supporting numbers. A separate reporter will write the final answer.")
REPORTER = ("You are the reporter. Using the task, plan, and the analyst's tool results, give the final answer. "
            "Do not introduce numbers that are not supported by tool results. Do not mention internal roles.")
SUBTASKS = ('You are the orchestrator. Split the task into independent sub-tasks that can be solved in parallel. '
            'Return ONLY a JSON list of strings (max {k}). If it cannot be split, return a list with one string.')


def make_llm(cfg: ModelConfig, *, temperature: float | None = None, thinking: bool | None = None):
    from langchain_openai import ChatOpenAI
    think = cfg.thinking if thinking is None else thinking
    return ChatOpenAI(
        model=cfg.name, base_url=cfg.base_url, api_key=os.environ[cfg.api_key_env],
        temperature=cfg.temperature if temperature is None else temperature,
        max_tokens=cfg.max_tokens, timeout=cfg.timeout_s, max_retries=cfg.max_retries,
        extra_body={"chat_template_kwargs": {"enable_thinking": bool(think)}},
    )


def usage_entry(node: str, msg) -> dict:
    u = getattr(msg, "usage_metadata", None) or {}
    missing = not u
    return {"node": node, "input": int(u.get("input_tokens", 0) or 0), "output": int(u.get("output_tokens", 0) or 0),
            "total": int(u.get("total_tokens", 0) or 0), "missing": missing}


def _prefix(system: str, use_system_role: bool) -> list:
    from langchain_core.messages import SystemMessage, HumanMessage
    return [SystemMessage(content=system)] if use_system_role else [HumanMessage(content=system)]


def _graph():
    from langgraph.graph import StateGraph
    from langgraph.graph.message import add_messages
    class S(TypedDict):
        messages: Annotated[list, add_messages]
        usage: Annotated[list, operator.add]
    return StateGraph(S)


def _agent_node(llm, system, node, use_system_role):
    def run(state):
        resp = llm.invoke(_prefix(system, use_system_role) + list(state["messages"]))
        return {"messages": [resp], "usage": [usage_entry(node, resp)]}
    return run


def build_single(llm, tools, cfg: ModelConfig, name="single"):
    from langgraph.graph import START
    from langgraph.prebuilt import ToolNode, tools_condition
    g = _graph()
    g.add_node("agent", _agent_node(llm.bind_tools(tools), CORE, name, cfg.use_system_role))
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile()


def build_reflection(llm, tools, cfg: ModelConfig):
    from langgraph.graph import START, END
    from langgraph.prebuilt import ToolNode, tools_condition
    bound = llm.bind_tools(tools)
    g = _graph()
    g.add_node("agent", _agent_node(bound, CORE, "agent", cfg.use_system_role))
    g.add_node("agent_tools", ToolNode(tools))
    g.add_node("checker", _agent_node(bound, CORE + " " + CHECKER, "checker", cfg.use_system_role))
    g.add_node("checker_tools", ToolNode(tools))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "agent_tools", END: "checker"})
    g.add_edge("agent_tools", "agent")
    g.add_conditional_edges("checker", tools_condition, {"tools": "checker_tools", END: END})
    g.add_edge("checker_tools", "checker")
    return g.compile()


def build_multi_agent(llm, tools, cfg: ModelConfig):
    from langchain_core.messages import HumanMessage
    from langgraph.graph import START, END
    from langgraph.prebuilt import ToolNode
    bound = llm.bind_tools(tools)

    def planner(state):
        resp = llm.invoke(_prefix(PLANNER, cfg.use_system_role) + list(state["messages"]))
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        # plan enters the transcript as a user-side note: keeps roles alternating for strict chat templates
        return {"messages": [HumanMessage(content=f"[Planner's plan]\n{text}")],
                "usage": [usage_entry("planner", resp)]}

    def route(state):
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else "reporter"

    g = _graph()
    g.add_node("planner", planner)
    g.add_node("analyst", _agent_node(bound, CORE + " " + ANALYST, "analyst", cfg.use_system_role))
    g.add_node("tools", ToolNode(tools))
    g.add_node("reporter", _agent_node(llm, CORE + " " + REPORTER, "reporter", cfg.use_system_role))
    g.add_edge(START, "planner")
    g.add_edge("planner", "analyst")
    g.add_conditional_edges("analyst", route, {"tools": "tools", "reporter": "reporter"})
    g.add_edge("tools", "analyst")
    g.add_edge("reporter", END)
    return g.compile()


def _final_text(state) -> str:
    c = state["messages"][-1].content
    return c if isinstance(c, str) else str(c)


def _canon(text: str):
    obj = parse_final_answer(text)
    if obj is None:
        return None
    return json.dumps({k: (round(float(v), 2) if isinstance(v, (int, float)) and not isinstance(v, bool)
                           else normalize(v)) for k, v in sorted(obj.items())}, sort_keys=True)


class SelfConsistency:
    """k independent single-agent runs (temperature > 0) + majority vote on the parsed FINAL ANSWER."""
    def __init__(self, graph, k: int):
        self.graph, self.k = graph, k

    def invoke(self, inputs, config=None):
        runs = [self.graph.invoke(inputs, config) for _ in range(self.k)]
        votes = Counter(c for c in (_canon(_final_text(r)) for r in runs) if c is not None)
        pick = 0
        if votes:
            winner = votes.most_common(1)[0][0]
            pick = next(i for i, r in enumerate(runs) if _canon(_final_text(r)) == winner)
        return {"messages": [m for r in runs for m in r["messages"]],
                "usage": [u for r in runs for u in r["usage"]],
                "final_text": _final_text(runs[pick]),
                "meta": {"votes": dict(votes), "k": self.k}}


class ParallelOrchestrator:
    """planner -> k independent sub-analysts (run concurrently) -> reporter. For parallelizable tasks.

    `ask(system, user, node) -> (text, usage_entry)` and `sub_graph.invoke` are injected, so the control
    flow can be unit-tested with fakes.
    """
    def __init__(self, ask, sub_graph, k: int):
        self.ask, self.sub_graph, self.k = ask, sub_graph, k

    def _split(self, task_text: str):
        text, u = self.ask(SUBTASKS.format(k=self.k), task_text, "orchestrator")
        try:
            parts = json.loads(text[text.index("["): text.rindex("]") + 1])
            parts = [str(p) for p in parts][: self.k] or [task_text]
        except (ValueError, json.JSONDecodeError):
            parts = [task_text]
        return parts, [u]

    def invoke(self, inputs, config=None):
        task_text = inputs["messages"][0].content
        parts, usage = self._split(task_text)
        from langchain_core.messages import HumanMessage

        def solve(sub):
            msg = HumanMessage(content=f"Full task (for context):\n{task_text}\n\nYour sub-task only:\n{sub}\n"
                                       "Solve just this sub-task with the tools and report the numbers.")
            return self.sub_graph.invoke({"messages": [msg], "usage": []}, config)

        with ThreadPoolExecutor(max_workers=len(parts)) as ex:
            subs = list(ex.map(solve, parts))
        notes = "\n\n".join(f"Sub-task {i + 1}: {p}\nResult: {_final_text(s)}" for i, (p, s) in enumerate(zip(parts, subs)))
        final, u = self.ask(CORE + " " + REPORTER, f"{task_text}\n\n[Sub-analyst results]\n{notes}", "reporter")
        usage += [x for s in subs for x in s["usage"]] + [u]
        messages = [m for s in subs for m in s["messages"]]
        return {"messages": messages, "usage": usage, "final_text": final, "meta": {"subtasks": len(parts)}}


def build_architectures(cfg: ModelConfig, run: RunConfig, tools: list, names: list[str]) -> dict[str, Any]:
    llm = make_llm(cfg)
    out: dict[str, Any] = {}
    for n in names:
        if n == "single":
            out[n] = build_single(llm, tools, cfg, "single")
        elif n == "single_thinking":
            out[n] = build_single(make_llm(cfg, thinking=True), tools, cfg, "single_thinking")
        elif n == "reflection":
            out[n] = build_reflection(llm, tools, cfg)
        elif n == "multi_agent":
            out[n] = build_multi_agent(llm, tools, cfg)
        elif n == "self_consistency":
            sc = build_single(make_llm(cfg, temperature=run.sc_temperature), tools, cfg, "sc_sample")
            out[n] = SelfConsistency(sc, run.sc_samples)
        elif n == "parallel":
            def ask(system, user, node, _llm=llm):
                from langchain_core.messages import HumanMessage
                r = _llm.invoke(_prefix(system, cfg.use_system_role) + [HumanMessage(content=user)])
                return (r.content if isinstance(r.content, str) else str(r.content)), usage_entry(node, r)
            out[n] = ParallelOrchestrator(ask, build_single(llm, tools, cfg, "sub_analyst"), k=4)
        else:
            raise ValueError(f"Unknown architecture: {n}")
    return out
