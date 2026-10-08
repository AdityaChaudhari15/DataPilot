"""A strict miniature stand-in for LangChain / LangGraph / langchain-openai, used ONLY when the real packages
are not installed. It reproduces the behaviours our code depends on, including the strictness that matters:
LangGraph resolves state annotations with typing.get_type_hints (so string annotations naming local imports fail),
ToolNode executes the tool calls on the last message, and recursion_limit raises GraphRecursionError."""
import json, sys, types, typing, operator


def install(answer_for_prompt):
    """answer_for_prompt(text) -> dict for the FINAL ANSWER line (looked up by task prompt substring)."""
    # ----------------------------------------------------------------- messages
    class Msg:
        def __init__(self, content="", **kw):
            self.content, self.tool_calls = content, kw.get("tool_calls", [])
            self.usage_metadata = kw.get("usage_metadata")
            self.tool_call_id = kw.get("tool_call_id")
    class SystemMessage(Msg): pass
    class HumanMessage(Msg): pass
    class AIMessage(Msg): pass
    class ToolMessage(Msg): pass

    m_core = types.ModuleType("langchain_core"); m_msgs = types.ModuleType("langchain_core.messages")
    for c in (SystemMessage, HumanMessage, AIMessage, ToolMessage):
        setattr(m_msgs, c.__name__, c)

    # ----------------------------------------------------------------- tools
    class StructuredTool:
        def __init__(self, f, name, description): self.f, self.name, self.description = f, name, description
        @classmethod
        def from_function(cls, f, name=None, description=None): return cls(f, name, description)
        def invoke(self, args): return self.f(**args)
    m_tools = types.ModuleType("langchain_core.tools"); m_tools.StructuredTool = StructuredTool

    # ----------------------------------------------------------------- graph
    START, END = "__start__", "__end__"
    class GraphRecursionError(Exception): pass
    def add_messages(left, right): return list(left) + list(right)

    class StateGraph:
        def __init__(self, state_cls):
            self.hints = typing.get_type_hints(state_cls, include_extras=True)     # strict, like LangGraph
            self.reducers = {k: h.__metadata__[0] for k, h in self.hints.items() if hasattr(h, "__metadata__")}
            self.nodes, self.edges, self.cond = {}, {}, {}
        def add_node(self, name, fn): self.nodes[name] = fn
        def add_edge(self, a, b): self.edges[a] = b
        def add_conditional_edges(self, a, fn, mapping=None): self.cond[a] = (fn, mapping)
        def compile(self): return Compiled(self)

    class Compiled:
        def __init__(self, g): self.g = g
        def invoke(self, state, config=None):
            limit = (config or {}).get("recursion_limit", 25)
            state = {k: list(v) if isinstance(v, list) else v for k, v in state.items()}
            cur, steps = self.g.edges[START], 0
            while cur != END:
                steps += 1
                if steps > limit: raise GraphRecursionError("recursion limit")
                upd = self.g.nodes[cur](state)
                for k, v in upd.items():
                    state[k] = self.g.reducers[k](state.get(k, []), v)
                if cur in self.g.cond:
                    fn, mapping = self.g.cond[cur]
                    key = fn(state)
                    cur = mapping[key] if mapping else key
                else:
                    cur = self.g.edges.get(cur, END)
            return state

    class ToolNode:
        def __init__(self, tools): self.tools = {t.name: t for t in tools}
        def __call__(self, state):
            out = []
            for c in state["messages"][-1].tool_calls:
                out.append(ToolMessage(content=str(self.tools[c["name"]].invoke(c["args"])), tool_call_id=c["id"]))
            return {"messages": out}
    def tools_condition(state): return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    m_lg = types.ModuleType("langgraph"); m_graph = types.ModuleType("langgraph.graph")
    m_graph.StateGraph, m_graph.START, m_graph.END = StateGraph, START, END
    m_gm = types.ModuleType("langgraph.graph.message"); m_gm.add_messages = add_messages
    m_pre = types.ModuleType("langgraph.prebuilt"); m_pre.ToolNode, m_pre.tools_condition = ToolNode, tools_condition
    GraphRecursionError.__name__ = "GraphRecursionError"

    # ----------------------------------------------------------------- scripted model
    created = []
    class ChatOpenAI:
        def __init__(self, **kw):
            self.kw, self.tools = kw, False
            created.append(kw)
        def bind_tools(self, tools):
            b = ChatOpenAI(**self.kw); b.tools = True; created.pop(); return b
        def invoke(self, messages):
            text = " ".join(str(m.content) for m in messages)
            sys_text = str(messages[0].content)
            n_in = 50 * len(messages)
            def ai(content="", **kw):
                return AIMessage(content, usage_metadata={"input_tokens": n_in, "output_tokens": 30, "total_tokens": n_in + 30}, **kw)
            if "You are the planner" in sys_text: return ai("1. add the numbers 2. answer")
            if "You are the orchestrator" in sys_text: return ai('["part one", "part two"]')
            if self.tools and not any(isinstance(m, ToolMessage) for m in messages):
                return ai("", tool_calls=[{"name": "add", "args": {"a": 1, "b": 2}, "id": "c1"}])
            return ai("Work done.\nFINAL ANSWER: " + json.dumps(answer_for_prompt(text)))
    m_oa = types.ModuleType("langchain_openai"); m_oa.ChatOpenAI = ChatOpenAI

    for name, mod in {"langchain_core": m_core, "langchain_core.messages": m_msgs, "langchain_core.tools": m_tools,
                      "langgraph": m_lg, "langgraph.graph": m_graph, "langgraph.graph.message": m_gm,
                      "langgraph.prebuilt": m_pre, "langchain_openai": m_oa}.items():
        sys.modules[name] = mod
    return types.SimpleNamespace(created_llms=created, GraphRecursionError=GraphRecursionError)
