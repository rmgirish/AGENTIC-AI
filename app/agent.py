import time  # noqa: F401  (you will time each tool call)
from collections.abc import Callable

from app.memory import ConversationStore
from app.providers import AgentError  # noqa: F401
from app.tools.placement_tools import PlacementTools

MAX_STEPS = 8

SYSTEM = """You are the Placement Assistant for an engineering college's placement cell.
You are talking to the student with roll number {student_id}. Act only for this student.
Use the tools for every fact about drives, eligibility, applications and slots; never guess.
Eligibility is decided by check_eligibility, not by you. Keep replies short and concrete."""


class Agent:
    """A small agent: one student, one conversation, the placement tools."""

    def __init__(self, provider, tools: PlacementTools, student_id: str,
                 memory: ConversationStore | None = None, thread_id: str | None = None,
                 on_step: Callable[[dict], None] | None = None):
        self.provider = provider
        self.tools = tools
        self.system = SYSTEM.format(student_id=student_id)
        self.memory = memory
        self.thread_id = thread_id
        self.on_step = on_step
        self.contents: list[dict] = []     # what the model sees, turn after turn
        self.trace: list[dict] = []        # what happened, step by step
        if self.memory and self.thread_id:
            self.contents = [{"role": m["role"], "text": m["text"]}
                             for m in self.memory.load_history(self.thread_id)]

    def _log(self, entry: dict) -> None:
        """Add one entry to the trace and tell on_step about it. (Given.)"""
        self.trace.append(entry)
        if self.on_step:
            self.on_step(entry)

    # ------------------------------------------------------------------ Part 2.1

    def run_tool(self, name: str, args: dict) -> dict:
        """Call one tool with self.tools.call(name, args). Never raise.

        - NotImplementedError -> {"error": "not_implemented", "hint": ...}
        - any other exception  -> {"error": "tool_failed", "hint": ...}
        """
        try:
            return self.tools.call(name, args)
        except NotImplementedError:
            return {"error": "not_implemented",
                    "hint": f"Tool {name} is not implemented yet."}
        except Exception as e:
            return {"error": "tool_failed",
                    "hint": f"Tool {name} failed: {e}"}

    # ------------------------------------------------------------------ Part 2.2

    def ask(self, text: str) -> str:
        """One user turn: loop model calls and tool calls until the model answers."""
        run_id = None
        if self.memory and self.thread_id:
            run_id = self.memory.start_run(self.thread_id, self.provider.model)
        self.contents.append({"role": "user", "text": text})
        if run_id is not None:
            self.memory.append_message(self.thread_id, "user", text)

        try:
            step = 1
            while step <= MAX_STEPS:
                turn = self.provider.generate(
                    self.system, self.contents, list(self.tools.functions().values())
                )
                if run_id is not None:
                    self.memory.record_model_step(run_id, step, turn.tokens_in, turn.tokens_out)
                self._log({"step": step, "kind": "model",
                           "tokens_in": turn.tokens_in, "tokens_out": turn.tokens_out})

                if not turn.tool_calls:
                    reply = turn.text
                    self.contents.append({"role": "model", "text": reply, "raw": turn.raw})
                    if run_id is not None:
                        self.memory.append_message(self.thread_id, "model", reply)
                        self.memory.finish_run(run_id, "succeeded")
                    return reply

                self.contents.append({"role": "model", "text": turn.text,
                                      "raw": turn.raw,
                                      "tool_calls": [{"name": c.name, "args": c.args}
                                                     for c in turn.tool_calls]})

                tool_step = step + 1
                for call in turn.tool_calls:
                    start = time.time()
                    result = self.run_tool(call.name, call.args)
                    ms = int((time.time() - start) * 1000)
                    ok = "error" not in result
                    self._log({"step": tool_step, "kind": "tool", "tool": call.name,
                               "args": call.args, "result": result, "ok": ok, "ms": ms})
                    if run_id is not None:
                        self.memory.record_tool_call(run_id, tool_step, call.name, call.args, result,
                                                    ok, ms)
                    self.contents.append({"role": "tool", "name": call.name, "result": result})
                    tool_step += 1

                step = tool_step

            raise AgentError("step_limit", f"Exceeded {MAX_STEPS} steps without answer")
        except AgentError as exc:
            if run_id is not None:
                self.memory.finish_run(run_id, "failed", exc.code)
            raise
        except Exception as exc:
            code = getattr(exc, "code", None)
            if run_id is not None:
                self.memory.finish_run(run_id, "failed", code)
            raise

