"""Planning tools: a per-session task list and user clarification."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Protocol

from ..tools import Tool, ToolRegistry


class ClarificationGate(Protocol):
    """A surface that can ask the user a question and return the answers."""

    def request(self, questions: list[dict[str, Any]]) -> dict[str, Any]: ...


class UnavailableClarificationGate:
    """Fail-soft gate for library/headless surfaces without a chat channel."""

    def request(self, questions: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "outcome": "undelivered",
            "responses": [
                {
                    "question": str(item.get("question") or ""),
                    "status": "unanswered",
                    "user_response": None,
                }
                for item in questions
            ],
            "notice": "clarification is not available on this surface",
        }


class ConsoleClarificationGate:
    """Simple terminal gate for the local CLI."""

    def request(self, questions: list[dict[str, Any]]) -> dict[str, Any]:
        responses = []
        for item in questions:
            question = str(item.get("question") or "").strip()
            choices = [str(choice) for choice in (item.get("choices") or [])][:4]
            print(f"\n{question}")
            for index, choice in enumerate(choices, 1):
                print(f"  {index}. {choice}")
            try:
                raw = input("> ").strip()
            except EOFError:
                raw = ""
            if not raw:
                responses.append(
                    {
                        "question": question,
                        "choices_offered": choices,
                        "status": "skipped",
                        "user_response": None,
                    }
                )
                continue
            if choices and raw.isdigit():
                selected = int(raw)
                value: Any = (
                    choices[selected - 1] if 1 <= selected <= len(choices) else raw
                )
            elif choices and item.get("multi_select"):
                values = []
                for part in raw.replace(",", " ").split():
                    if part.isdigit() and 1 <= int(part) <= len(choices):
                        values.append(choices[int(part) - 1])
                value = values or raw
            else:
                value = raw
            responses.append(
                {
                    "question": question,
                    "choices_offered": choices,
                    "status": "answered",
                    "user_response": value,
                }
            )
        return {"outcome": "submitted", "responses": responses}


@dataclass
class TodoStore:
    items: list[dict[str, Any]] = field(default_factory=list)
    _next_id: int = 1

    def add(self, text: str) -> dict[str, Any]:
        item = {"id": self._next_id, "text": text, "done": False}
        self._next_id += 1
        self.items.append(item)
        return item

    def find(self, item_id: int) -> dict[str, Any] | None:
        return next((item for item in self.items if item["id"] == item_id), None)

    def render(self) -> str:
        if not self.items:
            return "No tasks."
        return "\n".join(
            "[{}] {} {}".format(
                "x" if item["done"] else " ",
                item["id"],
                item["text"],
            )
            for item in self.items
        )


def _normalize_questions(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise TypeError("questions must be an array of 1-5 objects")
    questions = []
    for item in raw[:5]:
        if not isinstance(item, dict):
            raise TypeError("each question must be an object")
        text = str(item.get("question") or "").strip()
        if not text:
            raise ValueError("each question needs a non-empty question field")
        choices = item.get("choices") or []
        if not isinstance(choices, list):
            choices = []
        questions.append(
            {
                "question": text,
                "choices": [str(choice) for choice in choices][:4],
                "multi_select": bool(item.get("multi_select")),
            }
        )
    if not questions:
        raise ValueError("questions must contain at least one question")
    return questions


def register_planning_tools(
    registry: ToolRegistry,
    config: Any,
    *,
    clarify_gate: ClarificationGate | None = None,
) -> list[Tool]:
    todos = TodoStore()
    gate = clarify_gate or UnavailableClarificationGate()
    added: list[Tool] = []

    def todo_list(
        action: Annotated[
            Literal["list", "add", "complete", "remove", "clear"],
            "Task-list action",
        ] = "list",
        items: Annotated[list[str] | None, "Task text(s) to add"] = None,
        ids: Annotated[list[int] | None, "Task ids to complete or remove"] = None,
    ) -> str:
        item_values = list(items or [])
        id_values = list(ids or [])
        mode = str(action or "list").strip().lower()
        if mode == "list":
            return todos.render()
        if mode == "add":
            added_items = []
            for text in item_values:
                value = str(text or "").strip()
                if value:
                    added_items.append(todos.add(value))
            if not added_items:
                return "error: add requires at least one non-empty item"
            return "added:\n" + "\n".join(
                "[{}] {}".format(item["id"], item["text"]) for item in added_items
            )
        if mode == "complete":
            changed = []
            for item_id in id_values:
                item = todos.find(int(item_id))
                if item is not None:
                    item["done"] = True
                    changed.append(item)
            return f"completed: {len(changed)}\n{todos.render()}"
        if mode == "remove":
            wanted = {int(item_id) for item_id in id_values}
            todos.items[:] = [item for item in todos.items if item["id"] not in wanted]
            return todos.render()
        if mode == "clear":
            todos.items.clear()
            todos._next_id = 1
            return "No tasks."
        return f"error: unknown action: {action}"

    def clarify(
        questions: Annotated[
            list[dict],
            "1-5 questions, each {question, choices?, multi_select?}",
        ],
    ) -> str:
        try:
            normalized = _normalize_questions(questions)
        except ValueError as exc:
            return f"error: {exc}"
        try:
            result = gate.request(normalized)
        except Exception as exc:  # noqa: BLE001 - clarify must fail soft
            return json.dumps(
                {
                    "outcome": "undelivered",
                    "notice": f"clarification gate failed: {exc}",
                },
                ensure_ascii=False,
            )
        return json.dumps(result, ensure_ascii=False, sort_keys=True)

    definitions = [
        Tool(
            todo_list,
            "todo_list",
            "Manage the session task list. Call with no arguments to read it.",
            aliases=("todo", "task list", "plan tasks", "checklist"),
        ),
        Tool(
            clarify,
            "clarify",
            "Ask the user 1-5 questions when a decision or missing detail blocks progress.",
            aliases=("ask user", "question", "clarification"),
        ),
    ]
    for definition in definitions:
        if not registry.has(definition.name):
            registry.add(definition)
            added.append(definition)
    return added
