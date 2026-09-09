from __future__ import annotations

import json
import logging
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .config import MAX_ITERATIONS, MODEL, TOOL_RESULT_LIMIT, client
from .extraction.youtube import youtube_urls
from .ingest import ingest_upload
from .models import Asset, State
from .prompts import AGENT_SYSTEM_PROMPT
from .registry import asset_index, compact, new_asset
from .tools import TOOLS, run_tool

logger = logging.getLogger(__name__)


def log_token_usage(response: Any, messages: list[dict[str, Any]], logs: list[str] | None = None) -> None:
    usage = getattr(response, "usage", None)
    if usage:
        token_log = (
            f"Tokens | input={getattr(usage, 'prompt_tokens', '?')} "
            f"output={getattr(usage, 'completion_tokens', '?')} "
            f"total={getattr(usage, 'total_tokens', '?')}"
        )
        logger.info(token_log)
        if logs is not None:
            logs.append(token_log)

    total_approx = 0
    for i, msg in enumerate(messages, 1):
        content = msg.get("content", "")
        content_text = json.dumps(content) if isinstance(content, list) else str(content)
        approx = len(content_text) // 4
        total_approx += approx
        logger.info("[msg %s] role=%-10s approx=%s chars=%s", i, msg.get("role"), approx, len(content_text))
    logger.info("Total approximate input tokens: %s", total_approx)


def run_agent(
    request: str,
    assets: list[Asset],
    history: list[dict[str, str]],
    current_asset_ids: list[str] | None = None,
) -> dict[str, Any]:
    asset_by_id = {a["id"]: a for a in assets}
    current_asset_ids = current_asset_ids or []
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT.format(index=asset_index(assets, current_asset_ids))},
        *history,
        {"role": "user", "content": request or "[uploaded files, no message]"},
    ]
    logs: list[str] = []
    tool_results: dict[str, str] = {}  # dedupe identical tool calls within this request

    for _ in range(MAX_ITERATIONS):
        try:
            completion_args: dict[str, Any] = {"model": MODEL, "messages": messages, "max_tokens": 2000}
            if assets:  # no tool-schema token cost for plain conversational turns
                completion_args.update(tools=TOOLS, tool_choice="auto")
            response = client().chat.completions.create(**completion_args)
            log_token_usage(response, messages, logs)
        except Exception as e:
            logger.warning("Agent step failed: %s", e)
            logs.append(f"error: {e}")
            return {"answer": "Sorry, I ran into an error processing that.", "decision": "answer", "logs": logs}

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        if not tool_calls:
            return {"answer": (message.content or "").strip(), "decision": "answer", "logs": logs}

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [tc.model_dump() for tc in tool_calls],
        })

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            cache_key = json.dumps([name, args], sort_keys=True, default=str)
            if cache_key in tool_results:
                result = tool_results[cache_key]
                logs.append(f"tool: {name}({args}) [cached]")
            else:
                result = run_tool(name, args, asset_by_id)
                tool_results[cache_key] = result
                logs.append(f"tool: {name}({args})")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": compact(result, TOOL_RESULT_LIMIT),
            })

    logs.append("hit MAX_ITERATIONS without a final answer")
    return {
        "answer": "I wasn't able to finish reasoning about that in time. Could you narrow the request down?",
        "decision": "clarify",
        "logs": logs,
    }


def build_graph():

    def ingest(state: State) -> dict[str, Any]:
        existing = state.get("assets", [])
        added: list[Asset] = []

        for upload in state.get("uploads", []):
            added.extend(ingest_upload(upload, existing + added))

        for url in youtube_urls(state.get("request", "")):
            if not any(a["name"] == url for a in existing + added):
                added.append(new_asset(existing + added, "youtube", url, ""))

        return {
            "assets": added,
            "current_asset_ids": [a["id"] for a in added],
            "uploads": [],
            "extracted": compact("\n\n".join(a.get("content", "") for a in added), 15_000),
            "logs": ["assets extracted"],
        }

    def agent_node(state: State) -> dict[str, Any]:
        request = state.get("request", "").strip()
        assets = state.get("assets", [])
        history = [
            {"role": m["role"], "content": m["content"][-300:]}
            for m in state.get("messages", [])[-6:]
        ]

        result = run_agent(request, assets, history, state.get("current_asset_ids", []))
        answer = result["answer"]

        return {
            "answer": answer,
            "decision": result["decision"],
            "messages": [
                {"role": "user", "content": request or "[uploaded files]"},
                {"role": "assistant", "content": answer},
            ],
            "logs": state.get("logs", []) + result["logs"],
        }

    builder = StateGraph(State)
    builder.add_node("ingest", ingest)
    builder.add_node("agent", agent_node)
    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "agent")
    builder.add_edge("agent", END)

    return builder.compile(checkpointer=InMemorySaver())


graph = build_graph()
