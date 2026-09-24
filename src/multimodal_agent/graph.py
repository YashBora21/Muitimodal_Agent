from __future__ import annotations

import json
import logging
from typing import Any

from .config import AGENT_MAX_TOKENS, MAX_ITERATIONS, MODEL, client
from .extraction.youtube import youtube_text, youtube_urls
from .ingest import ingest_upload
from .models import Asset
from .prompts import AGENT_SYSTEM_PROMPT
from .registry import asset_index, new_asset, referenced_youtube_ids
from .retrieval import needs_hybrid_retrieval, retrieval_scope
from .tools import TOOLS, run_tool

logger = logging.getLogger(__name__)


def log_token_usage(
    response: Any,
    messages: list[dict[str, Any]],
    logs: list[str] | None = None,
) -> None:
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
    for index, message in enumerate(messages, 1):
        content = message.get("content", "")
        content_text = json.dumps(content) if isinstance(content, list) else str(content)
        approximate_tokens = len(content_text) // 4
        total_approx += approximate_tokens
        logger.info(
            "[msg %s] role=%-10s approx=%s chars=%s",
            index,
            message.get("role"),
            approximate_tokens,
            len(content_text),
        )
    logger.info("Total approximate input tokens: %s", total_approx)


def run_agent(
    request: str,
    assets: list[Asset],
    history: list[dict[str, str]],
    current_asset_ids: list[str] | None = None,
) -> dict[str, Any]:
    asset_by_id = {asset["id"]: asset for asset in assets}
    current_asset_ids = current_asset_ids or []
    referenced_ids = referenced_youtube_ids(request, assets)
    grounding_ids = referenced_ids or current_asset_ids
    tool_assets = (
        {asset_id: asset_by_id[asset_id] for asset_id in referenced_ids}
        if referenced_ids
        else asset_by_id
    )
    logs: list[str] = []
    tool_results: dict[str, str] = {}
    read_asset_ids: set[str] = set()
    grounding_messages: list[dict[str, str]] = []
    logger.info(
        "Agent | start request_chars=%s assets=%s history=%s current_assets=%s",
        len(request),
        len(assets),
        len(history),
        len(current_asset_ids),
    )

    if referenced_ids:
        logs.append(f"Resolved YouTube order | {', '.join(referenced_ids)}")

    for asset_id in grounding_ids:
        args = {
            "asset_id": asset_id,
            "query": request.strip() or None,
            "scope": retrieval_scope(request),
        }
        result = run_tool("read_asset", args, asset_by_id)
        cache_key = json.dumps(["read_asset", args], sort_keys=True, default=str)
        tool_results[cache_key] = result
        read_asset_ids.add(asset_id)
        grounding_messages.append(
            {
                "role": "system",
                "content": f"Grounding evidence from {asset_id}:\n\n{result}",
            }
        )
        asset = asset_by_id[asset_id]
        if args["query"] and needs_hybrid_retrieval(asset.get("content", "")):
            mode = (
                "whole-document overview retrieval"
                if args["scope"] == "whole"
                else "hybrid keyword + semantic retrieval"
            )
        else:
            mode = "full asset read"
        logs.append(f"Grounding | {asset_id} preloaded before model response")
        logs.append(f"Retrieval | {asset_id} | {mode}")
        logger.info(
            "Grounding | asset=%s mode=%s result_chars=%s",
            asset_id,
            mode,
            len(result),
        )

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": AGENT_SYSTEM_PROMPT.format(
                index=asset_index(assets, grounding_ids)
            ),
        },
        *history,
        *grounding_messages,
        {"role": "user", "content": request or "[uploaded files, no message]"},
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        try:
            completion_args: dict[str, Any] = {
                "model": MODEL,
                "messages": messages,
                "max_tokens": AGENT_MAX_TOKENS,
            }
            if assets:
                completion_args.update(tools=TOOLS, tool_choice="auto")
            logger.info(
                "Agent | model request iteration=%s messages=%s tools=%s",
                iteration,
                len(messages),
                bool(assets),
            )
            response = client().chat.completions.create(**completion_args)
            log_token_usage(response, messages, logs)
        except Exception as error:
            logger.warning("Agent step failed: %s", error)
            logs.append(f"error: {error}")
            return {
                "answer": "Sorry, I ran into an error processing that.",
                "decision": "answer",
                "logs": logs,
            }

        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        if not tool_calls:
            logger.info(
                "Agent | complete iteration=%s answer_chars=%s",
                iteration,
                len(message.content or ""),
            )
            return {
                "answer": (message.content or "").strip(),
                "decision": "answer",
                "logs": logs,
            }

        logger.info(
            "Agent | tool calls iteration=%s count=%s", iteration, len(tool_calls)
        )
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [tool_call.model_dump() for tool_call in tool_calls],
            }
        )

        for tool_call in tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            cache_key = json.dumps([name, args], sort_keys=True, default=str)
            if cache_key in tool_results:
                result = tool_results[cache_key]
                logs.append(f"tool: {name}({args}) [cached]")
                logger.info(
                    "Tool | name=%s asset=%s cached=true result_chars=%s",
                    name,
                    args.get("asset_id"),
                    len(result),
                )
            elif name == "read_asset" and args.get("asset_id") in read_asset_ids:
                result = (
                    "This asset was already read for the current request. "
                    "Answer from the existing grounding evidence."
                )
                logs.append(f"tool: {name}({args}) [duplicate blocked]")
                logger.info(
                    "Tool | name=%s asset=%s duplicate_blocked=true",
                    name,
                    args.get("asset_id"),
                )
            else:
                result = run_tool(name, args, tool_assets)
                tool_results[cache_key] = result
                if name == "read_asset" and args.get("asset_id") in tool_assets:
                    read_asset_ids.add(args["asset_id"])
                logs.append(f"tool: {name}({args})")
                logger.info(
                    "Tool | name=%s asset=%s cached=false result_chars=%s",
                    name,
                    args.get("asset_id"),
                    len(result),
                )
                if name == "read_asset":
                    if args.get("page_number") is not None:
                        mode = f"exact page {args['page_number']}"
                    elif (
                        args.get("query")
                        and args.get("asset_id") in asset_by_id
                        and needs_hybrid_retrieval(
                            asset_by_id[args["asset_id"]].get("content", "")
                        )
                    ):
                        mode = (
                            "whole-document overview retrieval"
                            if retrieval_scope(
                                args["query"], str(args.get("scope") or "auto")
                            )
                            == "whole"
                            else "hybrid keyword + semantic retrieval"
                        )
                    else:
                        mode = "full asset read"
                    logs.append(
                        f"Retrieval | {args.get('asset_id', 'unknown')} | {mode}"
                    )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

    logs.append("hit MAX_ITERATIONS without a final answer")
    return {
        "answer": (
            "I wasn't able to finish reasoning about that in time. "
            "Could you narrow the request down?"
        ),
        "decision": "clarify",
        "logs": logs,
    }


def ingest_assets(
    request: str,
    uploads: list[dict[str, Any]],
    existing: list[Asset],
    persist_first: bool = False,
) -> tuple[list[Asset], list[str]]:
    added: list[Asset] = []
    current_asset_ids: list[str] = []
    logger.info(
        "Graph ingest | start uploads=%s existing_assets=%s",
        len(uploads),
        len(existing),
    )

    for upload in uploads:
        upload_assets = ingest_upload(upload, existing + added)
        added.extend(upload_assets)
        if upload_assets:
            current_asset_ids.append(upload_assets[0]["id"])

    for url in youtube_urls(request):
        if not any(asset["name"] == url for asset in existing + added):
            content = youtube_text(url) if persist_first else ""
            asset = new_asset(existing + added, "youtube", url, content)
            added.append(asset)
            current_asset_ids.append(asset["id"])

    logger.info(
        "Graph ingest | complete added_assets=%s current_assets=%s",
        len(added),
        len(current_asset_ids),
    )
    return added, current_asset_ids

