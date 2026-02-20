from __future__ import annotations

from typing import Any, Dict, List


def normalize_conversations(example: Dict[str, Any]) -> List[Dict[str, str]]:
    conversations = example.get("conversations")
    if not isinstance(conversations, list):
        return []

    normalized: List[Dict[str, str]] = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            continue
        text = content.strip()
        if not text:
            continue
        normalized.append({"role": role, "content": text})
    return normalized


def render_chat_prompt(tokenizer, conversations: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
    return tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=bool(add_generation_prompt),
        tokenize=False,
    )


def render_prompt_from_example(tokenizer, example: Dict[str, Any], add_generation_prompt: bool = True) -> str:
    conversations = normalize_conversations(example)
    if not conversations:
        return ""
    return render_chat_prompt(
        tokenizer=tokenizer,
        conversations=conversations,
        add_generation_prompt=add_generation_prompt,
    )

