from __future__ import annotations

from model.utils.chat_preprocess import render_chat_prompt, render_prompt_from_example


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, conversations, add_generation_prompt, tokenize):
        self.calls.append(
            {
                "conversations": conversations,
                "add_generation_prompt": add_generation_prompt,
                "tokenize": tokenize,
            }
        )
        return "rendered-text"


def test_render_chat_prompt_uses_baseline_signature() -> None:
    tokenizer = FakeTokenizer()
    conversations = [{"role": "user", "content": "hello"}]
    output = render_chat_prompt(tokenizer, conversations, add_generation_prompt=True)
    assert output == "rendered-text"
    assert tokenizer.calls[0]["add_generation_prompt"] is True
    assert tokenizer.calls[0]["tokenize"] is False


def test_render_prompt_from_example_reads_conversations() -> None:
    tokenizer = FakeTokenizer()
    example = {"conversations": [{"role": "user", "content": "q"}]}
    output = render_prompt_from_example(tokenizer, example, add_generation_prompt=True)
    assert output == "rendered-text"
    assert len(tokenizer.calls) == 1

