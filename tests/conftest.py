"""Shared test doubles. `scripted_llm` is a fake chat model: no network, tools "bind" to itself."""

from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

Responder = Callable[[list[BaseMessage]], AIMessage]


class ScriptedChat(BaseChatModel):
    """Answers each call with `responder(messages)`; records what it was shown in `seen`."""

    responder: Responder
    seen: list[list[BaseMessage]] = Field(default_factory=lambda: list[list[BaseMessage]]())

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChat":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.responder(messages))])


@pytest.fixture
def scripted_llm() -> Callable[[Responder], ScriptedChat]:
    return lambda responder: ScriptedChat(responder=responder)
