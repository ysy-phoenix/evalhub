import asyncio
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType

from omegaconf import OmegaConf
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from evalhub.callback.base_callback import BaseCallback
from evalhub.inference.generator import LLMGenerator
from evalhub.inference.schemas import GenerationConfig
from evalhub.tools.base_tool import BaseTool


def get_module(module_name: str) -> ModuleType:
    if module_name not in sys.modules:
        spec = importlib.util.find_spec(module_name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        module = sys.modules[module_name]
    return module


class MultiTurnGenerator(LLMGenerator):
    def __init__(self, config: GenerationConfig, system_prompt: str | None = None) -> None:
        super().__init__(config, system_prompt)
        self._initialize_tools(config.tool_config_path)
        self._initialize_callback(config.callback)

    def _initialize_tools(self, tool_config_path: Path | None = None) -> None:
        self.available_tools = defaultdict(BaseTool)
        self.tool_schemas: list[dict] = []

        if tool_config_path is None:
            return

        tools_config = OmegaConf.load(tool_config_path)
        for tool_config in tools_config.tools:
            cls_name = tool_config.class_name
            module_name, class_name = cls_name.rsplit(".", 1)

            tool_schema_dict = OmegaConf.to_container(tool_config.tool_schema, resolve=True)
            self.tool_schemas.append(tool_schema_dict)

            module = get_module(module_name)
            tool_cls: BaseTool = getattr(module, class_name)
            tool = tool_cls(
                name=tool_config.tool_schema.function.name,
                config=OmegaConf.to_container(tool_config.config, resolve=True),
            )
            self.available_tools[tool.name] = tool

    def _initialize_callback(self, cls_name: str | None = None) -> None:
        self.callback = None
        if cls_name is None:
            return

        module_name, class_name = cls_name.rsplit(".", 1)
        module = get_module(module_name)
        callback_cls = getattr(module, class_name)
        self.callback: BaseCallback = callback_cls()

    async def get_response_with_retry(
        self, messages: list[dict], max_retries: int = 3
    ) -> ChatCompletion | None:
        for _ in range(max_retries):
            response = await self.complete(messages, tools=self.tool_schemas)
            if response is not None:
                return response
        return None

    async def _handle_tool_call(
        self, messages: list, message: ChatCompletionMessage, instance_id: str
    ) -> None:
        tool_call_routines = []
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments
            while isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = arguments
            tool_call_routines.append(
                self.available_tools[tool_name].execute(instance_id, arguments)
            )
        results = await asyncio.gather(*tool_call_routines)
        for tool_call, result in zip(message.tool_calls, results, strict=False):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                    "name": tool_call.function.name,
                }
            )

    async def _handle_callback(
        self, messages: list, message: ChatCompletionMessage, instance_id: str
    ) -> None:
        raw_text = message.content
        feedback = await self.callback.execute(instance_id, raw_text)
        messages.append(
            {
                "role": "user",
                "content": feedback,
            }
        )

    async def _preprocess(self, instance_id: str, metadata: dict | None = None):
        # create tool and callback instances
        tool_creation_coroutines = []
        for name, tool in self.available_tools.items():
            create_kwargs = metadata.get("tools", {}).get(name, {}).get("create_kwargs", {})
            tool_creation_coroutines.append(tool.create(instance_id, **create_kwargs))
        await asyncio.gather(*tool_creation_coroutines)
        if self.callback is not None:
            create_kwargs = metadata.get("callback", {}).get("create_kwargs", {})
            await self.callback.create(instance_id, **create_kwargs)

    async def _postprocess(self, instance_id: str) -> dict[str, float]:
        # release tool instances
        tool_release_coroutines = []
        for tool in self.available_tools.values():
            tool_release_coroutines.append(tool.release(instance_id))
        tool_rewards = await asyncio.gather(*tool_release_coroutines)
        if self.callback is not None:
            callback_reward = await self.callback.release(instance_id)
        else:
            callback_reward = 0.0
        return dict(
            zip(self.available_tools.keys(), tool_rewards, strict=False),
            **{"callback": callback_reward},
        )

    async def _generate_single_sample(
        self, task_id: str, sample_id: str, prompt: str, metadata: dict | None = None
    ) -> tuple[str, str, dict[str, str] | None]:
        # call tool creation coroutines
        instance_id = f"{task_id}-{sample_id}"
        await self._preprocess(instance_id, metadata)

        # multi-turn generation
        messages = self._build_messages(prompt)
        for _ in range(self.config.max_turns):
            response = await self.get_response_with_retry(messages)
            if response is None:
                break

            message = response.choices[0].message
            messages.append({"role": "assistant", "content": message.content})
            content = message.content

            # Reached max tokens
            if response.choices[0].finish_reason == "length":
                break

            # tool call handler
            if message.tool_calls:
                await self._handle_tool_call(messages, message, instance_id)
            elif self.callback is not None:
                await self._handle_callback(messages, message, instance_id)
            else:
                break

            if self.callback is not None and await self.callback.check(instance_id, content):
                break

        rewards = await self._postprocess(instance_id)

        response = response.model_dump() if response is not None else {}
        response["messages"] = [m if isinstance(m, dict) else m.model_dump() for m in messages]
        response["content"] = content
        response["reward"] = dict(zip(self.available_tools.keys(), rewards, strict=False))

        return task_id, sample_id, response
