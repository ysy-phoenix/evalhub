import asyncio
import importlib.util
import json
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from evalhub.inference.generator import LLMGenerator
from evalhub.inference.schemas import GenerationConfig
from evalhub.tools.base_tool import BaseTool


class MultiTurnGenerator(LLMGenerator):
    def __init__(self, config: GenerationConfig, system_prompt: str | None = None) -> None:
        super().__init__(config, system_prompt)
        self.tokenizer = AutoTokenizer.from_pretrained(config.sample_params.model)
        self._initialize_tools(config.tool_config_path)

    def _initialize_tools(self, tool_config_path: Path) -> None:
        tools_config = OmegaConf.load(tool_config_path)
        self.available_tools = defaultdict(Callable)
        self.tool_schemas: list[dict] = []
        for tool_config in tools_config.tools:
            cls_name = tool_config.class_name
            module_name, class_name = cls_name.rsplit(".", 1)

            if module_name not in sys.modules:
                spec = importlib.util.find_spec(module_name)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            else:
                module = sys.modules[module_name]

            tool_schema_dict = OmegaConf.to_container(tool_config.tool_schema, resolve=True)
            self.tool_schemas.append(tool_schema_dict)

            tool_cls: BaseTool = getattr(module, class_name)
            tool = tool_cls(
                name=tool_config.tool_schema.function.name,
                config=OmegaConf.to_container(tool_config.config, resolve=True),
            )
            self.available_tools[tool.name] = tool

    async def _generate_single_sample(
        self, task_id: str, sample_id: str, prompt: str, metadata: dict | None = None
    ) -> tuple[str, str, dict[str, str] | None]:
        # call tool creation coroutines
        instance_id = f"{task_id}-{sample_id}"
        tool_creation_coroutines = []
        for name, tool in self.available_tools.items():
            create_kwargs = metadata.get("tools", {}).get(name, {}).get("create_kwargs", {})
            tool_creation_coroutines.append(tool.create(instance_id, **create_kwargs))
        await asyncio.gather(*tool_creation_coroutines)

        # multi-turn generation
        messages = self._build_messages(prompt)
        for _ in range(self.config.max_turns):
            response = await self.complete(messages, tools=self.tool_schemas)
            message = response.choices[0].message
            messages.append(message)

            if message.tool_calls:
                tool_call_routines = []
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    if isinstance(tool_call.function.arguments, str):
                        arguments = json.loads(tool_call.function.arguments)
                    else:
                        arguments = tool_call.function.arguments
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
            else:
                break

        # release tool instances
        tool_release_coroutines = []
        for tool in self.available_tools.values():
            tool_release_coroutines.append(tool.release(instance_id))
        await asyncio.gather(*tool_release_coroutines)

        return task_id, sample_id, response.model_dump()
