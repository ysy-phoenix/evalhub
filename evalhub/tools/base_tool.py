from collections import defaultdict
from typing import Any
from uuid import uuid4


class BaseTool:
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.instances = defaultdict(dict)

    async def create(self, instance_id: str | None = None, **kwargs) -> str:
        if instance_id is None:
            return str(uuid4())
        else:
            return instance_id

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[str, float, dict]:
        return "Updated the tool state.", 0.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        pass
