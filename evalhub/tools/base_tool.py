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

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> str:
        return "Updated the tool state."

    async def release(self, instance_id: str, **kwargs) -> float:
        reward = self.instances[instance_id].pop("reward", 0.0)
        del self.instances[instance_id]
        return reward
