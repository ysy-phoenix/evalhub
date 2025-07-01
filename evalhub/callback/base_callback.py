from collections import defaultdict
from uuid import uuid4


class BaseCallback:
    def __init__(self):
        self.instances = defaultdict(dict)

    async def create(self, instance_id: str | None = None, **kwargs) -> str:
        if instance_id is None:
            return str(uuid4())
        else:
            return instance_id

    async def check(self, instance_id: str, response: str, **kwargs) -> bool:
        return True

    async def execute(self, instance_id: str, response: str, **kwargs) -> str:
        return "Updated the callback state."

    async def release(self, instance_id: str, **kwargs) -> float:
        reward = self.instances[instance_id].pop("reward", 0.0)
        del self.instances[instance_id]
        return reward
