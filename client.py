# SPDX-License-Identifier: BSD-3-Clause

"""Client for the Lost Update environment.

    >>> from lost_update_env import LostUpdateEnv
    >>> with LostUpdateEnv(base_url="http://localhost:8000").sync() as env:
    ...     obs = env.reset(seed=7, tier="T1")
    ...     uri = obs.observation.metadata["uri"]
    ...     doc = env.call_tool("read", uri=uri)
    ...     env.call_tool("update", uri=uri, content=doc["content"], expected_revision=doc["revision"])
"""

from openenv.core.mcp_client import MCPToolClient


class LostUpdateEnv(MCPToolClient):
    """MCP tool client: `list_tools()`, `call_tool(name, **kwargs)`, `reset()`, `step()`."""
