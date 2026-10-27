# SPDX-License-Identifier: BSD-3-Clause

"""Data models for the Lost Update environment.

Actions and observations are the standard MCP types (`CallToolAction`,
`CallToolObservation`); only the state is specific to this environment.
"""

from typing import Optional

from openenv.core.env_server.types import State
from pydantic import Field


class LostUpdateState(State):
    seed: Optional[int] = Field(default=None, description="Seed that generated the episode")
    tier: Optional[str] = Field(default=None, description="Contention tier, T0 to T3")
    uri: Optional[str] = Field(default=None, description="URI of the shared document")
