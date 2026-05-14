# Copyright 2025-2026 Horizon RL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tools for `strands_env`."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .agent_core_pool import AgentCorePool, AgentCorePoolConfig, AgentCorePoolStats, AgentCoreSessionState
from .code_interpreter import CodeInterpreterQuotas, CodeInterpreterToolkit, create_aio_client

if TYPE_CHECKING:
    from .web_scraper import WebScraperToolkit
    from .web_search import WebSearchToolkit

_LAZY_EXPORTS = {
    "WebScraperToolkit": (".web_scraper", "WebScraperToolkit"),
    "WebSearchToolkit": (".web_search", "WebSearchToolkit"),
}

__all__ = [
    "AgentCorePool",
    "AgentCorePoolConfig",
    "AgentCorePoolStats",
    "AgentCoreSessionState",
    "CodeInterpreterQuotas",
    "CodeInterpreterToolkit",
    "create_aio_client",
    "WebScraperToolkit",
    "WebSearchToolkit",
]


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
