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

"""Core module: environment, types, and model factories."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .types import (
    Action,
    Observation,
    RewardFunction,
    RewardResult,
    StepResult,
    TaskContext,
    TerminationReason,
    TokenObservation,
)

if TYPE_CHECKING:
    from .environment import AsyncEnvFactory, Environment, EnvironmentConfig
    from .models import ModelFactory

_LAZY_EXPORTS = {
    "AsyncEnvFactory": (".environment", "AsyncEnvFactory"),
    "Environment": (".environment", "Environment"),
    "EnvironmentConfig": (".environment", "EnvironmentConfig"),
    "ModelFactory": (".models", "ModelFactory"),
}

__all__ = [
    "Action",
    "AsyncEnvFactory",
    "Environment",
    "EnvironmentConfig",
    "ModelFactory",
    "Observation",
    "RewardFunction",
    "RewardResult",
    "StepResult",
    "TaskContext",
    "TerminationReason",
    "TokenObservation",
]


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
