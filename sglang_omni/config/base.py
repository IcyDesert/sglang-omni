# SPDX-License-Identifier: Apache-2.0
"""Recursive conflict checks for resolved configuration models."""

from pydantic import BaseModel


class ConfigModel(BaseModel):
    """Override check_conflicts for cross-field rules; call super first so nested models go first."""

    def check_conflicts(self) -> None:
        """Check nested models before this one, without changing configuration values."""
        pending = [getattr(self, name) for name in type(self).model_fields]
        while pending:
            value = pending.pop()
            if isinstance(value, ConfigModel):
                value.check_conflicts()
            elif isinstance(value, BaseModel):
                pending.extend(
                    getattr(value, name) for name in type(value).model_fields
                )
            elif isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, (list, tuple)):
                pending.extend(value)
            else:
                pass
