"""File storage and sandbox backends for AI agents.

pydantic-ai-backend provides a unified interface for file storage and
command execution across different backends (in-memory, local, Docker).

Basic usage:
    ```python
    from pydantic_ai_backends import StateBackend, LocalBackend

    # In-memory storage (for testing)
    backend = StateBackend()
    backend.write("/app.py", "print('hello')")
    content = backend.read("/app.py")

    # Local filesystem with shell
    backend = LocalBackend("/workspace")
    result = backend.execute("python app.py")
    ```

Console toolset for AI agents:
    ```python
    from dataclasses import dataclass
    from pydantic_ai_backends import LocalBackend, create_console_toolset

    @dataclass
    class MyDeps:
        backend: LocalBackend

    toolset = create_console_toolset()
    # Provides: ls, read_file, write_file, edit_file, glob, grep, execute
    ```

Docker sandbox (requires optional dependencies):
    ```python
    from pydantic_ai_backends import DockerSandbox, RuntimeConfig

    # pip install pydantic-ai-backend[docker]
    sandbox = DockerSandbox(image="python:3.12-slim")
    result = sandbox.execute("python -c 'print(1+1)'")
    print(result.output)  # "2"
    ```
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Core exports - always available
from pydantic_ai_backends.backends.composite import CompositeBackend
from pydantic_ai_backends.backends.local import LocalBackend
from pydantic_ai_backends.backends.state import StateBackend
from pydantic_ai_backends.protocol import BackendProtocol, SandboxProtocol
from pydantic_ai_backends.types import (
    EditResult,
    ExecuteResponse,
    FileData,
    FileInfo,
    GrepMatch,
    RuntimeConfig,
    WriteResult,
)

if TYPE_CHECKING:
    from pydantic_ai_backends.backends.docker import (
        BUILTIN_RUNTIMES,
        BaseSandbox,
        DockerSandbox,
        SessionManager,
    )
    from pydantic_ai_backends.backends.docker.runtimes import get_runtime
    from pydantic_ai_backends.backends.kubernetes import (
        KubernetesSandbox,
        KubernetesSessionManager,
    )
    from pydantic_ai_backends.hashline import (
        apply_hashline_edit,
        apply_hashline_edit_with_summary,
        format_hashline_output,
        line_hash,
    )
    from pydantic_ai_backends.permissions import (
        DEFAULT_RULESET,
        PERMISSIVE_RULESET,
        READONLY_RULESET,
        SECRETS_PATTERNS,
        STRICT_RULESET,
        SYSTEM_PATTERNS,
        AskCallback,
        AskFallback,
        OperationPermissions,
        PermissionAction,
        PermissionChecker,
        PermissionDeniedError,
        PermissionError,
        PermissionOperation,
        PermissionRule,
        PermissionRuleset,
        create_ruleset,
    )
    from pydantic_ai_backends.toolsets.console import (
        DEFAULT_MAX_IMAGE_BYTES,
        HASHLINE_CONSOLE_PROMPT,
        IMAGE_EXTENSIONS,
        IMAGE_MEDIA_TYPES,
        ConsoleDeps,
        ConsoleToolset,
        EditFormat,
        create_console_toolset,
        get_console_system_prompt,
    )

# Lazy loading for optional dependencies
_LAZY_IMPORTS = {
    # Hashline editing
    "line_hash": "pydantic_ai_backends.hashline",
    "format_hashline_output": "pydantic_ai_backends.hashline",
    "apply_hashline_edit": "pydantic_ai_backends.hashline",
    "apply_hashline_edit_with_summary": "pydantic_ai_backends.hashline",
    # Console toolset (requires pydantic-ai)
    "create_console_toolset": "pydantic_ai_backends.toolsets.console",
    "get_console_system_prompt": "pydantic_ai_backends.toolsets.console",
    "ConsoleToolset": "pydantic_ai_backends.toolsets.console",
    "ConsoleDeps": "pydantic_ai_backends.toolsets.console",
    "EditFormat": "pydantic_ai_backends.toolsets.console",
    "HASHLINE_CONSOLE_PROMPT": "pydantic_ai_backends.toolsets.console",
    "IMAGE_EXTENSIONS": "pydantic_ai_backends.toolsets.console",
    "IMAGE_MEDIA_TYPES": "pydantic_ai_backends.toolsets.console",
    "DEFAULT_MAX_IMAGE_BYTES": "pydantic_ai_backends.toolsets.console",
    # Docker sandbox (requires docker extra)
    "DockerSandbox": "pydantic_ai_backends.backends.docker.sandbox",
    "BaseSandbox": "pydantic_ai_backends.backends.docker.sandbox",
    "SessionManager": "pydantic_ai_backends.backends.docker.session",
    "BUILTIN_RUNTIMES": "pydantic_ai_backends.backends.docker.runtimes",
    "get_runtime": "pydantic_ai_backends.backends.docker.runtimes",
    # Kubernetes sandbox (requires kubernetes extra)
    "KubernetesSandbox": "pydantic_ai_backends.backends.kubernetes.sandbox",
    "KubernetesSessionManager": "pydantic_ai_backends.backends.kubernetes.session",
    # Permissions system
    "PermissionAction": "pydantic_ai_backends.permissions",
    "PermissionOperation": "pydantic_ai_backends.permissions",
    "PermissionRule": "pydantic_ai_backends.permissions",
    "OperationPermissions": "pydantic_ai_backends.permissions",
    "PermissionRuleset": "pydantic_ai_backends.permissions",
    "PermissionChecker": "pydantic_ai_backends.permissions",
    "PermissionError": "pydantic_ai_backends.permissions",
    "PermissionDeniedError": "pydantic_ai_backends.permissions",
    "AskCallback": "pydantic_ai_backends.permissions",
    "AskFallback": "pydantic_ai_backends.permissions",
    "DEFAULT_RULESET": "pydantic_ai_backends.permissions",
    "PERMISSIVE_RULESET": "pydantic_ai_backends.permissions",
    "READONLY_RULESET": "pydantic_ai_backends.permissions",
    "STRICT_RULESET": "pydantic_ai_backends.permissions",
    "SECRETS_PATTERNS": "pydantic_ai_backends.permissions",
    "SYSTEM_PATTERNS": "pydantic_ai_backends.permissions",
    "create_ruleset": "pydantic_ai_backends.permissions",
}


def __getattr__(name: str) -> object:
    """Lazy loading for optional dependencies."""
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Protocols
    "BackendProtocol",
    "SandboxProtocol",
    # Types
    "FileData",
    "FileInfo",
    "WriteResult",
    "EditResult",
    "ExecuteResponse",
    "GrepMatch",
    "RuntimeConfig",
    # Backends
    "StateBackend",
    "LocalBackend",
    "CompositeBackend",
    # Hashline editing
    "line_hash",
    "format_hashline_output",
    "apply_hashline_edit",
    "apply_hashline_edit_with_summary",
    # Console toolset (requires pydantic-ai)
    "create_console_toolset",
    "get_console_system_prompt",
    "ConsoleToolset",
    "ConsoleDeps",
    "EditFormat",
    "HASHLINE_CONSOLE_PROMPT",
    # Image support constants
    "IMAGE_EXTENSIONS",
    "IMAGE_MEDIA_TYPES",
    "DEFAULT_MAX_IMAGE_BYTES",
    # Docker sandbox (optional - requires docker extra)
    "BaseSandbox",
    "DockerSandbox",
    "SessionManager",
    # Runtimes
    "BUILTIN_RUNTIMES",
    "get_runtime",
    # Kubernetes sandbox (optional - requires kubernetes extra)
    "KubernetesSandbox",
    "KubernetesSessionManager",
    # Permissions system
    "PermissionAction",
    "PermissionOperation",
    "PermissionRule",
    "OperationPermissions",
    "PermissionRuleset",
    "PermissionChecker",
    "PermissionError",
    "PermissionDeniedError",
    "AskCallback",
    "AskFallback",
    "DEFAULT_RULESET",
    "PERMISSIVE_RULESET",
    "READONLY_RULESET",
    "STRICT_RULESET",
    "SECRETS_PATTERNS",
    "SYSTEM_PATTERNS",
    "create_ruleset",
]

try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("pydantic-ai-backend")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"
