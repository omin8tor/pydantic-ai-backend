"""Session management for Kubernetes sandboxes."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai_backends.backends.kubernetes.sandbox import KubernetesSandbox
    from pydantic_ai_backends.types import RuntimeConfig


class KubernetesSessionManager:
    """Manages user sessions backed by Kubernetes Pods.

    Drop-in replacement for :class:`SessionManager` that creates
    :class:`KubernetesSandbox` instances instead of Docker containers.
    """

    def __init__(
        self,
        default_runtime: RuntimeConfig | str | None = None,
        default_idle_timeout: int = 3600,
        namespace: str = "default",
        service_account: str | None = None,
        resources: dict[str, dict[str, str]] | None = None,
    ):
        self._sessions: dict[str, KubernetesSandbox] = {}
        self._default_runtime = default_runtime
        self._default_idle_timeout = default_idle_timeout
        self._namespace = namespace
        self._service_account = service_account
        self._resources = resources
        self._cleanup_task: asyncio.Task[None] | None = None

    @property
    def sessions(self) -> dict[str, KubernetesSandbox]:
        return dict(self._sessions)

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    async def get_or_create(
        self,
        session_id: str,
        runtime: RuntimeConfig | str | None = None,
    ) -> KubernetesSandbox:
        from pydantic_ai_backends.backends.kubernetes.sandbox import KubernetesSandbox

        if session_id in self._sessions:
            sandbox = self._sessions[session_id]
            if sandbox.is_alive():
                sandbox._last_activity = time.time()
                return sandbox
            del self._sessions[session_id]

        effective_runtime = runtime or self._default_runtime
        sandbox = KubernetesSandbox(
            runtime=effective_runtime,
            session_id=session_id,
            idle_timeout=self._default_idle_timeout,
            namespace=self._namespace,
            service_account=self._service_account,
            resources=self._resources,
        )
        sandbox.start()
        self._sessions[session_id] = sandbox
        return sandbox

    async def release(self, session_id: str) -> bool:
        if session_id not in self._sessions:
            return False
        sandbox = self._sessions.pop(session_id)
        sandbox.stop()
        return True

    async def cleanup_idle(self, max_idle: int | None = None) -> int:
        max_idle = max_idle if max_idle is not None else self._default_idle_timeout
        now = time.time()
        to_remove: list[str] = []

        for session_id, sandbox in self._sessions.items():
            if now - sandbox._last_activity > max_idle:
                to_remove.append(session_id)

        for session_id in to_remove:
            await self.release(session_id)

        return len(to_remove)

    def start_cleanup_loop(self, interval: int = 300) -> None:
        if self._cleanup_task is not None:
            return

        async def _loop() -> None:  # pragma: no cover
            while True:
                await asyncio.sleep(interval)
                await self.cleanup_idle()

        self._cleanup_task = asyncio.create_task(_loop())

    def stop_cleanup_loop(self) -> None:
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            self._cleanup_task = None

    async def shutdown(self) -> int:
        self.stop_cleanup_loop()
        count = len(self._sessions)
        session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            await self.release(session_id)
        return count

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._sessions

    def __len__(self) -> int:
        return len(self._sessions)
