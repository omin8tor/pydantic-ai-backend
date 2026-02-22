"""Kubernetes sandbox for isolated command execution."""

from __future__ import annotations

import io
import re
import shlex
import tarfile
import time
import uuid
from pathlib import PurePosixPath
from typing import Any

from pydantic_ai_backends.backends.docker.sandbox import BaseSandbox
from pydantic_ai_backends.types import (
    EditResult,
    ExecuteResponse,
    RuntimeConfig,
    WriteResult,
)

_LABEL_UNSAFE = re.compile(r"[^a-zA-Z0-9._-]")
_MAX_LABEL_LEN = 63


def _sanitize_label(value: str) -> str:
    safe = _LABEL_UNSAFE.sub("-", value)
    safe = safe.strip("-.")
    return safe[:_MAX_LABEL_LEN] if safe else "unknown"


def _load_k8s() -> Any:  # pragma: no cover
    try:
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        return client
    except ImportError as e:
        raise ImportError(
            "kubernetes package not installed. "
            "Install with: pip install pydantic-ai-backend[kubernetes]"
        ) from e


class KubernetesSandbox(BaseSandbox):  # pragma: no cover
    """Kubernetes Pod-based sandbox for isolated command execution.

    Creates a Pod running ``sleep infinity`` and executes commands via
    the Kubernetes exec API, mirroring :class:`DockerSandbox` semantics.
    """

    def __init__(
        self,
        image: str = "python:3.12-slim",
        sandbox_id: str | None = None,
        work_dir: str = "/workspace",
        namespace: str = "default",
        runtime: RuntimeConfig | str | None = None,
        session_id: str | None = None,
        idle_timeout: int = 3600,
        service_account: str | None = None,
        labels: dict[str, str] | None = None,
        resources: dict[str, dict[str, str]] | None = None,
    ):
        effective_id = session_id or sandbox_id
        super().__init__(effective_id)

        self._namespace = namespace
        self._idle_timeout = idle_timeout
        self._last_activity = time.time()
        self._service_account = service_account
        self._extra_labels = labels or {}
        self._resources = resources
        self._pod_name: str | None = None
        self._client: Any = None

        if runtime is not None:
            if isinstance(runtime, str):
                from pydantic_ai_backends.backends.docker.runtimes import get_runtime

                runtime = get_runtime(runtime)
            self._runtime: RuntimeConfig | None = runtime
            self._work_dir = runtime.work_dir
            self._image = runtime.image or runtime.base_image or image
        else:
            self._runtime = None
            self._work_dir = work_dir
            self._image = image

    @property
    def runtime(self) -> RuntimeConfig | None:
        return self._runtime

    @property
    def session_id(self) -> str:
        return self._id

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def pod_name(self) -> str | None:
        return self._pod_name

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = _load_k8s()
        return self._client

    def _build_pod_name(self) -> str:
        suffix = uuid.uuid4().hex[:8]
        base = _sanitize_label(self._id)
        max_base = _MAX_LABEL_LEN - len(suffix) - 4  # "pab-" prefix + "-" separator
        return f"pab-{base[:max_base]}-{suffix}"

    def _ensure_pod(self) -> None:
        if self._pod_name is not None:
            return

        k8s = self._get_client()
        v1 = k8s.CoreV1Api()

        self._pod_name = self._build_pod_name()

        env_vars = []
        if self._runtime and self._runtime.env_vars:
            env_vars = [k8s.V1EnvVar(name=k, value=v) for k, v in self._runtime.env_vars.items()]

        resource_reqs = None
        if self._resources:
            resource_reqs = k8s.V1ResourceRequirements(
                limits=self._resources.get("limits"),
                requests=self._resources.get("requests"),
            )

        container = k8s.V1Container(
            name="sandbox",
            image=self._image,
            command=["sleep", "infinity"],
            working_dir=self._work_dir,
            env=env_vars or None,
            resources=resource_reqs,
        )

        labels = {
            "app.kubernetes.io/managed-by": "pydantic-ai-backend",
            "pydantic-ai-backend/session-id": _sanitize_label(self._id),
        }
        labels.update(self._extra_labels)

        annotations = {
            "pydantic-ai-backend/last-activity": str(time.time()),
        }

        pod = k8s.V1Pod(
            metadata=k8s.V1ObjectMeta(
                name=self._pod_name,
                namespace=self._namespace,
                labels=labels,
                annotations=annotations,
            ),
            spec=k8s.V1PodSpec(
                containers=[container],
                restart_policy="Never",
                service_account_name=self._service_account,
            ),
        )

        v1.create_namespaced_pod(namespace=self._namespace, body=pod)
        self._wait_for_running(v1)

    def _wait_for_running(self, v1: Any, timeout: int = 60) -> None:
        from kubernetes.client.rest import ApiException

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                pod = v1.read_namespaced_pod_status(name=self._pod_name, namespace=self._namespace)
                phase = pod.status.phase if pod.status else None
                if phase == "Running":
                    return
                if phase in ("Failed", "Succeeded"):
                    raise RuntimeError(f"Pod {self._pod_name} entered {phase} state")
            except ApiException as e:
                if e.status == 404:
                    raise RuntimeError(f"Pod {self._pod_name} not found") from e
                raise
            time.sleep(0.5)
        raise RuntimeError(f"Pod {self._pod_name} did not reach Running within {timeout}s")

    def execute(self, command: str, timeout: int | None = None) -> ExecuteResponse:
        from kubernetes.stream import stream

        self._ensure_pod()
        self._last_activity = time.time()

        k8s = self._get_client()
        v1 = k8s.CoreV1Api()

        try:
            if timeout:
                exec_cmd = ["timeout", str(timeout), "sh", "-c", command]
            else:
                exec_cmd = ["sh", "-c", command]

            resp = stream(
                v1.connect_get_namespaced_pod_exec,
                name=self._pod_name,
                namespace=self._namespace,
                command=exec_cmd,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False,
                _preload_content=False,
            )

            stdout_data = ""
            stderr_data = ""
            while resp.is_open():
                resp.update(timeout=1)
                if resp.peek_stdout():
                    stdout_data += resp.read_stdout()
                if resp.peek_stderr():
                    stderr_data += resp.read_stderr()

            resp.close()

            exit_code = resp.returncode
            output = stdout_data + stderr_data if stderr_data else stdout_data

            max_output = 100000
            truncated = len(output) > max_output
            if truncated:
                output = output[:max_output]

            return ExecuteResponse(
                output=output,
                exit_code=exit_code,
                truncated=truncated,
            )
        except Exception as e:
            return ExecuteResponse(
                output=f"Error: {e}",
                exit_code=1,
                truncated=False,
            )

    def _read_bytes(self, path: str) -> bytes:
        safe_path = shlex.quote(path)
        result = self.execute(f"cat {safe_path}")
        if result.exit_code != 0:
            return f"[Error: {result.output}]".encode()
        return result.output.encode("utf-8", errors="replace")

    def write(self, path: str, content: str | bytes) -> WriteResult:
        from kubernetes.stream import stream as k8s_stream

        self._ensure_pod()
        k8s = self._get_client()
        v1 = k8s.CoreV1Api()

        try:
            posix_path = PurePosixPath(path)
            parent_dir = str(posix_path.parent)
            filename = posix_path.name

            safe_parent = shlex.quote(parent_dir)
            mkdir_result = self.execute(f"mkdir -p {safe_parent}")
            if mkdir_result.exit_code != 0:
                return WriteResult(error=f"Failed to create directory: {mkdir_result.output}")

            content_bytes = content if isinstance(content, bytes) else content.encode()
            tar_buffer = io.BytesIO()
            with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
                tarinfo = tarfile.TarInfo(name=filename)
                tarinfo.size = len(content_bytes)
                tarinfo.mtime = int(time.time())
                tarinfo.mode = 0o644
                tar.addfile(tarinfo, io.BytesIO(content_bytes))
            tar_bytes = tar_buffer.getvalue()

            resp = k8s_stream(
                v1.connect_get_namespaced_pod_exec,
                name=self._pod_name,
                namespace=self._namespace,
                command=["tar", "xf", "-", "-C", parent_dir],
                stderr=True,
                stdin=True,
                stdout=True,
                tty=False,
                _preload_content=False,
            )

            resp.write_stdin(tar_bytes)
            resp.close()

            return WriteResult(path=path)
        except Exception as e:
            return WriteResult(error=f"Failed to write file: {e}")

    def edit(
        self, path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> EditResult:
        try:
            file_bytes = self._read_bytes(path)
            if file_bytes.startswith(b"[Error:"):
                return EditResult(error=file_bytes.decode("utf-8", errors="replace"))

            content = file_bytes.decode("utf-8", errors="replace")
            occurrences = content.count(old_string)
            if occurrences == 0:
                return EditResult(error="String not found in file")

            if occurrences > 1 and not replace_all:
                return EditResult(
                    error=f"String found {occurrences} times. "
                    "Use replace_all=True to replace all, or provide more context."
                )

            new_content = content.replace(old_string, new_string)
            write_result = self.write(path, new_content)

            if write_result.error:
                return EditResult(error=write_result.error)

            return EditResult(path=path, occurrences=occurrences)
        except Exception as e:
            return EditResult(error=f"Failed to edit file: {e}")

    def start(self) -> None:
        self._ensure_pod()

    def is_alive(self) -> bool:
        if self._pod_name is None:
            return False
        try:
            k8s = self._get_client()
            v1 = k8s.CoreV1Api()
            pod = v1.read_namespaced_pod_status(name=self._pod_name, namespace=self._namespace)
            return pod.status.phase == "Running" if pod.status else False
        except Exception:
            return False

    def stop(self) -> None:
        if self._pod_name is None:
            return
        try:
            k8s = self._get_client()
            v1 = k8s.CoreV1Api()
            v1.delete_namespaced_pod(
                name=self._pod_name,
                namespace=self._namespace,
                grace_period_seconds=5,
            )
        except Exception:
            pass
        self._pod_name = None

    def __del__(self) -> None:
        if hasattr(self, "_pod_name"):
            self.stop()
