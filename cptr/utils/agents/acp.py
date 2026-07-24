"""Small ACP JSON-RPC client for Cursor and Grok agents."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import suppress
from typing import Any, AsyncIterator

from cptr.utils.agents.acp_transport import (
    AcpTransport,
    StdioSubprocessTransport,
    TransportClosed,
)


class AcpClient:
    def __init__(
        self,
        *,
        command: str,
        args: list[str],
        cwd: str,
        env: dict[str, str],
        auth_method_id: str | None,
        client_capabilities: dict[str, Any] | None = None,
        resume_session_id: str | None = None,
        auto_approve_permissions: bool = False,
        transport: AcpTransport | None = None,
    ) -> None:
        self.command = command
        self.args = args
        self.cwd = cwd
        self.env = env
        self.auth_method_id = auth_method_id
        self.client_capabilities = client_capabilities or {}
        self.resume_session_id = resume_session_id
        self.auto_approve_permissions = auto_approve_permissions
        self.transport = transport
        self.reader_task: asyncio.Task | None = None
        self.pending: dict[Any, asyncio.Future[dict[str, Any]]] = {}
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.next_id = 1
        self.session_id: str | None = None
        self.initialize_result: dict[str, Any] = {}
        self.setup_result: dict[str, Any] = {}
        self.model_config_id: str | None = None

    async def start(self) -> None:
        if self.transport is None:
            self.transport = StdioSubprocessTransport(
                self.command, self.args, self.cwd or os.getcwd(), self.env
            )
        await self.transport.start()
        self.reader_task = asyncio.create_task(self._reader_loop())
        self.initialize_result = await self.request(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": {"readTextFile": False, "writeTextFile": False},
                    "terminal": False,
                    **self.client_capabilities,
                },
                "clientInfo": {"name": "cptr", "version": "0"},
            },
        )
        auth_method_id = self.auth_method_id or _first_auth_method(self.initialize_result)
        if auth_method_id:
            await self.request("authenticate", {"methodId": auth_method_id})
        await self._open_session()

    async def close(self) -> None:
        if self.reader_task:
            self.reader_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.reader_task
        if self.transport:
            await self.transport.close()
        for future in self.pending.values():
            if not future.done():
                future.cancel()
        self.pending.clear()

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self.next_id
        self.next_id += 1
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self.pending[request_id] = future
        assert self.transport is not None
        await self.transport.send(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        )
        response = await future
        if "error" in response:
            raise RuntimeError(str(response["error"]))
        result = response.get("result")
        return result if isinstance(result, dict) else {}

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        assert self.transport is not None
        await self.transport.send({"jsonrpc": "2.0", "method": method, "params": params})

    async def prompt(self, text: str, images: list[dict[str, str]] | None = None) -> dict[str, Any]:
        if not self.session_id:
            raise RuntimeError("ACP session has not started")
        prompt: list[dict[str, str]] = []
        if text.strip():
            prompt.append({"type": "text", "text": text})
        for image in images or []:
            prompt.append(
                {
                    "type": "image",
                    "data": image["data"],
                    "mimeType": image["mimeType"],
                }
            )
        return await self.request(
            "session/prompt",
            {"sessionId": self.session_id, "prompt": prompt},
        )

    async def cancel(self) -> None:
        if self.session_id:
            with suppress(Exception):
                await self.notify("session/cancel", {"sessionId": self.session_id})

    async def set_model(self, model: str) -> None:
        if not self.session_id:
            return
        if self.model_config_id:
            with suppress(Exception):
                await self.request(
                    "session/set_config_option",
                    {
                        "sessionId": self.session_id,
                        "configId": self.model_config_id,
                        "value": model,
                    },
                )
                return
        with suppress(Exception):
            await self.request(
                "session/set_model", {"sessionId": self.session_id, "modelId": model}
            )

    async def _open_session(self) -> None:
        setup: dict[str, Any] | None = None
        if self.resume_session_id:
            with suppress(Exception):
                setup = await self.request(
                    "session/load",
                    {"sessionId": self.resume_session_id, "cwd": self.cwd, "mcpServers": []},
                )
                self.session_id = self.resume_session_id
        if setup is None:
            setup = await self.request("session/new", {"cwd": self.cwd, "mcpServers": []})
            session_id = setup.get("sessionId")
            self.session_id = session_id if isinstance(session_id, str) else None
        if not self.session_id:
            raise RuntimeError("ACP did not return a session id")
        self.setup_result = setup
        self.model_config_id = _extract_model_config_id(setup)

    async def _reader_loop(self) -> None:
        assert self.transport is not None
        while True:
            try:
                message = await self.transport.receive()
            except TransportClosed:
                break
            await self._handle_message(message)

    async def _handle_message(self, message: dict[str, Any]) -> None:
        if (
            "id" in message
            and ("result" in message or "error" in message)
            and "method" not in message
        ):
            future = self.pending.pop(message["id"], None)
            if future and not future.done():
                future.set_result(message)
            return

        method = message.get("method")
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        if "id" in message and method == "session/request_permission":
            await self._reply_permission(message["id"], params)
            return
        await self.events.put(message)

    async def _reply_permission(self, request_id: Any, params: dict[str, Any]) -> None:
        option_id = None
        if self.auto_approve_permissions:
            option_id = _select_permission_option(
                params, "allow_always"
            ) or _select_permission_option(params, "allow_once")
        outcome = (
            {"outcome": {"outcome": "selected", "optionId": option_id}}
            if option_id
            else {"outcome": {"outcome": "cancelled"}}
        )
        assert self.transport is not None
        await self.transport.send({"jsonrpc": "2.0", "id": request_id, "result": outcome})


def _extract_model_config_id(setup: dict[str, Any]) -> str | None:
    options = setup.get("configOptions")
    if not isinstance(options, list):
        return None
    for option in options:
        if isinstance(option, dict) and option.get("category") == "model":
            option_id = option.get("id")
            if isinstance(option_id, str) and option_id.strip():
                return option_id.strip()
    return None


def _first_auth_method(initialize_result: dict[str, Any]) -> str | None:
    methods = initialize_result.get("authMethods")
    if not isinstance(methods, list):
        return None
    for method in methods:
        if isinstance(method, dict):
            method_id = method.get("id")
            if isinstance(method_id, str) and method_id.strip():
                return method_id.strip()
    return None


def _select_permission_option(params: dict[str, Any], kind: str) -> str | None:
    options = params.get("options")
    if not isinstance(options, list):
        return None
    for option in options:
        if isinstance(option, dict) and option.get("kind") == kind:
            option_id = option.get("optionId")
            if isinstance(option_id, str) and option_id.strip():
                return option_id.strip()
    return None


def acp_text_from_update(params: dict[str, Any]) -> str | None:
    update = params.get("update")
    if not isinstance(update, dict):
        return None
    if update.get("sessionUpdate") != "agent_message_chunk":
        return None
    content = update.get("content")
    if isinstance(content, dict) and content.get("type") == "text":
        text = content.get("text")
        return text if isinstance(text, str) and text else None
    return None


def acp_tool_from_update(params: dict[str, Any]) -> dict[str, Any] | None:
    update = params.get("update")
    if not isinstance(update, dict):
        return None
    if update.get("sessionUpdate") not in {"tool_call", "tool_call_update"}:
        return None
    call_id = update.get("toolCallId")
    if not isinstance(call_id, str) or not call_id.strip():
        return None

    raw_input = update.get("rawInput")
    command = _command_from_raw_input(raw_input)
    title = str(update.get("title") or update.get("kind") or "Agent tool").strip()
    status = _tool_status(update.get("status"))
    args: dict[str, Any] = {}
    if command:
        name = "run_command"
        args["command"] = command
    else:
        name = "agent_tool"
        args["title"] = title
        if raw_input is not None:
            args["input"] = raw_input
    output = _tool_output(update)
    return {
        "call_id": call_id.strip(),
        "name": name,
        "status": status,
        "arguments": args,
        "output": output,
    }


def _tool_status(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"completed", "success"}:
        return "completed"
    if normalized in {"failed", "error"}:
        return "failed"
    if normalized in {"pending"}:
        return "pending"
    return "in_progress"


def _command_from_raw_input(raw_input: Any) -> str | None:
    if not isinstance(raw_input, dict):
        return None
    command = raw_input.get("command")
    if isinstance(command, str) and command.strip():
        return command.strip()
    executable = raw_input.get("executable")
    args = raw_input.get("args")
    if isinstance(executable, str) and isinstance(args, list):
        parts = [executable, *(str(arg) for arg in args)]
        return " ".join(part for part in parts if part.strip())
    return None


def _tool_output(update: dict[str, Any]) -> str | None:
    raw_output = update.get("rawOutput")
    if raw_output is not None:
        return raw_output if isinstance(raw_output, str) else json.dumps(raw_output, indent=2)
    content = update.get("content")
    if not isinstance(content, list):
        return None
    text: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        nested = item.get("content")
        if isinstance(nested, dict) and nested.get("type") == "text":
            value = nested.get("text")
            if isinstance(value, str) and value.strip():
                text.append(value.strip())
    return "\n".join(text) or None


def acp_models_from_setup(setup: dict[str, Any]) -> list[str]:
    models = setup.get("models")
    if not isinstance(models, dict):
        return []
    available = models.get("availableModels")
    if not isinstance(available, list):
        return []
    result = []
    for item in available:
        if isinstance(item, dict):
            model = item.get("modelId")
            if isinstance(model, str) and model.strip():
                result.append(model.strip())
    return result


async def acp_turn_event_stream(
    client: AcpClient, prompt_task: asyncio.Task, *, grace: float = 0.25
) -> AsyncIterator[dict[str, Any]]:
    """Yield session events for one turn, ending shortly after the prompt completes.

    Races the events queue against the prompt task instead of only checking
    ``prompt_task.done()`` after an event arrives: when the agent's final
    notifications and the prompt response land in a single read burst, no further
    event ever arrives to trigger that check, and a naive ``events.get()`` loop
    blocks forever. After the prompt resolves, keeps draining events that arrive
    within ``grace`` seconds so trailing notifications aren't dropped.
    """
    get_task: asyncio.Task | None = None
    try:
        while True:
            get_task = asyncio.create_task(client.events.get())
            if not prompt_task.done():
                await asyncio.wait({get_task, prompt_task}, return_when=asyncio.FIRST_COMPLETED)
            if get_task.done():
                yield get_task.result()
                continue
            try:
                yield await asyncio.wait_for(get_task, timeout=grace)
            except asyncio.TimeoutError:
                return
    finally:
        if get_task is not None and not get_task.done():
            get_task.cancel()
            with suppress(asyncio.CancelledError):
                await get_task
