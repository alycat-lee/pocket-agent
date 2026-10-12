from __future__ import annotations

import json
import logging
import threading
from unittest.mock import MagicMock, patch

import httpx
import pytest
from pocket_agent import llm_client as llm_client_mod
from pocket_agent.config import AgentConfig
from pocket_agent.llm_client import (
    EMPTY_THINK_NOTE,
    LlamaCppClient,
    LLMAbortedError,
    LLMConnectionError,
    LLMFormatError,
    OpenAICompatibleClient,
    parse_usage,
    resolve_trust_env,
)


def _openai_response(
    content: str,
    status_code: int = 200,
    usage: dict | None = None,
) -> httpx.Response:
    body: dict = {
        "choices": [
            {"message": {"role": "assistant", "content": content}},
        ]
    }
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(
        status_code,
        json=body,
        request=httpx.Request("POST", "http://localhost:8080/v1/chat/completions"),
    )


def _as_stream_cm(item: object) -> object:
    if isinstance(item, BaseException):
        return item
    cm = MagicMock()
    cm.__enter__.return_value = item
    cm.__exit__.return_value = None
    return cm


def _patch_client(post_side_effect: object) -> MagicMock:
    mock_client = MagicMock()
    items = (
        list(post_side_effect)
        if isinstance(post_side_effect, list)
        else [post_side_effect]
    )
    mock_client.stream.side_effect = [_as_stream_cm(item) for item in items]
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    return mock_client


def test_generate_parses_content() -> None:
    config = AgentConfig()
    client = LlamaCppClient(config)
    mock_http = _patch_client([_openai_response("最终回答")])

    with patch("pocket_agent.llm_client.httpx.Client", return_value=mock_http):
        result = client.generate("请回答")

    assert result == "最终回答"
    payload = mock_http.stream.call_args.kwargs["json"]
    assert payload["messages"] == [
        {"role": "system", "content": ""},
        {"role": "user", "content": "请回答"},
    ]
    assert payload["temperature"] == config.temperature
    assert payload["repeat_penalty"] == config.repeat_penalty
    assert payload["top_p"] == config.top_p
    assert payload["max_tokens"] == config.max_tokens
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "response_format" not in payload
    assert client.last_usage is None


def _empty_reasoning_response(reasoning: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": reasoning,
                    }
                }
            ]
        },
        request=httpx.Request("POST", "http://localhost:8000/v1/chat/completions"),
    )


def test_empty_content_retries_then_returns_text() -> None:
    client = OpenAICompatibleClient(AgentConfig(), thinking="off")
    mock_http = _patch_client(
        [_empty_reasoning_response("先想一遍"), _openai_response("正常正文")]
    )
    with patch("pocket_agent.llm_client.httpx.Client", return_value=mock_http):
        assert client.generate("问") == "正常正文"
    assert mock_http.stream.call_count == 2


def test_empty_content_twice_returns_thinking_tail() -> None:
    client = OpenAICompatibleClient(AgentConfig(), thinking="strip")
    reasoning = ("前段推理" * 40) + "要点结尾7381"
    mock_http = _patch_client(
        [_empty_reasoning_response(reasoning), _empty_reasoning_response(reasoning)]
    )
    with patch("pocket_agent.llm_client.httpx.Client", return_value=mock_http):
        text = client.generate("问")
    assert EMPTY_THINK_NOTE in text
    assert "要点结尾7381" in text
    assert "[生成回答失败]" not in text
    assert "empty content" not in text
    assert len(text) <= 300 + len(EMPTY_THINK_NOTE) + 5


def test_generate_strips_think_blocks_from_answer() -> None:
    client = LlamaCppClient(AgentConfig())
    mock_http = _patch_client(
        [_openai_response("<think>不该出现</think>可见正文")]
    )
    with patch("pocket_agent.llm_client.httpx.Client", return_value=mock_http):
        assert client.generate("问") == "可见正文"
    assert client.last_thinking == ""


def test_reasoning_enabled_captures_thinking_not_answer() -> None:
    client = LlamaCppClient(AgentConfig(reasoning_enabled=True))
    mock_http = _patch_client(
        [_openai_response("<think>推理过程</think>最终答案")]
    )
    with patch("pocket_agent.llm_client.httpx.Client", return_value=mock_http):
        assert client.generate("问") == "最终答案"
    assert "推理过程" in client.last_thinking
    payload = mock_http.stream.call_args.kwargs["json"]
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}


def test_generate_int_from_json_schema_value() -> None:
    client = LlamaCppClient(AgentConfig())
    mock_http = _patch_client([_openai_response('{"value": 2}')])

    with patch("pocket_agent.llm_client.httpx.Client", return_value=mock_http):
        result = client.generate_int("选工具", minimum=-1, maximum=3)

    assert result == 2
    payload = mock_http.stream.call_args.kwargs["json"]
    assert payload["response_format"]["type"] == "json_schema"
    schema = payload["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["value"]["minimum"] == -1
    assert schema["properties"]["value"]["maximum"] == 3


def test_generate_int_regex_fallback() -> None:
    client = LlamaCppClient(AgentConfig())
    mock_http = _patch_client([_openai_response("我认为应该选 1 号工具")])

    with patch("pocket_agent.llm_client.httpx.Client", return_value=mock_http):
        assert client.generate_int("选", minimum=-1, maximum=5) == 1


def test_generate_int_out_of_range_raises_format_error() -> None:
    client = LlamaCppClient(AgentConfig())
    mock_http = _patch_client([_openai_response('{"value": 99}')])

    with (
        patch("pocket_agent.llm_client.httpx.Client", return_value=mock_http),
        pytest.raises(LLMFormatError),
    ):
        client.generate_int("选", minimum=-1, maximum=2)


def test_generate_json_success() -> None:
    client = LlamaCppClient(AgentConfig())
    body = {"city": "北京"}
    mock_http = _patch_client([_openai_response(json.dumps(body, ensure_ascii=False))])
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }

    with patch("pocket_agent.llm_client.httpx.Client", return_value=mock_http):
        result = client.generate_json("填参", schema)

    assert result == body
    rf = mock_http.stream.call_args.kwargs["json"]["response_format"]
    assert rf["json_schema"]["schema"] == schema


def test_generate_json_invalid_raises_format_error() -> None:
    client = LlamaCppClient(AgentConfig())
    mock_http = _patch_client([_openai_response("not-json")])

    with (
        patch("pocket_agent.llm_client.httpx.Client", return_value=mock_http),
        pytest.raises(LLMFormatError),
    ):
        client.generate_json("填参", {"type": "object"})


def test_retries_on_5xx_then_succeeds() -> None:
    client = LlamaCppClient(AgentConfig())
    mock_http = _patch_client(
        [
            _openai_response("err", status_code=503),
            _openai_response("err", status_code=500),
            _openai_response("ok"),
        ]
    )

    with (
        patch("pocket_agent.llm_client.httpx.Client", return_value=mock_http),
        patch("pocket_agent.llm_client.time.sleep") as sleep,
    ):
        assert client.generate("hi") == "ok"

    assert mock_http.stream.call_count == 3
    assert sleep.call_args_list == [((1.0,),), ((2.0,),)]


def test_retries_exhausted_raises_connection_error() -> None:
    client = LlamaCppClient(AgentConfig())
    mock_http = _patch_client(
        [
            _openai_response("e", status_code=502),
            _openai_response("e", status_code=502),
            _openai_response("e", status_code=502),
        ]
    )

    with (
        patch("pocket_agent.llm_client.httpx.Client", return_value=mock_http),
        patch("pocket_agent.llm_client.time.sleep"),
        pytest.raises(LLMConnectionError),
    ):
        client.generate("hi")

    assert mock_http.stream.call_count == 3


def test_retries_on_connect_error() -> None:
    client = LlamaCppClient(AgentConfig())
    request = httpx.Request("POST", "http://localhost:8080/v1/chat/completions")
    mock_http = _patch_client(
        [
            httpx.ConnectError("boom", request=request),
            _openai_response("recovered"),
        ]
    )

    with (
        patch("pocket_agent.llm_client.httpx.Client", return_value=mock_http),
        patch("pocket_agent.llm_client.time.sleep") as sleep,
    ):
        assert client.generate("hi") == "recovered"

    sleep.assert_called_once_with(1.0)


def test_4xx_does_not_retry() -> None:
    client = LlamaCppClient(AgentConfig())
    mock_http = _patch_client([_openai_response("bad", status_code=400)])

    with (
        patch("pocket_agent.llm_client.httpx.Client", return_value=mock_http),
        patch("pocket_agent.llm_client.time.sleep") as sleep,
        pytest.raises(LLMConnectionError),
    ):
        client.generate("hi")

    assert mock_http.stream.call_count == 1
    sleep.assert_not_called()


def test_messages_are_single_turn_only() -> None:
    client = LlamaCppClient(AgentConfig())
    mock_http = _patch_client([_openai_response("x"), _openai_response("y")])

    with patch("pocket_agent.llm_client.httpx.Client", return_value=mock_http):
        client.generate("turn-one")
        client.generate("turn-two")

    for call in mock_http.stream.call_args_list:
        messages = call.kwargs["json"]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


def test_mock_llm_queue_and_prompts(mock_llm_factory: type) -> None:
    llm = mock_llm_factory(["answer", 0, {"city": "上海"}])
    assert llm.generate("p1") == "answer"
    assert llm.generate_int("p2", -1, 3) == 0
    assert llm.generate_json("p3", {"type": "object"}) == {"city": "上海"}
    assert llm.prompts == ["p1", "p2", "p3"]
    with pytest.raises(AssertionError, match="exhausted"):
        llm.generate("p4")


def test_local_client_ignores_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP_PROXY 环境变量不得劫持本地 llama.cpp 请求（trust_env=False）。"""
    monkeypatch.setenv("HTTP_PROXY", "http://evil-proxy:9999")
    monkeypatch.setenv("HTTPS_PROXY", "http://evil-proxy:9999")
    monkeypatch.setenv("ALL_PROXY", "http://evil-proxy:9999")

    config = AgentConfig(base_url="http://localhost:8000/v1")
    client = LlamaCppClient(config)

    captured: dict[str, object] = {}

    real_init = httpx.Client.__init__

    def _spy_init(self: httpx.Client, **kwargs: object) -> None:
        captured["trust_env"] = kwargs.get("trust_env")
        real_init(self, **kwargs)

    mock_http = MagicMock()
    mock_http.stream.side_effect = [_as_stream_cm(_openai_response("ok"))]
    mock_http.__enter__.return_value = mock_http
    mock_http.__exit__.return_value = None

    with patch("pocket_agent.llm_client.httpx.Client") as mock_cls:
        mock_cls.return_value = mock_http
        mock_cls.side_effect = lambda **kw: (
            captured.__setitem__("kwargs", kw),
            mock_http,
        )[-1]
        result = client.generate("hi")

    assert result == "ok"
    # trust_env=False → httpx 不读取 HTTP_PROXY 环境变量
    kwargs = captured.get("kwargs", {})
    assert kwargs.get("trust_env") is False


def test_resolve_trust_env_localhost_intranet_public() -> None:
    assert resolve_trust_env("http://localhost:8000/v1") is False
    assert resolve_trust_env("http://127.0.0.1:8000/v1", True) is False
    assert resolve_trust_env("http://10.0.0.8:11434/v1") is False
    assert resolve_trust_env("http://192.168.1.10:8000/v1", True) is False
    assert resolve_trust_env("https://api.deepseek.com/v1") is True
    assert resolve_trust_env("https://api.openai.com/v1", False) is False


def test_thinking_off_sends_enable_false() -> None:
    client = OpenAICompatibleClient(AgentConfig(), thinking="off")
    mock_http = _patch_client([_openai_response("<think>隐</think>正文")])
    with patch("pocket_agent.llm_client.httpx.Client", return_value=mock_http):
        assert client.generate("问") == "正文"
    payload = mock_http.stream.call_args.kwargs["json"]
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert client.last_thinking == ""


def test_thinking_strip_omits_kwargs_and_drops_reasoning() -> None:
    client = OpenAICompatibleClient(AgentConfig(), thinking="strip")
    response = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>推理</think>答案",
                        "reasoning_content": "云端思维",
                    }
                }
            ]
        },
        request=httpx.Request("POST", "http://localhost:8000/v1/chat/completions"),
    )
    mock_http = _patch_client([response])
    with patch("pocket_agent.llm_client.httpx.Client", return_value=mock_http):
        assert client.generate("问") == "答案"
    payload = mock_http.stream.call_args.kwargs["json"]
    assert "chat_template_kwargs" not in payload
    assert client.last_thinking == ""


def test_thinking_off_degrades_to_strip_when_reasoning_leaks(caplog: pytest.LogCaptureFixture) -> None:
    llm_client_mod._thinking_off_warned = False
    client = OpenAICompatibleClient(AgentConfig(), thinking="off")
    response = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "最终答案",
                        "reasoning_content": "这是不该泄漏的思维链",
                    }
                }
            ]
        },
        request=httpx.Request("POST", "http://localhost:8000/v1/chat/completions"),
    )
    follow = _openai_response("第二次")
    mock_http = _patch_client([response, follow])
    with (
        caplog.at_level(logging.WARNING, logger="pocket_agent.llm_client"),
        patch("pocket_agent.llm_client.httpx.Client", return_value=mock_http),
    ):
        first = client.generate("问")
        second = client.generate("再问")
    assert first == "最终答案"
    assert "思维链" not in first
    assert second == "第二次"
    assert client._thinking == "strip"
    assert client.last_thinking == ""
    warnings = [r for r in caplog.records if "enable_thinking" in r.message]
    assert len(warnings) == 1
    assert "已按 strip 处理" in warnings[0].message
    second_payload = mock_http.stream.call_args_list[1].kwargs["json"]
    assert "chat_template_kwargs" not in second_payload


def test_parse_usage_from_non_stream_response() -> None:
    usage = {
        "prompt_tokens": 24,
        "completion_tokens": 165,
        "total_tokens": 189,
        "prompt_tokens_details": {"cached_tokens": 2},
    }
    client = OpenAICompatibleClient(AgentConfig(), thinking="strip")
    mock_http = _patch_client([_openai_response("正文", usage=usage)])
    with patch("pocket_agent.llm_client.httpx.Client", return_value=mock_http):
        assert client.generate("问") == "正文"
    assert client.last_usage == {
        "prompt_tokens": 24,
        "completion_tokens": 165,
        "total_tokens": 189,
        "cached_tokens": 2,
    }


def test_parse_usage_missing_field_is_none() -> None:
    assert parse_usage(None) is None
    assert parse_usage({}) is None
    assert parse_usage("nope") is None
    client = OpenAICompatibleClient(AgentConfig(), thinking="strip")
    mock_http = _patch_client([_openai_response("正文")])
    with patch("pocket_agent.llm_client.httpx.Client", return_value=mock_http):
        assert client.generate("问") == "正文"
    assert client.last_usage is None


def test_parse_usage_from_stream_chunk() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _sse_bytes(
            [
                'data: {"choices":[{"delta":{"content":"好"}}]}',
                (
                    'data: {"choices":[{"delta":{}}],'
                    '"usage":{"prompt_tokens":10,"completion_tokens":2,'
                    '"total_tokens":12}}'
                ),
                "data: [DONE]",
            ]
        )
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    client = OpenAICompatibleClient(AgentConfig(), thinking="strip")
    deltas: list[str] = []
    with patch(
        "pocket_agent.llm_client.httpx.Client",
        side_effect=_transport_for(handler),
    ):
        text = client.generate_stream("问", deltas.append)
    assert text == "好"
    assert client.last_usage == {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
    }


def test_thinking_keep_captures_and_enables() -> None:
    client = OpenAICompatibleClient(AgentConfig(), thinking="keep")
    mock_http = _patch_client(
        [_openai_response("<think>推理过程</think>最终答案")]
    )
    with patch("pocket_agent.llm_client.httpx.Client", return_value=mock_http):
        assert client.generate("问") == "最终答案"
    payload = mock_http.stream.call_args.kwargs["json"]
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert "推理过程" in client.last_thinking


def test_public_host_respects_requested_trust_env() -> None:
    config = AgentConfig(base_url="https://api.deepseek.com/v1")
    captured: dict[str, object] = {}
    mock_http = MagicMock()
    mock_http.stream.side_effect = [_as_stream_cm(_openai_response("ok"))]
    mock_http.__enter__.return_value = mock_http
    mock_http.__exit__.return_value = None
    client = OpenAICompatibleClient(config, trust_env=True, thinking="strip")
    with patch("pocket_agent.llm_client.httpx.Client") as mock_cls:
        mock_cls.side_effect = lambda **kw: (
            captured.__setitem__("kwargs", kw),
            mock_http,
        )[-1]
        assert client.generate("hi") == "ok"
    assert captured["kwargs"].get("trust_env") is True  # type: ignore[union-attr]


def test_abort_current_interrupts_in_flight_stream() -> None:
    started = threading.Event()
    released = threading.Event()

    class HangResponse:
        status_code = 200
        text = ""

        def read(self) -> bytes:
            started.set()
            if not released.wait(timeout=5):
                raise TimeoutError("abort_current did not close the stream")
            raise httpx.ReadError(
                "closed",
                request=httpx.Request("POST", "http://localhost:8000/v1/chat/completions"),
            )

        def close(self) -> None:
            released.set()

    hang = HangResponse()
    mock_http = MagicMock()
    cm = MagicMock()
    cm.__enter__.return_value = hang
    cm.__exit__.return_value = None
    mock_http.stream.return_value = cm
    mock_http.__enter__.return_value = mock_http
    mock_http.__exit__.return_value = None

    client = OpenAICompatibleClient(AgentConfig())
    caught: list[BaseException] = []

    def _worker() -> None:
        try:
            client.generate("长文")
        except BaseException as exc:  # noqa: BLE001 — 测试捕获线程异常
            caught.append(exc)

    with patch("pocket_agent.llm_client.httpx.Client", return_value=mock_http):
        thread = threading.Thread(target=_worker)
        thread.start()
        assert started.wait(timeout=2)
        client.abort_current()
        thread.join(timeout=3)
    assert not thread.is_alive()
    assert caught
    assert isinstance(caught[0], LLMAbortedError)


def _sse_bytes(lines: list[str]) -> bytes:
    return "".join(f"{line}\n" for line in lines).encode("utf-8")


def _transport_for(handler: object) -> object:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    real = httpx.Client

    def factory(*args: object, **kwargs: object) -> httpx.Client:
        kwargs["transport"] = transport
        return real(*args, **kwargs)  # type: ignore[arg-type]

    return factory


def test_generate_stream_strips_think_and_reasoning() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        body = _sse_bytes(
            [
                'data: {"choices":[{"delta":{"content":"你"}}]}',
                'data: {"choices":[{"delta":{"reasoning_content":"先想"}}]}',
                'data: {"choices":[{"delta":{"content":"<think>隐"}}]}',
                'data: {"choices":[{"delta":{"content":"藏</think>好"}}]}',
                'data: {"choices":[{"delta":{"content":"的"}}]}',
                "data: [DONE]",
            ]
        )
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    client = OpenAICompatibleClient(AgentConfig(), thinking="keep")
    deltas: list[str] = []
    with patch(
        "pocket_agent.llm_client.httpx.Client",
        side_effect=_transport_for(handler),
    ):
        text = client.generate_stream("问", deltas.append)
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload.get("stream") is True
    assert "response_format" not in payload
    assert deltas == ["你", "好", "的"]
    assert text == "你好的"
    assert "先想" in client.last_thinking
    assert "隐藏" in client.last_thinking


def test_generate_stream_abort_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        def _gen() -> object:
            yield b'data: {"choices":[{"delta":{"content":"a"}}]}\n'
            for _ in range(40):
                yield b'data: {"choices":[{"delta":{"content":"b"}}]}\n'
            yield b"data: [DONE]\n"

        return httpx.Response(
            200,
            content=_gen(),
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    client = OpenAICompatibleClient(AgentConfig(), thinking="keep")

    def on_delta(_chunk: str) -> None:
        client.abort_current()

    with (
        patch(
            "pocket_agent.llm_client.httpx.Client",
            side_effect=_transport_for(handler),
        ),
        pytest.raises(LLMAbortedError),
    ):
        client.generate_stream("问", on_delta)


def test_generate_stream_falls_back_when_stream_rejected() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                400,
                text='{"error":"stream not supported"}',
                request=request,
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "降级回答"}}
                ]
            },
            request=request,
        )

    client = OpenAICompatibleClient(AgentConfig())
    deltas: list[str] = []
    with patch(
        "pocket_agent.llm_client.httpx.Client",
        side_effect=_transport_for(handler),
    ):
        text = client.generate_stream("问", deltas.append)
    assert text == "降级回答"
    assert deltas == ["降级回答"]
