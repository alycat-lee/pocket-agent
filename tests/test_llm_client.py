from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from pocket_agent.config import AgentConfig
from pocket_agent.llm_client import LlamaCppClient, LLMConnectionError, LLMFormatError


def _openai_response(content: str, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        json={
            "choices": [
                {"message": {"role": "assistant", "content": content}},
            ]
        },
        request=httpx.Request("POST", "http://localhost:8080/v1/chat/completions"),
    )


def _patch_client(post_side_effect: object) -> MagicMock:
    mock_client = MagicMock()
    mock_client.post.side_effect = post_side_effect
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
    payload = mock_http.post.call_args.kwargs["json"]
    assert payload["messages"] == [
        {"role": "system", "content": ""},
        {"role": "user", "content": "请回答"},
    ]
    assert payload["temperature"] == config.temperature
    assert payload["repeat_penalty"] == config.repeat_penalty
    assert payload["top_p"] == config.top_p
    assert payload["max_tokens"] == config.max_tokens
    assert "response_format" not in payload


def test_generate_falls_back_to_reasoning_content() -> None:
    client = LlamaCppClient(AgentConfig())
    response = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "北京是中国的首都。",
                    }
                }
            ]
        },
        request=httpx.Request("POST", "http://localhost:8000/v1/chat/completions"),
    )
    mock_http = _patch_client([response])
    with patch("pocket_agent.llm_client.httpx.Client", return_value=mock_http):
        assert client.generate("介绍北京") == "北京是中国的首都。"


def test_generate_int_from_json_schema_value() -> None:
    client = LlamaCppClient(AgentConfig())
    mock_http = _patch_client([_openai_response('{"value": 2}')])

    with patch("pocket_agent.llm_client.httpx.Client", return_value=mock_http):
        result = client.generate_int("选工具", minimum=-1, maximum=3)

    assert result == 2
    payload = mock_http.post.call_args.kwargs["json"]
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
    rf = mock_http.post.call_args.kwargs["json"]["response_format"]
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

    assert mock_http.post.call_count == 3
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

    assert mock_http.post.call_count == 3


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

    assert mock_http.post.call_count == 1
    sleep.assert_not_called()


def test_messages_are_single_turn_only() -> None:
    client = LlamaCppClient(AgentConfig())
    mock_http = _patch_client([_openai_response("x"), _openai_response("y")])

    with patch("pocket_agent.llm_client.httpx.Client", return_value=mock_http):
        client.generate("turn-one")
        client.generate("turn-two")

    for call in mock_http.post.call_args_list:
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
