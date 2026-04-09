"""Tests for LLMGate main gateway class."""

from unittest.mock import MagicMock, patch

import pytest

from llmgate.gate import LLMGate, _load_provider_class
from llmgate.providers.openai import OpenAIProvider
from llmgate.response import LLMResponse


@pytest.fixture()
def simple_config(tmp_path):
    cfg = tmp_path / "llmgate.yaml"
    cfg.write_text("provider: openai\nmodel: gpt-4o\napi_key: sk-test\n")
    return str(cfg)


@pytest.fixture()
def multi_config(tmp_path):
    cfg = tmp_path / "llmgate.yaml"
    cfg.write_text(
        "active_profile: fast\n"
        "defaults:\n  temperature: 0.7\n"
        "profiles:\n"
        "  fast:\n    provider: groq\n    model: llama-3.1-8b-instant\n    api_key: gsk-test\n"
        "  smart:\n    provider: openai\n    model: gpt-4o\n    api_key: sk-test\n"
    )
    return str(cfg)


def _fake_response(**overrides):
    defaults = dict(
        text="hello",
        model="gpt-4o",
        provider="openai",
        tokens_used=10,
        finish_reason="stop",
        raw={},
    )
    defaults.update(overrides)
    return LLMResponse(**defaults)


class TestLoadProviderClass:
    def test_known_provider(self):
        cls = _load_provider_class("openai")
        assert cls is OpenAIProvider

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider 'nope'"):
            _load_provider_class("nope")

    def test_all_registered_providers_importable(self):
        from llmgate.gate import PROVIDER_REGISTRY

        for name in PROVIDER_REGISTRY:
            cls = _load_provider_class(name)
            assert callable(cls)


class TestLLMGateInit:
    def test_creates_correct_provider(self, simple_config):
        gate = LLMGate(config_path=simple_config)
        assert gate.provider_name == "openai"
        assert gate.model == "gpt-4o"
        assert isinstance(gate._provider, OpenAIProvider)

    def test_config_property(self, simple_config):
        gate = LLMGate(config_path=simple_config)
        cfg = gate.config
        assert cfg["provider"] == "openai"
        assert cfg["model"] == "gpt-4o"
        # Returns a copy
        cfg["provider"] = "changed"
        assert gate.config["provider"] == "openai"

    def test_profile_selection(self, multi_config):
        gate = LLMGate(config_path=multi_config, profile="smart")
        assert gate.provider_name == "openai"
        assert gate.model == "gpt-4o"


class TestSwitch:
    def test_switch_changes_provider(self, multi_config):
        gate = LLMGate(config_path=multi_config)
        assert gate.provider_name == "groq"

        gate.switch("smart")
        assert gate.provider_name == "openai"
        assert gate.model == "gpt-4o"

    def test_switch_to_invalid_profile_raises(self, multi_config):
        gate = LLMGate(config_path=multi_config)
        with pytest.raises(ValueError, match="nope"):
            gate.switch("nope")


class TestChat:
    def test_chat_delegates_to_provider(self, simple_config):
        gate = LLMGate(config_path=simple_config)
        fake = _fake_response()
        gate._provider.send = MagicMock(return_value=fake)

        result = gate.chat("hi")

        assert result is fake
        gate._provider.send.assert_called_once_with(
            [{"role": "user", "content": "hi"}]
        )

    def test_chat_passes_kwargs(self, simple_config):
        gate = LLMGate(config_path=simple_config)
        fake = _fake_response()
        gate._provider.send = MagicMock(return_value=fake)

        gate.chat("hi", temperature=0.5, max_tokens=100)

        gate._provider.send.assert_called_once_with(
            [{"role": "user", "content": "hi"}],
            temperature=0.5,
            max_tokens=100,
        )

    def test_chat_messages_passes_full_list(self, simple_config):
        gate = LLMGate(config_path=simple_config)
        fake = _fake_response()
        gate._provider.send = MagicMock(return_value=fake)

        msgs = [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ]
        result = gate.chat_messages(msgs)

        assert result is fake
        gate._provider.send.assert_called_once_with(msgs)


class TestStream:
    def test_stream_delegates_to_provider(self, simple_config):
        gate = LLMGate(config_path=simple_config)
        gate._provider.stream = MagicMock(return_value=iter(["hel", "lo"]))

        chunks = list(gate.stream("hi"))

        assert chunks == ["hel", "lo"]
        gate._provider.stream.assert_called_once_with(
            [{"role": "user", "content": "hi"}]
        )

    def test_stream_messages_passes_full_list(self, simple_config):
        gate = LLMGate(config_path=simple_config)
        gate._provider.stream = MagicMock(return_value=iter(["ok"]))

        msgs = [{"role": "user", "content": "hi"}]
        chunks = list(gate.stream_messages(msgs))

        assert chunks == ["ok"]
        gate._provider.stream.assert_called_once_with(msgs)

    def test_stream_passes_kwargs(self, simple_config):
        gate = LLMGate(config_path=simple_config)
        gate._provider.stream = MagicMock(return_value=iter([]))

        list(gate.stream("hi", temperature=0))

        gate._provider.stream.assert_called_once_with(
            [{"role": "user", "content": "hi"}],
            temperature=0,
        )
