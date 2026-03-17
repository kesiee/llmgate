"""Tests for llmgate config loading."""

import os
from unittest.mock import patch

import pytest

from llmgate.config import load_config


class TestFlatConfig:
    def test_flat_load(self, tmp_path):
        cfg = tmp_path / "llmgate.yaml"
        cfg.write_text("provider: openai\nmodel: gpt-4o\ntemperature: 0.5\n")
        result = load_config(str(cfg))
        assert result["provider"] == "openai"
        assert result["model"] == "gpt-4o"
        assert result["temperature"] == 0.5

    def test_env_interpolation(self, tmp_path):
        cfg = tmp_path / "llmgate.yaml"
        cfg.write_text("provider: openai\nmodel: gpt-4o\napi_key: ${TEST_LLM_KEY}\n")
        with patch.dict(os.environ, {"TEST_LLM_KEY": "sk-test123"}):
            result = load_config(str(cfg))
        assert result["api_key"] == "sk-test123"

    def test_missing_env_var_resolves_empty(self, tmp_path):
        cfg = tmp_path / "llmgate.yaml"
        cfg.write_text("provider: openai\nmodel: gpt-4o\napi_key: ${NONEXISTENT_KEY_XYZ}\n")
        result = load_config(str(cfg))
        assert result["api_key"] == ""


class TestProfileConfig:
    def test_profile_resolution(self, tmp_path):
        cfg = tmp_path / "llmgate.yaml"
        cfg.write_text(
            "active_profile: fast\n"
            "defaults:\n  temperature: 0.7\n  max_tokens: 1024\n"
            "profiles:\n  fast:\n    provider: groq\n    model: llama-3.1-8b-instant\n"
        )
        result = load_config(str(cfg))
        assert result["provider"] == "groq"
        assert result["model"] == "llama-3.1-8b-instant"
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 1024

    def test_profile_overrides_defaults(self, tmp_path):
        cfg = tmp_path / "llmgate.yaml"
        cfg.write_text(
            "active_profile: custom\n"
            "defaults:\n  temperature: 0.7\n"
            "profiles:\n  custom:\n    provider: openai\n    model: gpt-4o\n    temperature: 0.2\n"
        )
        result = load_config(str(cfg))
        assert result["temperature"] == 0.2

    def test_explicit_profile_overrides_active(self, tmp_path):
        cfg = tmp_path / "llmgate.yaml"
        cfg.write_text(
            "active_profile: a\n"
            "profiles:\n"
            "  a:\n    provider: openai\n    model: gpt-4o\n"
            "  b:\n    provider: groq\n    model: llama\n"
        )
        result = load_config(str(cfg), profile="b")
        assert result["provider"] == "groq"

    def test_nested_env_interpolation(self, tmp_path):
        cfg = tmp_path / "llmgate.yaml"
        cfg.write_text(
            "provider: azure_openai\nmodel: gpt-4o\n"
            "nested:\n  deep:\n    value: ${TEST_NESTED_VAL}\n"
        )
        with patch.dict(os.environ, {"TEST_NESTED_VAL": "resolved"}):
            result = load_config(str(cfg))
        assert result["nested"]["deep"]["value"] == "resolved"


class TestErrors:
    def test_missing_provider(self, tmp_path):
        cfg = tmp_path / "llmgate.yaml"
        cfg.write_text("model: gpt-4o\n")
        with pytest.raises(ValueError, match="provider"):
            load_config(str(cfg))

    def test_unknown_profile(self, tmp_path):
        cfg = tmp_path / "llmgate.yaml"
        cfg.write_text(
            "active_profile: missing\n"
            "profiles:\n  a:\n    provider: openai\n    model: gpt-4o\n"
        )
        with pytest.raises(ValueError, match="missing"):
            load_config(str(cfg))

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/llmgate.yaml")
