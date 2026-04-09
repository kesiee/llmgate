"""Tests for custom exceptions."""

import pytest

from llmgate.exceptions import (
    AuthError,
    EmbeddingsNotSupported,
    LLMGateError,
    ModelNotFoundError,
    ProviderAPIError,
    RateLimitError,
)


class TestExceptionHierarchy:
    def test_all_inherit_from_base(self):
        assert issubclass(AuthError, LLMGateError)
        assert issubclass(RateLimitError, LLMGateError)
        assert issubclass(ProviderAPIError, LLMGateError)
        assert issubclass(ModelNotFoundError, LLMGateError)
        assert issubclass(EmbeddingsNotSupported, LLMGateError)

    def test_base_inherits_from_exception(self):
        assert issubclass(LLMGateError, Exception)


class TestAuthError:
    def test_message(self):
        e = AuthError("openai")
        assert "openai" in str(e)
        assert e.provider == "openai"

    def test_custom_message(self):
        e = AuthError("openai", "bad key")
        assert str(e) == "bad key"


class TestRateLimitError:
    def test_message(self):
        e = RateLimitError("groq")
        assert "groq" in str(e)
        assert e.provider == "groq"


class TestProviderAPIError:
    def test_message(self):
        e = ProviderAPIError("anthropic", 500)
        assert "500" in str(e)
        assert e.provider == "anthropic"
        assert e.status_code == 500


class TestModelNotFoundError:
    def test_message(self):
        e = ModelNotFoundError("gpt-99")
        assert "gpt-99" in str(e)
        assert e.model == "gpt-99"


class TestEmbeddingsNotSupported:
    def test_message(self):
        e = EmbeddingsNotSupported("anthropic")
        assert "anthropic" in str(e)
        assert e.provider == "anthropic"


class TestRaiseForStatus:
    def test_401_raises_auth_error(self):
        import httpx
        from llmgate.providers.openai import _raise_for_status

        request = httpx.Request("POST", "https://fake")
        resp = httpx.Response(401, text="Unauthorized", request=request)
        with pytest.raises(AuthError):
            _raise_for_status(resp, "openai")

    def test_429_raises_rate_limit(self):
        import httpx
        from llmgate.providers.openai import _raise_for_status

        request = httpx.Request("POST", "https://fake")
        resp = httpx.Response(429, text="Too many requests", request=request)
        with pytest.raises(RateLimitError):
            _raise_for_status(resp, "openai")

    def test_500_raises_provider_api_error(self):
        import httpx
        from llmgate.providers.openai import _raise_for_status

        request = httpx.Request("POST", "https://fake")
        resp = httpx.Response(500, text="Internal error", request=request)
        with pytest.raises(ProviderAPIError) as exc_info:
            _raise_for_status(resp, "openai")
        assert exc_info.value.status_code == 500

    def test_200_no_error(self):
        import httpx
        from llmgate.providers.openai import _raise_for_status

        request = httpx.Request("POST", "https://fake")
        resp = httpx.Response(200, text="OK", request=request)
        _raise_for_status(resp, "openai")  # should not raise
