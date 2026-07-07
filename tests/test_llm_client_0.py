import json
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

from src.llm_client_0 import (
    _make_cache_key,
    cached_request,
)


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    """Redirect CACHE_DIR to a per-test temp folder so tests don't share state
    and don't touch the real on-disk cache."""
    monkeypatch.setattr("src.llm_client_0.CACHE_DIR", tmp_path)
    return tmp_path


def test_cache_key_is_deterministic():
    k1 = _make_cache_key("v1", "model-x", "completion", {"a": 1, "b": 2})
    k2 = _make_cache_key("v1", "model-x", "completion", {"b": 2, "a": 1})
    assert k1 == k2


def test_cache_key_changes_with_prompt_version():
    k1 = _make_cache_key("v1", "model-x", "completion", {"a": 1})
    k2 = _make_cache_key("v2", "model-x", "completion", {"a": 1})
    assert k1 != k2


def test_cache_key_changes_with_endpoint():
    params = {"model": "m", "input": "hi"}
    k_completion = _make_cache_key("v1", "m", "completion", params)
    k_embedding = _make_cache_key("v1", "m", "embedding", params)
    assert k_completion != k_embedding


def test_cache_key_changes_with_model():
    params = {"messages": [{"role": "user", "content": "hi"}]}
    k1 = _make_cache_key("v1", "model-a", "completion", params)
    k2 = _make_cache_key("v1", "model-b", "completion", params)
    assert k1 != k2


def test_first_call_hits_network_second_call_does_not(tmp_cache):
    fake = MagicMock()
    fake.json.return_value = {"choices": [{"text": "meow"}]}

    with patch("src.llm_client_0.requests.post", return_value=fake) as mock_post:
        r1 = cached_request(
            api_key="sk-test",
            url="https://example.com/v1/chat",
            endpoint="completion",
            model="model-x",
            params={"model": "model-x", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert mock_post.call_count == 1, "first call must hit the network"

        r2 = cached_request(
            api_key="sk-test",
            url="https://example.com/v1/chat",
            endpoint="completion",
            model="model-x",
            params={"model": "model-x", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert mock_post.call_count == 1, "second call must be served from cache (no network)"

    assert r1 == r2 == {"choices": [{"text": "meow"}]}


def test_cache_entry_records_required_fields(tmp_cache):
    fake = MagicMock()
    fake.json.return_value = {"ok": True}

    with patch("src.llm_client_0.requests.post", return_value=fake):
        cached_request(
            api_key="sk-test",
            url="https://example.com",
            endpoint="completion",
            model="model-x",
            params={"model": "model-x", "messages": []},
            prompt_version="v3",
        )

    files = list(tmp_cache.glob("*.json"))
    assert len(files) == 1, "exactly one cache file should have been written"

    entry = json.loads(files[0].read_text())

    for field in ("prompt_version", "model", "params", "raw_response", "timestamp"):
        assert field in entry, f"cache entry missing required field: {field}"

    assert entry["prompt_version"] == "v3"
    assert entry["model"] == "model-x"
    assert entry["endpoint"] == "completion"
    assert entry["raw_response"] == {"ok": True}

    # Timestamp should be a parseable ISO-8601 string.
    datetime.fromisoformat(entry["timestamp"])


def test_different_params_produce_different_cache_entries(tmp_cache):
    fake = MagicMock()
    fake.json.side_effect = [{"a": 1}, {"a": 2}]

    with patch("src.llm_client_0.requests.post", return_value=fake):
        r1 = cached_request(
            api_key="sk-test",
            url="https://example.com",
            endpoint="completion",
            model="m",
            params={"messages": [{"role": "user", "content": "hi"}]},
        )
        r2 = cached_request(
            api_key="sk-test",
            url="https://example.com",
            endpoint="completion",
            model="m",
            params={"messages": [{"role": "user", "content": "bye"}]},
        )

    assert r1 != r2
    assert len(list(tmp_cache.glob("*.json"))) == 2


def test_bumping_prompt_version_invalidates_cache(tmp_cache):
    fake = MagicMock()
    fake.json.side_effect = [{"v": 1}, {"v": 2}]

    with patch("src.llm_client_0.requests.post", return_value=fake) as mock_post:
        cached_request(
            api_key="sk-test",
            url="https://example.com",
            endpoint="completion",
            model="m",
            params={"messages": [{"role": "user", "content": "hi"}]},
            prompt_version="v1",
        )
        cached_request(
            api_key="sk-test",
            url="https://example.com",
            endpoint="completion",
            model="m",
            params={"messages": [{"role": "user", "content": "hi"}]},
            prompt_version="v2",
        )

    assert mock_post.call_count == 2, "bumping prompt_version must miss cache and hit network again"
    assert len(list(tmp_cache.glob("*.json"))) == 2
