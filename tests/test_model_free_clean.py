"""Explicit model-free compatibility without silently bypassing rewrite requests."""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service" / "scripts"))
import server


@pytest.mark.parametrize("default_strategy", [None, "paraphrase@0.8,mlm@0.2"])
def test_layer_a_mode_preserves_words_without_calling_rewrite(monkeypatch, default_strategy):
    monkeypatch.setattr(server, "_TEXT_CLEAN_MODE", "layer-a")
    monkeypatch.setattr(server, "_DEFAULT_STRATEGY", default_strategy)
    rewrite = Mock(side_effect=AssertionError("Layer A must not load/call a rewrite backend"))
    monkeypatch.setattr(server, "_apply_layer_b", rewrite)
    result = server._clean_payload("Hello\u200b world. Exact 123!".encode(), "a.txt", {})
    assert base64.b64decode(result["cleaned"]) == b"Hello world. Exact 123!"
    assert result["report"]["layer_b"]["applied"] is False
    assert "statistical watermarks are not removed" in result["report"]["layer_b"]["reason"]
    rewrite.assert_not_called()


def test_layer_a_explicit_strategy_is_not_ignored(monkeypatch):
    monkeypatch.setattr(server, "_TEXT_CLEAN_MODE", "layer-a")
    rewrite = Mock(return_value=("rewritten", {"steps": []}))
    monkeypatch.setattr(server, "_apply_layer_b", rewrite)
    options = server._parse_clean_options({"strategy": "paraphrase@0.8"})
    result = server._clean_payload("Hello\u200b world.".encode(), "a.txt", options)
    rewrite.assert_called_once_with("Hello world.", "paraphrase@0.8", options)
    assert base64.b64decode(result["cleaned"]) == b"rewritten"


@pytest.mark.parametrize("mode", ["layer-a", "layer-b"])
def test_requested_rewrite_never_falls_back_when_backend_is_missing(monkeypatch, mode):
    monkeypatch.setattr(server, "_TEXT_CLEAN_MODE", mode)
    monkeypatch.delenv("WATERMARKS_REWRITE_BACKEND", raising=False)
    with pytest.raises(ValueError, match="needs an LLM rewrite backend"):
        server._clean_payload(b"Hello world.", "a.txt", {"strategy": "paraphrase@0.8"})


def test_layer_b_mode_still_applies_default_strategy(monkeypatch):
    monkeypatch.setattr(server, "_TEXT_CLEAN_MODE", "layer-b")
    monkeypatch.setattr(server, "_DEFAULT_STRATEGY", "paraphrase@0.8")
    rewrite = Mock(return_value=("rewritten", {"steps": []}))
    monkeypatch.setattr(server, "_apply_layer_b", rewrite)
    server._clean_payload(b"Hello world.", "a.txt", {})
    rewrite.assert_called_once_with("Hello world.", "paraphrase@0.8", {})


@pytest.mark.parametrize("mode", ["layer-a", "layer-b"])
def test_capabilities_advertise_text_clean_mode(monkeypatch, mode):
    monkeypatch.setattr(server, "_TEXT_CLEAN_MODE", mode)
    monkeypatch.setattr(server, "_tool_usable", lambda name: False)
    monkeypatch.setattr(server, "_ghostscript_usable", lambda: False)
    assert server.capabilities()["text_clean_mode"] == mode


@pytest.mark.parametrize(
    ("env_mode", "args", "expected"),
    [
        (None, [], "layer-b"),
        ("layer-a", [], "layer-a"),
        ("layer-a", ["--text-clean-mode", "layer-b"], "layer-b"),
        ("layer-b", ["--text-clean-mode", "layer-a"], "layer-a"),
    ],
)
def test_startup_mode_precedence_without_binding(monkeypatch, env_mode, args, expected):
    monkeypatch.setattr(sys, "argv", ["server.py", *args])
    monkeypatch.setattr(server, "_TEXT_CLEAN_MODE", "layer-b")
    monkeypatch.setattr(server, "_DEFAULT_STRATEGY", None)
    monkeypatch.setattr(server, "API_KEY", None)
    monkeypatch.setattr(server, "_load_default_strategy", lambda path: None)
    if env_mode is None:
        monkeypatch.delenv("WATERMARKS_TEXT_CLEAN_MODE", raising=False)
    else:
        monkeypatch.setenv("WATERMARKS_TEXT_CLEAN_MODE", env_mode)
    http_server = Mock()
    monkeypatch.setattr(server, "ThreadingHTTPServer", Mock(return_value=http_server))
    assert server.main() == 0
    assert expected == server._TEXT_CLEAN_MODE
    http_server.serve_forever.assert_called_once()


@pytest.mark.parametrize("args", [[], ["--text-clean-mode", "invalid"]])
def test_invalid_mode_rejected_before_bind(monkeypatch, args):
    monkeypatch.setenv("WATERMARKS_TEXT_CLEAN_MODE", "invalid")
    monkeypatch.setattr(sys, "argv", ["server.py", *args])
    bind = Mock(side_effect=AssertionError("must not bind"))
    monkeypatch.setattr(server, "ThreadingHTTPServer", bind)
    with pytest.raises(SystemExit) as exc:
        server.main()
    assert exc.value.code == 2
    bind.assert_not_called()
