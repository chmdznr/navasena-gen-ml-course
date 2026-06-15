import re
import pytest
from nvidia_utils import detect_pii_id, mask_pii_id, summarize_activated_rails, nim_client


def test_detect_nik_16_digits():
    found = detect_pii_id("KTP saya 3204010101900001 ya")
    assert any(f["type"] == "NIK" and f["value"] == "3204010101900001" for f in found)


def test_detect_phone_plus62_and_08():
    a = detect_pii_id("hubungi +6281234567890")
    b = detect_pii_id("WA 081234567890")
    assert any(f["type"] == "PHONE" for f in a)
    assert any(f["type"] == "PHONE" for f in b)


def test_detect_account_number():
    found = detect_pii_id("transfer ke rekening 1234567890")
    assert any(f["type"] == "ACCOUNT" for f in found)


def test_nik_not_confused_with_account():
    # 16 digits -> NIK wins over ACCOUNT (longest/most-specific first)
    found = detect_pii_id("3204010101900001")
    types = {f["type"] for f in found}
    assert "NIK" in types and "ACCOUNT" not in types


def test_mask_replaces_with_placeholders():
    masked = mask_pii_id("NIK 3204010101900001 HP +6281234567890")
    assert "3204010101900001" not in masked
    assert "+6281234567890" not in masked
    assert "[NIK]" in masked and "[PHONE]" in masked


def test_mask_is_idempotent_on_clean_text():
    clean = "Berapa harga produk ini?"
    assert mask_pii_id(clean) == clean


def test_summarize_activated_rails_from_loglike():
    class _Rail:
        def __init__(self, name, stop):
            self.type = name
            self.stop = stop

    class _Log:
        activated_rails = [_Rail("self check input", True), _Rail("self check output", False)]

    class _Resp:
        log = _Log()

    out = summarize_activated_rails(_Resp())
    assert out == ["self check input (BLOCKED)", "self check output (passed)"]


def test_summarize_handles_no_log():
    assert summarize_activated_rails(object()) == []


def test_summarize_prefers_name_over_type():
    # Real NeMo Guardrails ActivatedRail has BOTH .name ("self check input") and
    # .type ("input"); the readable summary should use the more specific .name.
    class _Rail:
        name = "self check input"
        type = "input"
        stop = True

    class _Log:
        activated_rails = [_Rail()]

    class _Resp:
        log = _Log()

    assert summarize_activated_rails(_Resp()) == ["self check input (BLOCKED)"]


def test_nim_client_raises_without_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    pytest.importorskip("openai")
    with pytest.raises(RuntimeError):
        nim_client()


class _FakeNIM:
    """Minimal stand-in for an OpenAI client to test nim_chat hermetically."""
    def __init__(self, content):
        self.captured = {}
        outer = self

        class _Completions:
            def create(self, **kw):
                outer.captured = kw
                msg = type("M", (), {"content": content})()
                choice = type("C", (), {"message": msg})()
                return type("R", (), {"choices": [choice]})()

        self.chat = type("Chat", (), {"completions": _Completions()})()


def test_nim_chat_disables_reasoning_and_strips():
    from nvidia_utils import nim_chat, NEMOTRON_NO_THINK
    fake = _FakeNIM("  Halo  ")
    out = nim_chat(fake, "nvidia/nemotron-3-nano-30b-a3b", [{"role": "user", "content": "hi"}])
    assert out == "Halo"
    assert fake.captured["extra_body"] == NEMOTRON_NO_THINK
    assert fake.captured["model"] == "nvidia/nemotron-3-nano-30b-a3b"


def test_nim_chat_handles_none_content():
    from nvidia_utils import nim_chat
    assert nim_chat(_FakeNIM(None), "m", [{"role": "user", "content": "x"}]) == ""


def test_nim_chat_no_think_false_omits_extra_body():
    from nvidia_utils import nim_chat
    fake = _FakeNIM("ok")
    nim_chat(fake, "m", [{"role": "user", "content": "x"}], no_think=False)
    assert "extra_body" not in fake.captured
