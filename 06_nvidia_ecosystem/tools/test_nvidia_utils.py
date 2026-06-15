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


def test_nim_client_raises_without_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    pytest.importorskip("openai")
    with pytest.raises(RuntimeError):
        nim_client()
