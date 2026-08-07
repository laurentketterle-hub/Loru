"""Tests for Loru voice TTS backends (CI-compatible, no real hardware required)."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from loru.voice.tts import (
    EdgeTTSTTS,
    OfflineStubTTS,
    Pyttsx3TTS,
    TextToSpeech,
    get_default_tts,
)


# ---------------------------------------------------------------------------
# OfflineStubTTS
# ---------------------------------------------------------------------------

class TestOfflineStubTTS:
    def test_creates_wav_and_text_sidecar(self) -> None:
        tts = OfflineStubTTS()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "output.wav"
            result = tts.speak("hello world", out)
            assert result.suffix == ".wav"
            assert result.exists()
            assert result.stat().st_size > 44
            txt = out.with_suffix(".txt")
            assert txt.exists()
            assert txt.read_text() == "hello world"

    def test_backend_name(self) -> None:
        assert OfflineStubTTS().backend_name == "OfflineStubTTS"

    def test_short_text_produces_short_audio(self) -> None:
        tts = OfflineStubTTS()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "short.wav"
            result = tts.speak("a", out)
            import wave
            with wave.open(str(result), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / rate
                assert duration <= 3.5


# ---------------------------------------------------------------------------
# EdgeTTSTTS - mock tests
# ---------------------------------------------------------------------------

class TestEdgeTTSTTS:
    def test_default_voice(self) -> None:
        tts = EdgeTTSTTS()
        assert tts._voice == "en-US-AriaNeural"

    def test_custom_voice(self) -> None:
        tts = EdgeTTSTTS(voice="en-GB-SoniaNeural")
        assert tts._voice == "en-GB-SoniaNeural"

    def test_backend_name(self) -> None:
        assert EdgeTTSTTS().backend_name == "EdgeTTSTTS"

    def test_speak_falls_back_to_stub_when_edge_tts_missing(self) -> None:
        tts = EdgeTTSTTS()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "output.wav"
            with mock.patch.dict("sys.modules", {"edge_tts": None}):
                result = tts.speak("hello", out)
            assert result.suffix == ".wav"
            assert result.exists()
            assert result.stat().st_size > 44

    def test_speak_success_writes_mp3(self) -> None:
        tts = EdgeTTSTTS()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "output.wav"

            async def fake_save(self, path):
                Path(path).write_bytes(bytes([0xff, 0xfb]) + bytes(1024))

            with mock.patch("edge_tts.Communicate.save", fake_save):
                with mock.patch.dict("sys.modules", {"edge_tts": mock.MagicMock()}):
                    import edge_tts as et_mod
                    et_mod.Communicate = mock.MagicMock()
                    et_mod.Communicate.return_value.save = fake_save
                    result = tts.speak("hello world", out)

            assert result.suffix == ".mp3"
            assert result.exists()
            assert result.stat().st_size >= 100
            assert out.with_suffix(".txt").read_text() == "hello world"

    def test_speak_creates_parent_dirs(self) -> None:
        tts = EdgeTTSTTS()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sub" / "deep" / "out.wav"

            async def fake_save(self, path):
                Path(path).write_bytes(bytes([0xff, 0xfb]) + bytes(1024))

            with mock.patch("edge_tts.Communicate.save", fake_save):
                with mock.patch.dict("sys.modules", {"edge_tts": mock.MagicMock()}):
                    import edge_tts as et_mod
                    et_mod.Communicate = mock.MagicMock()
                    et_mod.Communicate.return_value.save = fake_save
                    result = tts.speak("deep", out)
            assert result.exists()
            assert result.parent.exists()


# ---------------------------------------------------------------------------
# Pyttsx3TTS - mock tests
# ---------------------------------------------------------------------------

class TestPyttsx3TTS:
    def test_backend_name(self) -> None:
        with mock.patch("pyttsx3.init"):
            tts = Pyttsx3TTS()
            assert tts.backend_name == "Pyttsx3TTS"


# ---------------------------------------------------------------------------
# get_default_tts() selection logic
# ---------------------------------------------------------------------------

class TestGetDefaultTTS:
    def teardown_method(self) -> None:
        os.environ.pop("LORU_TTS", None)

    def test_env_stub_returns_offline_stub(self) -> None:
        os.environ["LORU_TTS"] = "stub"
        tts = get_default_tts()
        assert isinstance(tts, OfflineStubTTS)

    def test_env_offline_returns_offline_stub(self) -> None:
        os.environ["LORU_TTS"] = "offline"
        tts = get_default_tts()
        assert isinstance(tts, OfflineStubTTS)

    def test_env_tone_returns_offline_stub(self) -> None:
        os.environ["LORU_TTS"] = "tone"
        tts = get_default_tts()
        assert isinstance(tts, OfflineStubTTS)

    def test_env_pyttsx3_returns_pyttsx3(self) -> None:
        os.environ["LORU_TTS"] = "pyttsx3"
        with mock.patch("pyttsx3.init"):
            tts = get_default_tts()
            assert isinstance(tts, Pyttsx3TTS)

    def test_env_native_returns_pyttsx3(self) -> None:
        os.environ["LORU_TTS"] = "native"
        with mock.patch("pyttsx3.init"):
            tts = get_default_tts()
            assert isinstance(tts, Pyttsx3TTS)

    def test_env_edge_returns_edge_tts(self) -> None:
        os.environ["LORU_TTS"] = "edge"
        tts = get_default_tts()
        assert isinstance(tts, EdgeTTSTTS)

    def test_env_auto_tries_pyttsx3_first(self) -> None:
        os.environ["LORU_TTS"] = "auto"
        with mock.patch("pyttsx3.init"):
            tts = get_default_tts()
            assert isinstance(tts, Pyttsx3TTS)

    def test_auto_falls_back_to_edge_when_pyttsx3_fails(self) -> None:
        os.environ["LORU_TTS"] = "auto"
        with mock.patch("pyttsx3.init", side_effect=ImportError("no pyttsx3")):
            tts = get_default_tts()
            assert isinstance(tts, EdgeTTSTTS)

    def test_auto_falls_back_to_stub_when_both_fail(self) -> None:
        os.environ["LORU_TTS"] = "auto"
        with mock.patch("pyttsx3.init", side_effect=ImportError("no pyttsx3")):
            with mock.patch("loru.voice.tts.EdgeTTSTTS.__init__", side_effect=ImportError("no edge-tts")):
                tts = get_default_tts()
                assert isinstance(tts, OfflineStubTTS)

    def test_env_pyttsx3_raises_on_failure(self) -> None:
        os.environ["LORU_TTS"] = "pyttsx3"
        with mock.patch("pyttsx3.init", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                get_default_tts()

    def test_env_edge_raises_on_failure(self) -> None:
        os.environ["LORU_TTS"] = "edge"
        with mock.patch("loru.voice.tts.EdgeTTSTTS.__init__", side_effect=ImportError("no edge-tts")):
            with pytest.raises(ImportError):
                get_default_tts()

    def test_default_no_env_is_auto(self) -> None:
        os.environ.pop("LORU_TTS", None)
        with mock.patch("pyttsx3.init"):
            tts = get_default_tts()
            assert isinstance(tts, Pyttsx3TTS)


# ---------------------------------------------------------------------------
# TextToSpeech base class
# ---------------------------------------------------------------------------

class TestTextToSpeechBase:
    def test_speak_raises_not_implemented(self) -> None:
        base = TextToSpeech()
        with pytest.raises(NotImplementedError):
            base.speak("test", Path("out.wav"))

    def test_backend_name_default(self) -> None:
        base = TextToSpeech()
        assert base.backend_name == "TextToSpeech"
