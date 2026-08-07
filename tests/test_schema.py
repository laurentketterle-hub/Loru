"""Tests for Pydantic schema validation (Issue #4)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

# Ensure src/ is on the path when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loru.data.schema import SampleRecord, FrameValidator


# -- Helpers ----------------------------------------------------------------

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"


def _load_json(rel_path: str) -> dict:
    """Load a JSON file from the samples directory as a dict."""
    return json.loads((SAMPLES_DIR / rel_path).read_text(encoding="utf-8"))


def _make_payload(
    gloss: str = "test",
    frames: list | None = None,
    fps: float = 30.0,
    language: str = "unknown",
) -> dict:
    return {
        "gloss": gloss,
        "frames": _valid_frames() if frames is None else frames,
        "fps": fps,
        "language": language,
    }


def _valid_frames() -> list[list[list[float]]]:
    """Return a small 3-D landmark-style frame sequence (2 frames, 3 landmarks)."""
    return [
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        [[1.0, 1.1, 1.2], [1.3, 1.4, 1.5], [1.6, 1.7, 1.8]],
    ]


def _valid_frames_2d() -> list[list[float]]:
    """Return a small 2-D frame sequence (3 frames, 4 features)."""
    return [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
    ]


# -- FrameValidator standalone ----------------------------------------------


class TestFrameValidator:
    def test_valid_2d_frames(self) -> None:
        fv = FrameValidator(frames=_valid_frames_2d())
        assert len(fv.frames) == 3

    def test_valid_3d_landmarks(self) -> None:
        """3-D landmark data is flattened and accepted."""
        fv = FrameValidator(frames=_valid_frames())
        # 2 frames × 3 landmarks = 6 rows after flattening
        assert len(fv.frames) == 6
        assert all(len(row) == 3 for row in fv.frames)

    def test_empty_frames(self) -> None:
        with pytest.raises(ValidationError, match="non-empty"):
            FrameValidator(frames=[])

    def test_non_list_frames(self) -> None:
        with pytest.raises(ValidationError, match="must be a list"):
            FrameValidator(frames="not_a_list")  # type: ignore[arg-type]

    def test_jagged_2d_rows(self) -> None:
        with pytest.raises(ValidationError, match="has 2 values, expected 3"):
            FrameValidator(frames=[[1.0, 2.0, 3.0], [4.0, 5.0]])

    def test_jagged_3d_landmarks(self) -> None:
        with pytest.raises(ValidationError, match="has 2 values, expected 3"):
            FrameValidator(
                frames=[
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[7.0, 8.0, 9.0], [10.0, 11.0]],  # second landmark missing z
                ]
            )

    def test_non_numeric_2d(self) -> None:
        with pytest.raises(ValidationError, match="is not numeric"):
            FrameValidator(frames=[[1.0, "bad"], [3.0, 4.0]])

    def test_non_numeric_3d(self) -> None:
        with pytest.raises(ValidationError, match="is not numeric"):
            FrameValidator(
                frames=[
                    [[1.0, 2.0, 3.0]],
                    [[4.0, "nope", 6.0]],
                ]
            )


# -- SampleRecord validation ------------------------------------------------


class TestSampleRecord:
    def test_valid_record(self) -> None:
        rec = SampleRecord(**_make_payload())
        assert rec.gloss == "test"
        assert rec.fps == 30.0
        assert rec.language == "unknown"
        assert len(rec.frames) == 6  # 2 frames × 3 landmarks

    def test_missing_gloss(self) -> None:
        with pytest.raises(ValidationError):
            SampleRecord.model_validate({"frames": _valid_frames()})

    def test_empty_gloss(self) -> None:
        with pytest.raises(ValidationError, match="at least 1 character"):
            SampleRecord(**_make_payload(gloss=""))

    def test_missing_frames(self) -> None:
        with pytest.raises(ValidationError):
            SampleRecord.model_validate({"gloss": "hello"})

    def test_empty_frames(self) -> None:
        with pytest.raises(ValidationError, match="non-empty"):
            SampleRecord(**_make_payload(frames=[]))

    def test_jagged_frames(self) -> None:
        with pytest.raises(ValidationError, match="has 2 values, expected"):
            SampleRecord(
                **_make_payload(
                    frames=[
                        [[1.0, 2.0, 3.0]],
                        [[4.0, 5.0]],  # missing last coord
                    ]
                )
            )

    def test_non_numeric_frames(self) -> None:
        with pytest.raises(ValidationError, match="is not numeric"):
            SampleRecord(
                **_make_payload(
                    frames=[
                        [[1.0, "x", 3.0]],
                    ]
                )
            )

    def test_bad_fps_zero(self) -> None:
        with pytest.raises(ValidationError, match="greater than 0"):
            SampleRecord(**_make_payload(fps=0.0))

    def test_bad_fps_negative(self) -> None:
        with pytest.raises(ValidationError, match="greater than 0"):
            SampleRecord(**_make_payload(fps=-5.0))

    def test_default_fps(self) -> None:
        """fps defaults to 30.0 when omitted."""
        payload = {"gloss": "test", "frames": _valid_frames()}
        rec = SampleRecord.model_validate(payload)
        assert rec.fps == 30.0

    def test_default_language(self) -> None:
        """language defaults to 'unknown' when omitted."""
        payload = {"gloss": "test", "frames": _valid_frames()}
        rec = SampleRecord.model_validate(payload)
        assert rec.language == "unknown"

    def test_extra_fields_ignored(self) -> None:
        """Extra fields like 'source', 'extractor' should not cause errors."""
        payload = {
            "gloss": "hello",
            "frames": _valid_frames(),
            "fps": 15.0,
            "language": "demo-asl",
            "source": "synthetic",
            "extractor": "mediapipe",
        }
        rec = SampleRecord.model_validate(payload)
        assert rec.gloss == "hello"
        assert rec.fps == 15.0


# -- All existing data/samples/*.json pass ----------------------------------


def _collect_sample_json_paths() -> list[Path]:
    """Return every .json under data/samples/ (recursive)."""
    return sorted(SAMPLES_DIR.rglob("*.json"))


@pytest.mark.parametrize("path", _collect_sample_json_paths(), ids=lambda p: str(p.relative_to(SAMPLES_DIR)))
def test_all_sample_files_pass_validation(path: Path) -> None:
    """Every shipped .json sample must validate successfully."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    SampleRecord.model_validate(payload)
