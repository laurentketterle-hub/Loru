"""Pydantic v2 schema validation for Loru sample data."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


def _is_numeric(val: Any) -> bool:
    """Check if a value is numeric (int or float, not bool)."""
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def _flatten_vsl_landmarks(frames: list) -> list[list[float]]:
    """Flatten VSL-style dict frames into a 2-D landmark list.

    Input: ``[{"frame_idx": 0, "landmarks": {"hand1": [[x,y,z],...], ...}}, ...]``
    Output: ``[[x, y, z], ...]`` for every landmark across all hands and frames.
    """
    result: list[list[float]] = []
    ref_dim: int | None = None

    for frame_idx, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"frame {frame_idx} is not a dict")
        landmarks = frame.get("landmarks")
        if not isinstance(landmarks, dict):
            raise ValueError(
                f"frame {frame_idx} 'landmarks' is missing or not a dict"
            )
        for hand_name, hand_landmarks in landmarks.items():
            if not isinstance(hand_landmarks, list):
                raise ValueError(
                    f"frame {frame_idx} hand '{hand_name}' landmarks is not a list"
                )
            for lm_idx, lm in enumerate(hand_landmarks):
                if not isinstance(lm, list):
                    raise ValueError(
                        f"frame {frame_idx} hand '{hand_name}' "
                        f"landmark {lm_idx} is not a list"
                    )
                if ref_dim is None:
                    ref_dim = len(lm)
                elif len(lm) != ref_dim:
                    raise ValueError(
                        f"frame {frame_idx} hand '{hand_name}' "
                        f"landmark {lm_idx} has {len(lm)} values, "
                        f"expected {ref_dim}"
                    )
                for coord_idx, coord in enumerate(lm):
                    if not _is_numeric(coord):
                        raise ValueError(
                            f"frame {frame_idx} hand '{hand_name}' "
                            f"landmark {lm_idx} coord {coord_idx} "
                            f"is not numeric"
                        )
                result.append([float(x) for x in lm])

    if not result:
        raise ValueError("frames must be non-empty (no landmarks found)")
    return result


def _validate_and_flatten_frames(v: Any, field_name: str = "frames") -> list[list[float]]:
    """Validate frame data and flatten to 2-D.

    Accepts three formats:

    1. **2-D numeric**: ``[[f1, f2, ...], [f1, f2, ...], ...]``
    2. **3-D landmark**: ``[[[x,y,z], ...], [[x,y,z], ...], ...]``
       (standard Loru samples — flattened to 2-D)
    3. **VSL dict**: ``[{"frame_idx":0, "landmarks":{"hand":[[x,y,z],...]}}, ...]``
       (flattened to 2-D)

    Returns a clean ``list[list[float]]`` with uniform row lengths.
    """
    if not isinstance(v, list):
        raise ValueError(f"{field_name} must be a list")
    if len(v) == 0:
        raise ValueError(f"{field_name} must be non-empty")

    first = v[0]

    # --- VSL dict format -------------------------------------------------
    if isinstance(first, dict):
        return _flatten_vsl_landmarks(v)

    if not isinstance(first, list):
        raise ValueError(f"first element of {field_name} must be a list")
    if len(first) == 0:
        raise ValueError(f"first row of {field_name} is empty")

    # --- 3-D landmark data (list of frames of landmarks) -----------------
    if isinstance(first[0], list):
        ref_dim = len(first[0])
        for frame_idx, frame in enumerate(v):
            if not isinstance(frame, list):
                raise ValueError(f"frame {frame_idx} is not a list")
            for lm_idx, lm in enumerate(frame):
                if not isinstance(lm, list):
                    raise ValueError(
                        f"frame {frame_idx} landmark {lm_idx} is not a list"
                    )
                if len(lm) != ref_dim:
                    raise ValueError(
                        f"frame {frame_idx} landmark {lm_idx} has "
                        f"{len(lm)} values, expected {ref_dim}"
                    )
                for coord_idx, coord in enumerate(lm):
                    if not _is_numeric(coord):
                        raise ValueError(
                            f"frame {frame_idx} landmark {lm_idx} "
                            f"coord {coord_idx} is not numeric"
                        )
        # Flatten: each landmark becomes a row
        flattened: list[list[float]] = []
        for frame in v:
            for lm in frame:
                flattened.append([float(x) for x in lm])
        return flattened

    # --- Plain 2-D validation -------------------------------------------
    row_len = len(first)
    for row_idx, row in enumerate(v):
        if not isinstance(row, list):
            raise ValueError(f"row {row_idx} is not a list")
        if len(row) != row_len:
            raise ValueError(
                f"row {row_idx} has {len(row)} values, expected {row_len}"
            )
        for col_idx, val in enumerate(row):
            if not _is_numeric(val):
                raise ValueError(f"row {row_idx} col {col_idx} is not numeric")
    return [[float(x) for x in row] for row in v]


class FrameValidator(BaseModel):
    """Standalone validator for frame-data structural constraints."""

    frames: list = Field(..., min_length=1)

    @field_validator("frames", mode="before")
    @classmethod
    def frames_must_be_2d_numeric(cls, v: Any) -> Any:
        """Ensure *v* is a non-empty list of lists where all rows have the
        same dimension and every leaf value is numeric.  3-D landmark
        and VSL dict data are flattened to 2-D.
        """
        return _validate_and_flatten_frames(v, "frames")


class SampleRecord(BaseModel):
    """Pydantic model for a single Loru sample record.

    Validates the JSON payload loaded from ``data/samples/*.json`` and
    ensures it meets the structural invariants required by the rest of
    the pipeline.
    """

    gloss: str = Field(..., min_length=1)
    frames: list = Field(..., min_length=1)
    fps: float = Field(default=30.0, gt=0)
    language: str = Field(default="unknown")

    @field_validator("frames", mode="before")
    @classmethod
    def _validate_frames(cls, v: Any) -> Any:
        """Delegate to the shared frame validation/flattening logic."""
        return _validate_and_flatten_frames(v, "frames")
