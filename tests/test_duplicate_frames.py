"""Tests for near-duplicate gloss frame detector (Closes #246)."""
import json
import tempfile
from pathlib import Path
from scripts.detect_duplicate_frames import (
    frame_fingerprint, detect_near_duplicates, format_duplicate_report,
)

def create_gloss_file(directory, name, frames):
    path = directory / name
    path.write_text(json.dumps({"frames": frames}))
    return path

def make_frame(landmarks, label="A"):
    return {"label": label, "landmarks": landmarks, "confidence": 0.9}


class TestFrameFingerprint:
    def test_identical_frames_same_fingerprint(self):
        f1 = make_frame([0.1, 0.2, 0.3])
        f2 = make_frame([0.1, 0.2, 0.3])
        assert frame_fingerprint(f1) == frame_fingerprint(f2)

    def test_different_frames_different_fingerprint(self):
        f1 = make_frame([0.1, 0.2, 0.3])
        f2 = make_frame([0.4, 0.5, 0.6])
        assert frame_fingerprint(f1) != frame_fingerprint(f2)


class TestNearDuplicateDetection:
    def test_identical_files_detected(self, tmp_path):
        frames = [make_frame([i, i+1, i+2]) for i in range(5)]
        create_gloss_file(tmp_path, "a.json", frames)
        create_gloss_file(tmp_path, "b.json", frames)  # identical
        dups = detect_near_duplicates(tmp_path, threshold=0.5)
        assert len(dups) >= 1

    def test_different_files_not_detected(self, tmp_path):
        frames_a = [make_frame([i, i+1, i+2]) for i in range(5)]
        frames_b = [make_frame([i+10, i+11, i+12]) for i in range(5)]
        create_gloss_file(tmp_path, "a.json", frames_a)
        create_gloss_file(tmp_path, "b.json", frames_b)
        dups = detect_near_duplicates(tmp_path, threshold=0.5)
        assert len(dups) == 0

    def test_partial_overlap_detected(self, tmp_path):
        common = [make_frame([1.0, 2.0, 3.0]), make_frame([4.0, 5.0, 6.0])]
        frames_a = common + [make_frame([7.0, 8.0, 9.0])]
        frames_b = common + [make_frame([10.0, 11.0, 12.0])]
        create_gloss_file(tmp_path, "a.json", frames_a)
        create_gloss_file(tmp_path, "b.json", frames_b)
        dups = detect_near_duplicates(tmp_path, threshold=0.4)
        assert len(dups) >= 1

    def test_empty_dir_no_duplicates(self, tmp_path):
        dups = detect_near_duplicates(tmp_path)
        assert dups == []

    def test_single_file_no_duplicates(self, tmp_path):
        create_gloss_file(tmp_path, "only.json", [make_frame([1,2,3])])
        dups = detect_near_duplicates(tmp_path)
        assert dups == []


class TestFormatReport:
    def test_no_duplicates_report(self):
        report = format_duplicate_report([])
        assert "No near-duplicate" in report

    def test_duplicates_report(self):
        dups = [("a.json", "b.json", 0.85, 17, 20)]
        report = format_duplicate_report(dups)
        assert "a.json" in report
        assert "b.json" in report
        assert "85.0%" in report
