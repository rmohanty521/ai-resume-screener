from collections import Counter

import pytest

from resume_screener.dataset import RawResume, load_resumes, stratified_sample


def test_load_skips_empty_rows(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text('ID,Resume_str,Category\n1,"Python dev",IT\n2,"",IT\n3,"Chef",CHEF\n')
    assert [r.resume_id for r in load_resumes(path)] == ["1", "3"]


def test_missing_file_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        load_resumes(tmp_path / "nope.csv")


def test_stratified_sample_is_balanced_and_deterministic():
    rs = [RawResume(f"{c}{i}", c, "t") for c in "ABC" for i in range(10)]
    s1 = stratified_sample(rs, 9)
    assert Counter(r.category for r in s1) == {"A": 3, "B": 3, "C": 3}
    assert s1 == stratified_sample(rs, 9)
