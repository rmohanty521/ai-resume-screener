from resume_screener.text import clean_resume, normalize, redact_pii


def test_normalize_collapses_whitespace_and_control_chars():
    assert normalize("  A\x07   B \n\n\n\n C D ") == "A B\n\nC D"


def test_redacts_contact_details():
    out = redact_pii("jane@x.com (555) 123-4567 +91 98765 43210 https://github.com/jane")
    assert out == "[EMAIL] [PHONE] [PHONE] [URL]"


def test_keeps_date_and_salary_ranges():
    text = "Jan 2019 - Dec 2023, salary 50000-60000"
    assert redact_pii(text) == text


def test_truncation_is_reported():
    text, truncated = clean_resume("line\n" * 100, max_chars=50)
    assert truncated and len(text) <= 50
    assert clean_resume("short") == ("short", False)
