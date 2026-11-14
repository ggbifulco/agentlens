"""Packaged judge prompt templates and custom template checks."""

from __future__ import annotations

import pytest

from agentlens.evals import EvaluatorConfigError
from agentlens.evals.judge import check_template, load_template, render_template


@pytest.mark.parametrize("name", ["judge_system", "rubric", "pass_fail", "reference", "pairwise"])
def test_packaged_templates_exist(name: str) -> None:
    text = load_template(name)
    assert text.strip()


def test_rubric_template_placeholders_and_anchors() -> None:
    text = load_template("rubric")
    for placeholder in ("$criteria", "$input", "$output", "$reference"):
        assert placeholder in text
    for anchor in ("5 -", "4 -", "3 -", "2 -", "1 -"):
        assert anchor in text
    assert '"score"' in text
    assert "JSON" in text  # OpenAI json_object mode requires the word in the prompt


def test_pass_fail_template_asks_for_verdict() -> None:
    text = load_template("pass_fail")
    assert '"verdict"' in text
    assert "JSON" in text
    assert "$expected" in load_template("reference")


@pytest.mark.parametrize("name", ["missing", "", "../base", ".hidden", "a\\b"])
def test_unknown_template(name: str) -> None:
    with pytest.raises(FileNotFoundError):
        load_template(name)


def test_render_template_substitutes_once() -> None:
    assert render_template("a=$a b=${b}", {"a": "$b", "b": "{x}"}) == "a=$b b={x}"
    with pytest.raises(KeyError):
        render_template("$missing", {})


def test_check_template() -> None:
    assert check_template("$criteria / $$5", ["criteria"]) == "$criteria / $$5"
    with pytest.raises(EvaluatorConfigError, match="unknown placeholder"):
        check_template("$criteria $nope", ["criteria"])
    with pytest.raises(EvaluatorConfigError, match="invalid"):
        check_template("costs $5", ["criteria"])
    for bad in ("", "  ", 3, None):
        with pytest.raises(EvaluatorConfigError, match="non-empty string"):
            check_template(bad, ["criteria"])


def test_pairwise_template_placeholders() -> None:
    text = load_template("pairwise")
    for placeholder in ("$criteria", "$input", "$output_1", "$output_2", "$reference"):
        assert placeholder in text
    assert '"winner"' in text
    assert "JSON" in text
