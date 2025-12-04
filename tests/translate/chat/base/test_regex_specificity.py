# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from divyam_llm_interop.translate.chat.base.regex_specificity import specificity_score


def test_specificity_score():
    # Pure wildcard
    assert specificity_score(".*") < 0

    # Literal only
    assert specificity_score("gpt4") > specificity_score(".*")

    # Literal + wildcard
    score1 = specificity_score("gpt-4.*")
    score2 = specificity_score(".*")
    assert score1 > score2

    # Anchors increase specificity
    score3 = specificity_score("^gpt-4$")
    score4 = specificity_score("gpt-4")
    assert score3 > score4

    # Character class vs literal
    score5 = specificity_score("[a-z]")
    score6 = specificity_score("a")
    assert score6 > score5  # literal more specific than class

    # Predefined classes
    score7 = specificity_score(r"\d")
    score8 = specificity_score(r"[0-9]")
    assert score7 == score8  # similar specificity

    # Quantifiers
    score9 = specificity_score("a*")
    score10 = specificity_score("a{3}")
    assert score10 > score9  # exact repetition more specific

    # Alternation
    score11 = specificity_score("cat|dog")
    score12 = specificity_score("catdog")
    assert score12 > score11  # alternation reduces specificity

    # Escaped dot
    score13 = specificity_score(r"\.")
    score14 = specificity_score(".")
    assert score13 > score14  # literal dot more specific than wildcard

    print("All specificity_score tests passed.")


def more_specific(r1: str, r2: str) -> str:
    s1 = specificity_score(r1)
    s2 = specificity_score(r2)

    if s1 > s2:
        return r1
    elif s2 > s1:
        return r2
    return "equal"


def test_more_specific():
    assert more_specific(".*", "gpt-4.*") == "gpt-4.*"
    assert more_specific("gpt-4.*", "gpt-4-1106") == "gpt-4-1106"
    assert more_specific("a", "[a-z]") == "a"
    assert more_specific("^abc$", "abc") == "^abc$"
    assert more_specific("cat|dog", "catdog") == "catdog"
    assert more_specific(".*", ".*") == "equal"

    print("All more_specific tests passed.")


if __name__ == "__main__":
    test_specificity_score()
    test_more_specific()
