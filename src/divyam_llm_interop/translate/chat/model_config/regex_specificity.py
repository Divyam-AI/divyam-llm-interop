# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import re

WEIGHTS = {
    "LITERAL": 8,
    "CHARCLASS": 5,  # [abc], [0-9]
    "PREDEFINED": 5,  # \d, \w, \s same as CHARCLASS
    "DOT": -10,
    "ANCHOR": 4,
    "ALTERNATION": -6,
    "QUANTIFIER": -3,
    "EMPTY": -9999,
}


def specificity_score(pattern: str) -> int:
    """Computes a specificity score for a regex pattern.
    A simple implementation that uses weights and penalties for regex
    operators to score specificity of a pattern. A high value of specificity
    means this pattern matches a smaller subset of the .* regex and
    vice-versa.
    """
    if pattern == ".*":
        return WEIGHTS["EMPTY"]

    score = 0

    # Anchors
    score += pattern.count("^") * WEIGHTS["ANCHOR"]
    score += pattern.count("$") * WEIGHTS["ANCHOR"]

    # Extract character classes (ignore contents)
    char_classes = re.findall(r"\[[^\]]+\]", pattern)
    score += len(char_classes) * WEIGHTS["CHARCLASS"]
    temp_pattern = re.sub(r"\[[^\]]+\]", "", pattern)

    # Extract predefined classes (\d, \w, \s, \D, \W, \S)
    predefined = re.findall(r"\\[dwsDWS]", temp_pattern)
    score += len(predefined) * WEIGHTS["PREDEFINED"]
    temp_pattern = re.sub(r"\\[dwsDWS]", "", temp_pattern)

    # Count remaining literals
    literals = re.findall(r"[A-Za-z0-9_-]", temp_pattern)
    score += len(literals) * WEIGHTS["LITERAL"]

    # Count unescaped dot wildcards
    wildcard_dots = re.findall(r"(?<!\\)\.", temp_pattern)
    score += len(wildcard_dots) * WEIGHTS["DOT"]

    # Quantifiers
    quantifiers = re.findall(r"(\*|\+|\?|{\d+(?:,\d*)?})", temp_pattern)
    score += len(quantifiers) * WEIGHTS["QUANTIFIER"]

    # Alternation
    score += temp_pattern.count("|") * WEIGHTS["ALTERNATION"]

    return score
