# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from functools import cmp_to_key, partial

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.model_config.model_catalog import (
    ModelCatalogEntry,
)
from divyam_llm_interop.translate.chat.model_config.model_selector import (
    ModelSelector,
    SelectorRegex,
)
from divyam_llm_interop.translate.chat.types import Model


def make_selector(name=None, provider=None, version=None):
    return ModelSelector(
        name=name,
        provider=provider,
        version=version,
    )


def make_entry(name="gpt-4", provider=None, version=None):
    return ModelCatalogEntry(name=name, provider=provider, version=version)


def test_matches_exact_string():
    selector = make_selector(name="gpt-4")
    model = Model(name="gpt-4", api_type=ModelApiType.COMPLETIONS)
    assert selector.matches(model) is True


def test_matches_string_mismatch():
    selector = make_selector(name="gpt-4")
    model = Model(name="gpt-3.5", api_type=ModelApiType.COMPLETIONS)
    assert selector.matches(model) is False


def test_matches_regex_single():
    selector = make_selector(name=SelectorRegex("^gpt-4$"))
    model = Model(name="gpt-4", api_type=ModelApiType.COMPLETIONS)
    assert selector.matches(model) is True


def test_matches_regex_fails():
    selector = make_selector(name=SelectorRegex("^gpt-4$"))
    model = Model(name="gpt-3.5", api_type=ModelApiType.COMPLETIONS)
    assert selector.matches(model) is False


def test_matches_regex_multi():
    selector = make_selector(name=SelectorRegex(["^gpt-4$", "^gpt-5$"]))
    assert selector.matches(Model(name="gpt-4", api_type=ModelApiType.COMPLETIONS))
    assert selector.matches(Model(name="gpt-5", api_type=ModelApiType.COMPLETIONS))
    assert not selector.matches(
        Model(name="gpt-3.5", api_type=ModelApiType.COMPLETIONS)
    )


def test_matches_catalog_entry_name_only():
    selector = make_selector(name="gpt-4")
    entry = make_entry(name="gpt-4")
    assert selector.matches_catalog_entry(entry) is True


def test_matches_catalog_entry_mismatch():
    selector = make_selector(name="gpt-4")
    entry = make_entry(name="gpt-3.5")
    assert selector.matches_catalog_entry(entry) is False


def test_matches_catalog_entry_provider_and_version():
    selector = make_selector(
        name="gpt-4",
        provider=SelectorRegex("^openai$"),
        version="2025-01-01",
    )
    entry = make_entry(
        name="gpt-4",
        provider="openai",
        version="2025-01-01",
    )
    assert selector.matches_catalog_entry(entry) is True


def test_matches_catalog_entry_missing_provider_fails():
    selector = make_selector(provider=SelectorRegex("^anthropic$"))
    entry = make_entry(provider=None)
    assert selector.matches_catalog_entry(entry) is False


def test_selector_to_from_dict_roundtrip():
    s = make_selector(
        name="gpt-4",
        provider=SelectorRegex(["openai", "azure"]),
    )
    d = s.to_dict()
    restored = ModelSelector.from_dict(d)

    assert restored.name == "gpt-4"
    assert isinstance(restored.provider, SelectorRegex)
    assert restored.provider.patterns == ["openai", "azure"]


def test_from_dict_string_passthrough():
    restored = ModelSelector.from_dict({"name": "gpt-4"})
    assert restored.name == "gpt-4"


def test_from_dict_regex_parsed():
    restored = ModelSelector.from_dict({"name": {"regex": "^gpt-4$"}})
    assert isinstance(restored.name, SelectorRegex)
    assert restored.name.patterns == ["^gpt-4$"]


def test_compare_specificity_string_preferred():
    a = make_selector(name="gpt-4")
    b = make_selector(name=SelectorRegex("^gpt-4$"))
    assert ModelSelector.compare_specificity(a, b, ModelCatalogEntry(name="gpt-4")) > 0


def test_compare_specificity_none_lowest():
    a = make_selector(name=None)
    b = make_selector(name="gpt-4")
    assert ModelSelector.compare_specificity(a, b, ModelCatalogEntry(name="gpt-4")) < 0


def test_sorting_stability_with_cmp_to_key():
    s1 = make_selector(name="gpt-4")
    s2 = make_selector(name="gpt-4")
    lst = [s1, s2]

    # Sort by specificity from least specific to more specific.
    cmp_fn = partial(
        ModelSelector.compare_specificity,
        model_catalog_entry=ModelCatalogEntry(name="gpt-4"),
    )
    sorted_list = sorted(lst, key=cmp_to_key(cmp_fn))
    assert sorted_list == lst  # stable


def test_sorting_order():
    selectors = [
        make_selector(name=None),
        make_selector(name=SelectorRegex(".*")),
        make_selector(name="gpt-4"),
    ]

    cmp_fn = partial(
        ModelSelector.compare_specificity,
        model_catalog_entry=ModelCatalogEntry(name="gpt-4"),
    )
    sorted_list = sorted(selectors, key=cmp_to_key(cmp_fn))

    assert sorted_list[2].name == "gpt-4"
    assert isinstance(sorted_list[1].name, SelectorRegex)
    assert sorted_list[0].name is None


@pytest.mark.parametrize(
    "sel",
    [
        make_selector(name=None),
        make_selector(name="gpt-4"),
        make_selector(name=SelectorRegex("^gpt-4$")),
    ],
)
def test_selector_matches_variants(sel):
    result = sel.matches(Model(name="gpt-4", api_type=ModelApiType.COMPLETIONS))
    assert result in {True, False}
