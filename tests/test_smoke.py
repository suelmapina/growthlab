"""Smoke tests: the package and its heavy dependencies import cleanly."""

import importlib

import pytest

import growthlab
from growthlab import config


def test_version():
    assert growthlab.version


def test_paths_inside_repo():
    assert config.rawdir.is_relative_to(config.root)
    assert config.processeddir.is_relative_to(config.root)


@pytest.mark.parametrize(
    "module",
    ["pymc_marketing.clv", "econml.metalearners", "lightgbm", "mlflow", "fastapi", "streamlit"],
)
def test_dependencies_import(module):
    importlib.import_module(module)
