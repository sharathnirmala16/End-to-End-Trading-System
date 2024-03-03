import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "download_required: these are slower tests that are run by downloading from yfinance",
    )
