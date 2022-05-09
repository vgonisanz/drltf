#!/usr/bin/env python

"""Tests for `drltf` package."""

import drltf

def test_package_publishes_version_info():
    """Tests that the `drltf` publishes the current verion"""

    assert hasattr(drltf, '__version__')
