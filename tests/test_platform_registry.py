#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the PlatformRegistry, focused on the dict-spec config form
introduced to carry per-component params (e.g. remote daemon url/token)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triton_kernel_agent.platform.registry import PlatformRegistry


class _Recorder:
    """A factory that records the kwargs it was instantiated with."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class TestDictSpec:
    def test_plain_string_still_works(self):
        """The original {component: name_string} form is unchanged."""
        r = PlatformRegistry()
        r.register("verifier", "rec", _Recorder)
        inst = r.create_from_config({"verifier": "rec"}, shared="x")
        assert isinstance(inst["verifier"], _Recorder)
        # shared kwargs are forwarded (and filtered by signature, which accepts **kwargs)
        assert inst["verifier"].kwargs["shared"] == "x"

    def test_dict_spec_merges_per_component_params(self):
        """{component: {"impl": name, ...params}} passes params to the factory."""
        r = PlatformRegistry()
        r.register("profiler", "rec", _Recorder)
        inst = r.create_from_config(
            {"profiler": {"impl": "rec", "url": "http://x", "token": "t"}},
            shared="s",
            logger="L",
        )
        rec = inst["profiler"]
        assert "impl" not in rec.kwargs  # 'impl' is consumed, not forwarded
        assert rec.kwargs["url"] == "http://x"
        assert rec.kwargs["token"] == "t"
        assert rec.kwargs["shared"] == "s"  # shared kwargs also forwarded

    def test_per_component_param_overrides_shared(self):
        """A param present in both dict-spec and shared kwargs: dict wins."""
        r = PlatformRegistry()
        r.register("benchmarker", "rec", _Recorder)
        inst = r.create_from_config(
            {"benchmarker": {"impl": "rec", "timeout": 99}},
            timeout=5,
        )
        assert inst["benchmarker"].kwargs["timeout"] == 99

    def test_dict_spec_without_impl_raises(self):
        r = PlatformRegistry()
        r.register("verifier", "rec", _Recorder)
        with pytest.raises(ValueError, match="no 'impl' key"):
            r.create_from_config({"verifier": {"url": "http://x"}})

    def test_mixed_specs_in_one_config(self):
        """Plain string and dict specs can coexist in one config."""
        r = PlatformRegistry()
        r.register("verifier", "rec", _Recorder)
        r.register("profiler", "rec", _Recorder)
        inst = r.create_from_config(
            {
                "verifier": "rec",
                "profiler": {"impl": "rec", "url": "http://y"},
            },
            shared="s",
        )
        assert "url" not in inst["verifier"].kwargs
        assert inst["profiler"].kwargs["url"] == "http://y"
        assert inst["verifier"].kwargs["shared"] == "s"

    def test_filter_still_applies_to_dict_spec(self):
        """Per-component params that the factory doesn't accept are dropped."""
        r = PlatformRegistry()

        class Picky:
            def __init__(self, url):
                self.url = url

        r.register("profiler", "picky", Picky)
        inst = r.create_from_config(
            {"profiler": {"impl": "picky", "url": "http://z", "ignored": 1}}
        )
        assert inst["profiler"].url == "http://z"
        assert not hasattr(inst["profiler"], "ignored")
