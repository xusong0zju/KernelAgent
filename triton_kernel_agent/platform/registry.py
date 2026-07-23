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

"""Platform component registry.

Maps ``(component_key, implementation_name)`` pairs to factory
callables so that a YAML config like::

    platform:
      verifier: nvidia
      benchmarker: nvidia

can drive component selection at runtime.

Usage::

    from triton_kernel_agent.platform.registry import registry

    # Resolve a single component
    verifier = registry.create("verifier", "nvidia")

    # Resolve a full config dict (unrecognised kwargs are filtered
    # per-factory so callers can pass a shared kwargs bag)
    components = registry.create_from_config(
        {"verifier": "nvidia", "benchmarker": "nvidia"},
        log_dir=some_path, logger=some_logger, benchmark_lock=lock,
    )

    # Register a new backend
    registry.register("verifier", "rocm", RocmVerifier)
"""

from __future__ import annotations

import inspect
from typing import Any, Callable


class PlatformRegistry:
    """Central registry that maps component keys to named implementations.

    Each *component key* (e.g. ``"verifier"``) has one or more named
    implementations (e.g. ``"nvidia"``, ``"noop"``).  Each
    implementation is stored as a callable **factory** — typically a
    class, but any ``(**kwargs) -> instance`` callable works.

    When :meth:`create` is called the factory receives only the subset
    of ``**kwargs`` that its signature actually accepts, so callers can
    pass a shared bag of context (``log_dir``, ``logger``, …) without
    worrying about ``TypeError`` from unrelated keys.
    """

    def __init__(self) -> None:
        # {component_key: {impl_name: factory}}
        self._factories: dict[str, dict[str, Callable[..., Any]]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        component: str,
        name: str,
        factory: Callable[..., Any],
    ) -> None:
        """Register *factory* under ``(component, name)``.

        Args:
            component: Component key, e.g. ``"verifier"``.
            name: Implementation name, e.g. ``"nvidia"`` or ``"noop"``.
            factory: A callable (class or function) that returns an
                instance of the component.  It will be called with
                filtered ``**kwargs`` (see :meth:`create`).
        """
        self._factories.setdefault(component, {})[name] = factory

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def list_components(self) -> list[str]:
        """Return all registered component keys."""
        return sorted(self._factories)

    def list_implementations(self, component: str) -> list[str]:
        """Return all registered names for *component*."""
        return sorted(self._factories.get(component, {}))

    def has(self, component: str, name: str) -> bool:
        """Check whether ``(component, name)`` is registered."""
        return name in self._factories.get(component, {})

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def create(self, component: str, name: str, **kwargs: Any) -> Any:
        """Instantiate ``(component, name)`` with filtered *kwargs*.

        Only the kwargs whose names match the factory's ``__init__``
        (or function) signature are forwarded; the rest are silently
        dropped.  This makes it safe for callers to pass a superset of
        context.

        Raises:
            KeyError: If *component* or *name* is not registered.
        """
        impls = self._factories.get(component)
        if impls is None:
            raise KeyError(
                f"Unknown component {component!r}. Registered: {self.list_components()}"
            )
        factory = impls.get(name)
        if factory is None:
            raise KeyError(
                f"Unknown implementation {name!r} for {component!r}. "
                f"Registered: {self.list_implementations(component)}"
            )
        filtered = _filter_kwargs(factory, kwargs)
        return factory(**filtered)

    def create_from_config(
        self,
        config: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create multiple components from a ``{component: spec}`` dict.

        Each *spec* may be either:

        - a **plain string** — the implementation name (e.g. ``"nvidia"``);
          the factory receives only the shared ``**kwargs`` bag. This is
          the original form and stays fully backward-compatible.
        - a **dict** — ``{"impl": name, ...extra_params}``; the factory
          receives the shared ``**kwargs`` *plus* the per-component
          ``extra_params`` (per-component params override shared kwargs on
          key collision). This lets a single config carry connection info
          for ``remote`` components, e.g.
          ``{"profiler": {"impl": "remote", "url": "...", "token": "..."}}``
          without polluting other components or env-globals.

        Every component listed in *config* is instantiated via
        :meth:`create`, sharing the same *kwargs* bag (each factory
        only receives the kwargs it accepts).

        Returns:
            ``{component_key: instance}`` for every entry in *config*.
        """
        instances: dict[str, Any] = {}
        for component, spec in config.items():
            if isinstance(spec, dict):
                spec = dict(spec)  # shallow copy; we pop from it
                name = spec.pop("impl", None)
                if name is None:
                    raise ValueError(
                        f"platform config for {component!r} is a dict but "
                        f"has no 'impl' key: {spec!r}"
                    )
                merged = {**kwargs, **spec}
            else:
                name = spec
                merged = kwargs
            instances[component] = self.create(component, name, **merged)
        return instances



# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _filter_kwargs(
    factory: Callable[..., Any],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Return the subset of *kwargs* accepted by *factory*'s signature.

    If the factory accepts ``**kwargs`` (VAR_KEYWORD), all kwargs are
    passed through unfiltered.
    """
    try:
        # For classes, inspect __init__
        sig = inspect.signature(factory)
    except (ValueError, TypeError):
        return kwargs

    accepted: set[str] = set()
    has_var_keyword = False
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
            break
        if param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            accepted.add(param.name)

    if has_var_keyword:
        return kwargs

    return {k: v for k, v in kwargs.items() if k in accepted}


# ==================================================================
# Global singleton & built-in registrations
# ==================================================================

registry = PlatformRegistry()


def _register_builtins() -> None:
    """Register the built-in nvidia and noop implementations."""

    from triton_kernel_agent.platform.nvidia import (
        NvidiaAcceleratorSpecsProvider,
        NvidiaBenchmarker,
        NvidiaBottleneckAnalyzer,
        NvidiaKernelProfiler,
        NvidiaRAGPrescriber,
        NvidiaRooflineAnalyzer,
        NvidiaVerifier,
        NvidiaWorkerRunner,
    )

    _nvidia = {
        # Manager-level
        "verifier": NvidiaVerifier,
        "benchmarker": NvidiaBenchmarker,
        "worker_runner": NvidiaWorkerRunner,
        # Worker-level
        "specs_provider": NvidiaAcceleratorSpecsProvider,
        "profiler": NvidiaKernelProfiler,
        "roofline_analyzer": NvidiaRooflineAnalyzer,
        "bottleneck_analyzer": NvidiaBottleneckAnalyzer,
        "rag_prescriber": NvidiaRAGPrescriber,
    }
    for component, factory in _nvidia.items():
        registry.register(component, "nvidia", factory)

    from triton_kernel_agent.platform.noop import (
        NoOpBenchmarker,
        NoOpBottleneckAnalyzer,
        NoOpProfiler,
        NoOpRAGPrescriber,
        NoOpRooflineAnalyzer,
        NoOpSpecsProvider,
        NoOpVerifier,
        NoOpWorkerRunner,
    )

    _noop = {
        # Manager-level
        "verifier": NoOpVerifier,
        "benchmarker": NoOpBenchmarker,
        "worker_runner": NoOpWorkerRunner,
        # Worker-level
        "specs_provider": NoOpSpecsProvider,
        "profiler": NoOpProfiler,
        "roofline_analyzer": NoOpRooflineAnalyzer,
        "bottleneck_analyzer": NoOpBottleneckAnalyzer,
        "rag_prescriber": NoOpRAGPrescriber,
    }
    for component, factory in _noop.items():
        registry.register(component, "noop", factory)

    from triton_kernel_agent.platform.remote import (
        RemoteBenchmarker,
        RemoteKernelProfiler,
        RemoteVerifier,
    )

    # Remote (HTTP daemon) implementations. Only the GPU-touching
    # interfaces are remote; specs/roofline/bottleneck/rag stay local on
    # the agent brain, so they deliberately have no "remote" entry. A
    # config using ``platform: remote`` is therefore normally a *mixed*
    # dict (e.g. verifier/benchmarker/profiler=remote + the rest=nvidia/
    # noop) rather than the bare shorthand — see the plan's config section.
    _remote = {
        "verifier": RemoteVerifier,
        "benchmarker": RemoteBenchmarker,
        "profiler": RemoteKernelProfiler,
    }
    for component, factory in _remote.items():
        registry.register(component, "remote", factory)


_register_builtins()
