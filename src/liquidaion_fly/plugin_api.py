"""The LiquidAion plugin contracts, registry and discovery (headless; never imports Qt).

One registry holds every replaceable component, built-in or user-provided:

  model_adapter      language-model family integration (built-in: Qwen3 profiles)
  connectome         topology and neuron metadata (built-in: FlyWire CX512, random control)
  reservoir          neuronal dynamics (built-in: sparse leaky-tanh rate reservoir)
  neural_interface   LM <-> reservoir mathematics (built-in: the original interface)
  coupling           when the interface runs within a forward pass (legacy / synchronous)
  modulation         external stimulation groups (built-in: neuron cue groups)
  experiment         runnable experiments: run(api)
  ui_panel           dockable Qt panels (created only by the GUI)

A plugin is a directory with ``plugin.json`` (manifest, read without executing code) and
``plugin.py`` defining ``register(registrar)``. Only plugins explicitly enabled are imported.
Python plugins are trusted code with the full permissions of the user; nothing is sandboxed.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable

API_VERSION = "1.0"
CATEGORIES = ("model_adapter", "connectome", "reservoir", "neural_interface", "coupling", "modulation",
              "experiment", "ui_panel")
PARAMETER_TYPES = ("bool", "int", "float", "str", "enum", "path", "layer", "range")
ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$")
PLUGIN_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_PLUGIN_ROOT = REPO_ROOT / "plugins" / "packages"
USER_PLUGIN_ROOT = Path.home() / ".liquidaion" / "plugins"


class PluginError(RuntimeError):
    """A plugin or component problem with a message meant for the researcher."""


@dataclass(frozen=True)
class Parameter:
    """One configurable value; drives validation and the generated parameter editor."""
    name: str
    type: str
    default: Any = None
    description: str = ""
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple = ()
    step: float | None = None

    def __post_init__(self) -> None:
        if self.type not in PARAMETER_TYPES:
            raise PluginError(f"Parameter {self.name!r}: unknown type {self.type!r}; use one of {PARAMETER_TYPES}")
        if self.type == "enum" and not self.choices:
            raise PluginError(f"Parameter {self.name!r}: enum needs choices")

    def validate(self, value: Any) -> Any:
        """Return the value in its canonical type or raise PluginError with a clear message."""
        name = self.name
        try:
            if self.type == "bool":
                if not isinstance(value, bool):
                    raise TypeError("expected true or false")
                return value
            if self.type in {"int", "layer"}:
                if isinstance(value, bool) or int(value) != value:
                    raise TypeError("expected an integer")
                value = int(value)
            elif self.type == "float":
                value = float(value)
                if value != value or value in (float("inf"), float("-inf")):
                    raise ValueError("expected a finite number")
            elif self.type == "range":
                low, high = (float(v) for v in value)
                if low > high:
                    raise ValueError("range minimum exceeds maximum")
                self._bounds(low)
                self._bounds(high)
                return (low, high)
            elif self.type in {"str", "path"}:
                if not isinstance(value, str):
                    raise TypeError("expected text")
                return value
            elif self.type == "enum":
                if value not in self.choices:
                    raise ValueError(f"expected one of {list(self.choices)}")
                return value
            self._bounds(value)
            return value
        except (TypeError, ValueError) as exc:
            raise PluginError(f"Parameter {name!r}: {exc} (got {value!r})") from None

    def _bounds(self, value: float) -> None:
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"must be >= {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"must be <= {self.maximum}")


@dataclass(frozen=True)
class ComponentSpec:
    """A registered implementation. ``factory`` is called by the kernel (see docs/PLUGINS.md)."""
    component_id: str
    category: str
    name: str
    factory: Callable[..., Any]
    version: str = "1.0.0"
    api_version: str = API_VERSION
    description: str = ""
    capabilities: tuple[str, ...] = ()
    parameters: tuple[Parameter, ...] = ()
    requires: tuple[str, ...] = ()          # component ids that must also be registered
    plugin_id: str = "builtin"
    # Category-specific metadata, e.g. coupling semantics or a panel's default dock area.
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolve_params(self, values: dict[str, Any] | None) -> dict[str, Any]:
        """Defaults overlaid with validated values; unknown names are errors."""
        values = dict(values or {})
        known = {p.name: p for p in self.parameters}
        unknown = set(values) - set(known)
        if unknown:
            raise PluginError(f"{self.component_id}: unknown parameters {sorted(unknown)}; "
                              f"known: {sorted(known) or 'none'}")
        return {name: p.validate(values[name]) if name in values else p.default for name, p in known.items()}

    def describe(self) -> dict[str, Any]:
        return {"id": self.component_id, "category": self.category, "name": self.name, "version": self.version,
                "api_version": self.api_version, "plugin": self.plugin_id, "description": self.description,
                "capabilities": list(self.capabilities),
                "parameters": [{"name": p.name, "type": p.type, "default": p.default, "description": p.description}
                               for p in self.parameters]}


@dataclass(frozen=True)
class PluginManifest:
    plugin_id: str
    name: str
    version: str
    api_version: str
    description: str = ""
    requires: tuple[str, ...] = ()           # other plugin ids
    entry: str = "plugin.py"
    path: Path | None = None

    @classmethod
    def read(cls, directory: Path) -> "PluginManifest":
        path = directory / "plugin.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PluginError(f"{path}: unreadable manifest ({exc})") from None
        missing = {"id", "name", "version", "api_version"} - set(data)
        if missing:
            raise PluginError(f"{path}: manifest lacks {sorted(missing)}")
        if not PLUGIN_ID_PATTERN.match(str(data["id"])) or data["id"] == "builtin":
            raise PluginError(f"{path}: plugin id {data['id']!r} must be lowercase letters, digits and '_' "
                              "(and not 'builtin')")
        entry = str(data.get("entry", "plugin.py"))
        if Path(entry).name != entry or not entry.endswith(".py"):
            raise PluginError(f"{path}: entry must be a .py file inside the plugin directory")
        return cls(str(data["id"]), str(data["name"]), str(data["version"]), str(data["api_version"]),
                   str(data.get("description", "")), tuple(data.get("requires", ())), entry, directory)


def api_compatible(version: str) -> bool:
    """Same major version and a minor version this kernel provides."""
    try:
        major, minor = (int(x) for x in str(version).split(".")[:2])
        own_major, own_minor = (int(x) for x in API_VERSION.split("."))
    except ValueError:
        return False
    return major == own_major and minor <= own_minor


class PluginRegistrar:
    """What a plugin's register() receives: adds components under the plugin's own namespace."""

    def __init__(self, registry: "PluginRegistry", plugin_id: str):
        self._registry = registry
        self.plugin_id = plugin_id
        self._pending: list[ComponentSpec] = []

    def add(self, spec: ComponentSpec) -> ComponentSpec:
        prefix = "builtin." if self.plugin_id == "builtin" else f"{self.plugin_id}."
        if not spec.component_id.startswith(prefix):
            raise PluginError(f"Component {spec.component_id!r} of plugin {self.plugin_id!r} must start with {prefix!r}")
        spec = replace(spec, plugin_id=self.plugin_id)
        self._registry._check(spec, pending=self._pending)
        self._pending.append(spec)
        return spec

    # Convenience constructors -------------------------------------------------------------
    def neural_interface(self, component_id: str, name: str, factory: Callable, **kwargs: Any) -> ComponentSpec:
        return self.add(ComponentSpec(component_id, "neural_interface", name, factory, **kwargs))

    def connectome(self, component_id: str, name: str, factory: Callable, **kwargs: Any) -> ComponentSpec:
        return self.add(ComponentSpec(component_id, "connectome", name, factory, **kwargs))

    def reservoir(self, component_id: str, name: str, factory: Callable, **kwargs: Any) -> ComponentSpec:
        return self.add(ComponentSpec(component_id, "reservoir", name, factory, **kwargs))

    def experiment(self, component_id: str, name: str, run: Callable, **kwargs: Any) -> ComponentSpec:
        return self.add(ComponentSpec(component_id, "experiment", name, run, **kwargs))

    def panel(self, component_id: str, title: str, factory: Callable, area: str = "right",
              visible: bool = True, **kwargs: Any) -> ComponentSpec:
        """A dockable Qt panel. factory(context) -> QWidget runs on the GUI thread only.

        Import Qt inside the factory (or a module it imports) so headless use never needs Qt."""
        if area not in {"left", "right", "top", "bottom"}:
            raise PluginError(f"Panel {component_id!r}: area must be left, right, top or bottom")
        metadata = {**kwargs.pop("metadata", {}), "area": area, "visible": visible}
        return self.add(ComponentSpec(component_id, "ui_panel", title, factory, metadata=metadata, **kwargs))


class PluginRegistry:
    """The single component registry of a process. Read-only once loading has finished."""

    def __init__(self) -> None:
        self._components: dict[str, ComponentSpec] = {}
        self.plugins: dict[str, PluginManifest] = {}
        self.errors: dict[str, str] = {}          # plugin id or path -> message for plugins that failed
        self.discovered: list[PluginManifest] = []

    # ------------------------------------------------------------------- registration
    def _check(self, spec: ComponentSpec, pending: Iterable[ComponentSpec] = ()) -> None:
        if spec.category not in CATEGORIES:
            raise PluginError(f"{spec.component_id}: unknown category {spec.category!r}; use one of {CATEGORIES}")
        if not ID_PATTERN.match(spec.component_id):
            raise PluginError(f"{spec.component_id!r}: ids are dotted lowercase names, e.g. 'myplugin.my_interface'")
        if spec.component_id in self._components or any(p.component_id == spec.component_id for p in pending):
            raise PluginError(f"Duplicate component id {spec.component_id!r}; an existing implementation is never "
                              "replaced")
        if not api_compatible(spec.api_version):
            raise PluginError(f"{spec.component_id} targets plugin API {spec.api_version}; this LiquidAion "
                              f"provides {API_VERSION}")
        if not callable(spec.factory):
            raise PluginError(f"{spec.component_id}: factory is not callable")
        names = [p.name for p in spec.parameters]
        if len(names) != len(set(names)):
            raise PluginError(f"{spec.component_id}: duplicate parameter names")
        for parameter in spec.parameters:
            if parameter.default is not None:
                parameter.validate(parameter.default)

    def register_builtin(self, register: Callable[[PluginRegistrar], None]) -> None:
        self._load("builtin", register)

    def _load(self, plugin_id: str, register: Callable[[PluginRegistrar], None]) -> None:
        registrar = PluginRegistrar(self, plugin_id)
        register(registrar)  # all-or-nothing: nothing is committed if register() raises
        for spec in registrar._pending:
            self._components[spec.component_id] = spec

    def register_plugin(self, plugin_id: str, register: Callable[[PluginRegistrar], None], version: str = "0.0.0",
                        description: str = "") -> None:
        """Register an in-process plugin (tests, notebooks, scripts) under the same rules as a directory plugin."""
        if not PLUGIN_ID_PATTERN.match(plugin_id) or plugin_id == "builtin":
            raise PluginError(f"Plugin id {plugin_id!r} must be lowercase letters, digits and '_' (and not 'builtin')")
        if plugin_id in self.plugins:
            raise PluginError(f"Plugin {plugin_id!r} is already loaded")
        self._load(plugin_id, register)
        self.plugins[plugin_id] = PluginManifest(plugin_id, plugin_id, version, API_VERSION, description)

    def validate_dependencies(self) -> None:
        for spec in self._components.values():
            missing = [dep for dep in spec.requires if dep not in self._components]
            if missing:
                raise PluginError(f"{spec.component_id} requires unregistered components {missing}")

    # ------------------------------------------------------------------- lookup
    def get(self, component_id: str, category: str | None = None) -> ComponentSpec:
        spec = self._components.get(component_id)
        if spec is None:
            available = ", ".join(s.component_id for s in self.list(category)) if category else ""
            raise PluginError(f"Component {component_id!r} is not registered"
                              + (f"; available {category}: {available}" if category else "")
                              + ". Enable the plugin that provides it; nothing is substituted.")
        if category and spec.category != category:
            raise PluginError(f"{component_id} is a {spec.category}, not a {category}")
        return spec

    def list(self, category: str | None = None) -> list[ComponentSpec]:
        return [s for s in self._components.values() if category is None or s.category == category]

    def __contains__(self, component_id: str) -> bool:
        return component_id in self._components

    # ------------------------------------------------------------------- user plugins
    def load_plugins(self, manifests: Iterable[PluginManifest], enabled: Iterable[str]) -> list[str]:
        """Import enabled plugins in dependency order; a failing plugin is recorded, never half-loaded."""
        manifests = {m.plugin_id: m for m in manifests}
        wanted = [pid for pid in dict.fromkeys(enabled)]
        loaded = []
        for pid in wanted:
            if pid not in manifests:
                self.errors[pid] = "enabled but not found in any plugin directory"
        order: list[str] = []

        def visit(pid: str, chain: tuple[str, ...]) -> None:
            if pid in order or pid in self.errors:
                return
            if pid in chain:
                self.errors[pid] = f"dependency cycle {' -> '.join(chain + (pid,))}"
                return
            manifest = manifests.get(pid)
            if manifest is None:
                self.errors[pid] = "required plugin not found"
                return
            for dep in manifest.requires:
                if dep not in wanted:
                    self.errors[pid] = f"requires plugin {dep!r}, which is not enabled"
                    return
                visit(dep, chain + (pid,))
                if dep in self.errors:
                    self.errors[pid] = f"required plugin {dep!r} failed: {self.errors[dep]}"
                    return
            order.append(pid)
        for pid in wanted:
            if pid in manifests:
                visit(pid, ())
        for pid in order:
            manifest = manifests[pid]
            try:
                if not api_compatible(manifest.api_version):
                    raise PluginError(f"targets plugin API {manifest.api_version}; this LiquidAion provides {API_VERSION}")
                module = import_plugin(manifest)
                register = getattr(module, "register", None)
                if not callable(register):
                    raise PluginError(f"{manifest.entry} must define register(registrar)")
                self._load(pid, register)
                self.plugins[pid] = manifest
                loaded.append(pid)
            except Exception as exc:  # recorded and reported; the application keeps running
                self.errors[pid] = f"{type(exc).__name__}: {exc}"
        return loaded

    def report(self) -> dict[str, Any]:
        return {"api_version": API_VERSION,
                "plugins": {pid: {"version": m.version, "path": str(m.path)} for pid, m in self.plugins.items()},
                "errors": dict(self.errors),
                "components": {c: [s.component_id for s in self.list(c)] for c in CATEGORIES}}


def import_plugin(manifest: PluginManifest) -> Any:
    """Import plugin.py as the package ``liquidaion_plugins.<id>`` (relative imports work; no sys.path change)."""
    name = f"liquidaion_plugins.{manifest.plugin_id}"
    if "liquidaion_plugins" not in sys.modules:
        namespace = importlib.util.module_from_spec(importlib.machinery.ModuleSpec("liquidaion_plugins", None,
                                                                                    is_package=True))
        namespace.__path__ = []
        sys.modules["liquidaion_plugins"] = namespace
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, manifest.path / manifest.entry,
                                                  submodule_search_locations=[str(manifest.path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def plugin_roots(extra: Iterable[str | Path] = ()) -> list[Path]:
    """Bundled examples, the per-user directory, LIQUIDAION_PLUGIN_PATH entries and explicit extras."""
    roots = [BUNDLED_PLUGIN_ROOT, USER_PLUGIN_ROOT]
    roots += [Path(p) for p in os.environ.get("LIQUIDAION_PLUGIN_PATH", "").split(os.pathsep) if p]
    roots += [Path(p) for p in extra]
    return list(dict.fromkeys(r.expanduser().resolve() for r in roots))


def discover(roots: Iterable[Path]) -> tuple[list[PluginManifest], dict[str, str]]:
    """Read manifests of immediate subdirectories that contain plugin.json; executes no code."""
    manifests, errors, seen = [], {}, {}
    for root in roots:
        if not root.is_dir():
            continue
        for directory in sorted(p for p in root.iterdir() if p.is_dir() and (p / "plugin.json").exists()):
            try:
                manifest = PluginManifest.read(directory)
            except PluginError as exc:
                errors[str(directory)] = str(exc)
                continue
            if manifest.plugin_id in seen:
                errors[manifest.plugin_id] = f"duplicate plugin id in {seen[manifest.plugin_id]} and {directory}"
                continue
            seen[manifest.plugin_id] = directory
            manifests.append(manifest)
    return manifests, errors


def enabled_from_environment() -> list[str]:
    return [p.strip() for p in os.environ.get("LIQUIDAION_PLUGINS", "").split(",") if p.strip()]


def build_registry(enabled: Iterable[str] = (), roots: Iterable[str | Path] = ()) -> PluginRegistry:
    """Built-in components plus the explicitly enabled user plugins (and LIQUIDAION_PLUGINS)."""
    from .builtin_plugins import register as register_builtin
    registry = PluginRegistry()
    registry.register_builtin(register_builtin)
    manifests, errors = discover(plugin_roots(roots))
    registry.errors.update(errors)
    registry.load_plugins(manifests, list(enabled) + enabled_from_environment())
    registry.validate_dependencies()
    registry.discovered = manifests
    return registry


_DEFAULT: PluginRegistry | None = None


def default_registry() -> PluginRegistry:
    """The process-wide registry used when no registry is passed explicitly (built-ins + env-enabled)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = build_registry()
    return _DEFAULT
