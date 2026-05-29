"""
Sensor deployment file IO (load / dump).

Supports YAML (``.yaml``, ``.yml``) and JSON (``.json``) per R10.3. Format
selection is purely by file extension — callers either pass a path with
the appropriate suffix or supply pre-rendered text via the underlying
Pydantic ``.model_validate(...)`` API directly.

Both writers emit deterministic output (sorted keys) so two calls to
:func:`dump_deployment` on equal deployments produce byte-identical files,
which simplifies round-trip testing (R10.4) and CI diffing.

Validates: R10.3, R10.5 (via :class:`SensorDeployment` validators), R10.6
(by being the loader the runtime hooks into in task 3.14).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import yaml

from backend.triangulation.schema import SensorDeployment

PathLike = Union[str, Path]

_YAML_SUFFIXES = {".yaml", ".yml"}
_JSON_SUFFIXES = {".json"}


def _classify(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _YAML_SUFFIXES:
        return "yaml"
    if suffix in _JSON_SUFFIXES:
        return "json"
    raise ValueError(
        f"unsupported deployment file extension {suffix!r} for {path}; "
        "expected .yaml, .yml, or .json"
    )


def load_deployment(path: PathLike) -> SensorDeployment:
    """Parse a sensor deployment file from disk.

    The format is sniffed from the file extension. The returned object has
    already passed every Pydantic validator including the production-mode
    clock-source check (R10.5).
    """
    p = Path(path)
    fmt = _classify(p)
    text = p.read_text(encoding="utf-8")
    if fmt == "yaml":
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    if raw is None:
        raise ValueError(f"deployment file {p} is empty")
    if not isinstance(raw, dict):
        raise ValueError(
            f"deployment file {p} must decode to a mapping, got {type(raw).__name__}"
        )
    return SensorDeployment.model_validate(raw)


def dump_deployment(deployment: SensorDeployment, path: PathLike) -> None:
    """Serialize ``deployment`` to ``path`` in YAML or JSON.

    Output keys are sorted at every level so two equal deployments
    serialize to byte-identical files. The write is atomic-ish in the
    sense that ``Path.write_text`` performs a single ``write`` call after
    the full payload has been rendered in memory.
    """
    p = Path(path)
    fmt = _classify(p)
    payload = deployment.model_dump(mode="json")
    if fmt == "yaml":
        text = yaml.safe_dump(payload, sort_keys=True, default_flow_style=False)
    else:
        text = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
