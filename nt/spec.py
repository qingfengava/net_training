"""Label spec loader: class tables, validation thresholds, and the hash recorded in every manifest."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "configs" / "label_spec.yaml"


@dataclass(frozen=True)
class LabelSpec:
    version: int
    corner_order: tuple[str, ...]
    corner_definition: str
    pixel_convention: str
    colors: dict[int, str]
    training_colors: tuple[int, ...]
    labels: dict[int, str]
    types: tuple[str, ...]
    validation: dict
    sha256: str

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PATH) -> "LabelSpec":
        raw = Path(path).read_bytes()
        d = yaml.safe_load(raw)
        labels = {int(k): str(v) for k, v in d["labels"].items()}
        if sorted(labels) != list(range(len(labels))):
            raise ValueError("label ids must be contiguous from 0")
        colors = {int(k): str(v) for k, v in d["colors"].items()}
        training = tuple(int(c) for c in d["training_colors"])
        if not set(training) <= set(colors):
            raise ValueError("training_colors must be a subset of colors")
        return cls(
            version=int(d["version"]),
            corner_order=tuple(d["corner_order"]),
            corner_definition=str(d["corner_definition"]),
            pixel_convention=str(d["pixel_convention"]),
            colors=colors,
            training_colors=training,
            labels=labels,
            types=tuple(d["types"]),
            validation=dict(d["validation"]),
            sha256=hashlib.sha256(raw).hexdigest(),
        )

    @property
    def n_classes(self) -> int:
        return len(self.training_colors) * len(self.labels)

    def class_id(self, color: int, label: int) -> int:
        if color not in self.training_colors or label not in self.labels:
            raise ValueError(f"no training class for color={color} label={label}")
        return self.training_colors.index(color) * len(self.labels) + label

    def from_class_id(self, cid: int) -> tuple[int, int] | None:
        """(color, label) for a merged training class id, or None when the id is out of range."""
        if not 0 <= cid < self.n_classes:
            return None
        n = len(self.labels)
        return self.training_colors[cid // n], cid % n

    def class_name(self, cid: int) -> str:
        pair = self.from_class_id(cid)
        return "invalid" if pair is None else f"{self.colors[pair[0]]}_{self.labels[pair[1]]}"

    def v(self, key: str):
        return self.validation[key]
