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
    # (color, number, name) of the single class every armor of the off color trains as, or None
    off_class: tuple[int, int, str] | None = None

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PATH) -> "LabelSpec":
        raw = Path(path).read_bytes()
        d = yaml.safe_load(raw)
        labels = {int(k): str(v) for k, v in sorted(d["labels"].items(), key=lambda kv: int(kv[0]))}
        if not labels or min(labels) < 0 or len(set(labels.values())) != len(labels):
            raise ValueError("label ids must be non-negative and label names unique")
        colors = {int(k): str(v) for k, v in d["colors"].items()}
        training = tuple(int(c) for c in d["training_colors"])
        if not set(training) <= set(colors):
            raise ValueError("training_colors must be a subset of colors")
        off = None
        if d.get("off_class") is not None:
            o = d["off_class"]
            off = (int(o["color"]), int(o["number"]), str(o["name"]))
            if off[0] not in colors or off[0] in training:
                raise ValueError("off_class.color must be a color that is not in training_colors")
            if off[1] in labels or off[2] in labels.values():
                raise ValueError("off_class.number and name must not be a regular label")
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
            off_class=off,
        )

    @property
    def label_ids(self) -> tuple[int, ...]:
        """Label ids in ascending order; ids may have gaps (RM27 has no robot 2 or 5)."""
        return tuple(sorted(self.labels))

    @property
    def classes(self) -> tuple[tuple[int, int], ...]:
        """(color, number) of every training class, in class-id order: training colors x labels, then the off class."""
        table = [(c, l) for c in self.training_colors for l in self.label_ids]
        if self.off_class is not None:
            table.append(self.off_class[:2])
        return tuple(table)

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def number_ids(self) -> tuple[int, ...]:
        """Every number an exported model can report: the label ids plus the off class's number."""
        return self.label_ids + ((self.off_class[1],) if self.off_class else ())

    def is_off(self, color: int) -> bool:
        return self.off_class is not None and color == self.off_class[0]

    def trains(self, color: int) -> bool:
        """Whether armors of this color become training targets."""
        return color in self.training_colors or self.is_off(color)

    def valid_pair(self, color: int, label: int) -> bool:
        """A legal (color, label) for canonical data: any color with a label, or the off color with the off number."""
        if color not in self.colors:
            return False
        return label in self.labels or (self.is_off(color) and label == self.off_class[1])

    def number_name(self, number: int) -> str:
        if number in self.labels:
            return self.labels[number]
        if self.off_class is not None and number == self.off_class[1]:
            return self.off_class[2]
        return "?"

    def pair_name(self, color: int, label: int) -> str:
        return f"{self.colors.get(color, '?')}_{self.number_name(label)}"

    def class_id(self, color: int, label: int) -> int:
        if not (self.trains(color) and self.valid_pair(color, label)):
            raise ValueError(f"no training class for color={color} label={label}")
        if self.is_off(color):
            return self.n_classes - 1
        return self.training_colors.index(color) * len(self.labels) + self.label_ids.index(label)

    def from_class_id(self, cid: int) -> tuple[int, int] | None:
        """(color, number) for a merged training class id, or None when the id is out of range."""
        if not 0 <= cid < self.n_classes:
            return None
        return self.classes[cid]

    def class_name(self, cid: int) -> str:
        pair = self.from_class_id(cid)
        return "invalid" if pair is None else self.pair_name(*pair)

    def v(self, key: str):
        return self.validation[key]
