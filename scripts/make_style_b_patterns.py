"""生成可复现的样式 B 七段数字，并保留其他类别的既有图案。"""
from pathlib import Path
import hashlib
import json

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def patterns() -> dict[str, Image.Image]:
    # 使用斜切端部的七段字形，与样式 A 的内置 5×7 字库区别开。
    segments = {
        "a": [(18, 8), (82, 8), (90, 16), (82, 24), (18, 24), (10, 16)],
        "g": [(18, 72), (82, 72), (90, 80), (82, 88), (18, 88), (10, 80)],
        "d": [(18, 136), (82, 136), (90, 144), (82, 152), (18, 152), (10, 144)],
        "f": [(8, 18), (16, 26), (16, 66), (8, 74), (0, 66), (0, 26)],
        "b": [(92, 18), (100, 26), (100, 66), (92, 74), (84, 66), (84, 26)],
        "e": [(8, 86), (16, 94), (16, 134), (8, 142), (0, 134), (0, 94)],
        "c": [(92, 86), (100, 94), (100, 134), (92, 142), (84, 134), (84, 94)],
    }
    result = {}
    for name, active in {"one": "bc", "three": "abgcd", "four": "fgbc"}.items():
        img = Image.new("L", (101, 161), 0)
        draw = ImageDraw.Draw(img)
        for segment in active:
            draw.polygon(segments[segment], fill=255)
        # 仿真器按图案外接框采样；预先裁去空白使资产语义明确。
        result[name] = img.crop(img.getbbox())
    for name in ("sentry", "outpost", "base"):
        result[name] = Image.open(ROOT / "configs/screens/patterns" / f"{name}.png").convert("L")
    return result


def main() -> None:
    target = ROOT / "configs/screens/patterns_b"
    target.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, img in patterns().items():
        path = target / f"{name}.png"
        if path.exists():
            with Image.open(path) as old:
                if old.convert("L").size != img.size or old.convert("L").tobytes() != img.tobytes():
                    raise SystemExit(f"已有图案内容不同，拒绝覆盖：{path}")
        else:
            img.save(path)
        manifest[name] = {"size": list(img.size), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
