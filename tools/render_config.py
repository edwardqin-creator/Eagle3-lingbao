#!/usr/bin/env python3
"""Render a SpecForge YAML template from environment variables."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from string import Template


def gpu_yaml(value: str) -> str:
    devices = [part.strip() for part in value.split(",") if part.strip()]
    if not devices:
        raise ValueError("GPU 列表为空")
    return ", ".join(f'"{device}"' for device in devices)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = dict(os.environ)
    values["TRAIN_GPU_YAML"] = gpu_yaml(values["TRAIN_GPU_DEVICES"])
    values["CAPTURE_GPU_YAML"] = gpu_yaml(values["CAPTURE_GPU_DEVICES"])
    rendered = Template(args.template.read_text(encoding="utf-8")).safe_substitute(values)
    unresolved = sorted(set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", rendered)))
    if unresolved:
        raise ValueError("配置仍有未定义变量: " + ", ".join(unresolved))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(rendered, encoding="utf-8")
    os.replace(temp, args.output)
    print(f"已生成训练配置: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
