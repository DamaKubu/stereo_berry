from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class V4L2Mode:
    fourcc: str
    width: int
    height: int
    max_fps: float | None


_SIZE_RE = re.compile(r"Size:\s+Discrete\s+(\d+)x(\d+)")
_FMT_RE = re.compile(r"\[(\d+)\]:\s+'([A-Z0-9]{4})'\s+\(")
_INT_RE = re.compile(r"Interval:\s+Discrete\s+([0-9.]+)s\s+\(([0-9.]+)\s+fps\)")


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.stdout


def list_video_devices() -> list[str]:
    devs = sorted(Path("/dev").glob("video*"))
    return [str(p) for p in devs]


def v4l2_list_modes(dev: str) -> list[V4L2Mode]:
    out = _run(["v4l2-ctl", "-d", dev, "--list-formats-ext"])
    modes: list[V4L2Mode] = []

    current_fourcc: str | None = None
    current_size: tuple[int, int] | None = None
    current_max_fps: float | None = None

    def flush() -> None:
        nonlocal current_size, current_max_fps
        if current_fourcc and current_size:
            modes.append(
                V4L2Mode(
                    fourcc=current_fourcc,
                    width=current_size[0],
                    height=current_size[1],
                    max_fps=current_max_fps,
                )
            )
        current_size = None
        current_max_fps = None

    for line in out.splitlines():
        mfmt = _FMT_RE.search(line)
        if mfmt:
            flush()
            current_fourcc = mfmt.group(2)
            continue

        msz = _SIZE_RE.search(line)
        if msz:
            flush()
            current_size = (int(msz.group(1)), int(msz.group(2)))
            continue

        mint = _INT_RE.search(line)
        if mint and current_size is not None:
            fps = float(mint.group(2))
            if current_max_fps is None or fps > current_max_fps:
                current_max_fps = fps

    flush()
    return modes


def pick_max_mode(modes: Iterable[V4L2Mode], prefer_fourcc: str) -> V4L2Mode | None:
    prefer = prefer_fourcc.upper()
    cand = [m for m in modes if m.fourcc.upper() == prefer]
    if not cand:
        return None
    return max(cand, key=lambda m: (m.width * m.height, m.width))
