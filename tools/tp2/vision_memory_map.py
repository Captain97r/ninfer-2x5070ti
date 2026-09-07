"""Phase 3A-1: exact Vision tensor map from the actual artifact (no estimates).

Walks the qwen3.8-27b nvfp4 artifact object table with tools/artifact/container.py and
aggregates every Vision object (name prefix "vision/") by role, format, and device placement
under the proposed tp2 architectures. Emits research/phase3a-vision-memory.csv and feeds
research/phase3a-vision-memory.md.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from tools.artifact import container

ARTIFACT = Path(r"M:\qwen\Qwen3.8-27b-nvfp4.ninfer")
OUT_CSV = Path("research/phase3a-vision-memory.csv")

WS = Path(__file__).resolve().parents[2]


def main() -> None:
    art = container.Artifact(ARTIFACT)
    rows = []
    total = 0
    for obj in art.objects:
        name = obj.name
        if not name.startswith("vision/"):
            continue
        nbytes = obj.bytes
        total += nbytes
        rows.append(
            {
                "tensor": name,
                "format": obj.format,
                "shape": "x".join(str(d) for d in obj.shape),
                "bytes": nbytes,
                "role": role_of(name),
                "tp2_placement": "replicated (both ranks)",
            }
        )
    rows.sort(key=lambda r: (r["role"], -r["bytes"], r["tensor"]))

    by_role = defaultdict(int)
    by_format = defaultdict(int)
    for row in rows:
        by_role[row["role"]] += row["bytes"]
        by_format[row["format"]] += row["bytes"]

    out = WS / OUT_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(
            {
                "tensor": "TOTAL",
                "format": "",
                "shape": f"{len(rows)} objects",
                "bytes": total,
                "role": "all vision",
                "tp2_placement": "",
            }
        )

    print(f"objects={len(rows)} total_bytes={total} ({total / 1024**2:.2f} MiB)")
    for role, nbytes in sorted(by_role.items(), key=lambda kv: -kv[1]):
        print(f"  {role:<24} {nbytes:>12,} B  {nbytes / 1024**2:9.2f} MiB")
    for fmt, nbytes in sorted(by_format.items(), key=lambda kv: -kv[1]):
        print(f"  [format] {fmt:<28} {nbytes:>12,} B  {nbytes / 1024**2:9.2f} MiB")
    print(f"csv -> {out}")


def role_of(name: str) -> str:
    if "patch_embedding" in name:
        return "patch_embedding"
    if "position_embedding" in name:
        return "position_embedding"
    if "/attention/" in name:
        return "attention"
    if "/mlp/" in name:
        return "mlp"
    if "/merger/" in name:
        return "merger/projector"
    if "/norm" in name:
        return "norm"
    return "other"


if __name__ == "__main__":
    main()
