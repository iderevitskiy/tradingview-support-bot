"""Сравнение вариантов базы (раздел 9 ноутбука): выполняет ячейки ноутбука с заданным набором доработок.

    python src/ablation.py base            # без доработок, вся оценка -> cache/e2e_base.json
    python src/ablation.py images          # только поиск -> cache/retr_images.json
    python src/ablation.py images,headings
"""
import json, os, re, sys
from pathlib import Path

variant = sys.argv[1]
os.environ["KB_FEATURES"] = "" if variant == "base" else variant
os.environ["MPLBACKEND"] = "Agg"
root = Path(__file__).resolve().parent.parent
os.chdir(root)
stop = "## 6." if variant == "base" else "### 5.2."
ns = {}
for part in re.split(r"^# %%", (root / "src" / "notebook_src.py").read_text(), flags=re.M)[1:]:
    head, _, body = part.partition("\n")
    if "[markdown]" in head:
        if stop in body:
            break
        continue
    if re.search(r"^show\(", body, flags=re.M):      # примеры ответов для сравнения не нужны
        continue
    exec(compile(body, "<cell>", "exec"), ns)
if variant != "base":
    name = "+".join(sorted(ns["FEATURES"]))
    retr = ns["retr"]
    (root / "cache" / f"retr_{name}.json").write_text(json.dumps({
        "retr": {f"{a} | {b}": v for (a, b), v in retr.to_dict("index").items()},
        "docs": int(ns["kb"].id.nunique()), "chunks": len(ns["chunks"])}, ensure_ascii=False, indent=1))
print("done", variant)
