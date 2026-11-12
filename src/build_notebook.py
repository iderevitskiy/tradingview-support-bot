"""Собирает notebook.ipynb из src/notebook_src.py (ячейки разделены строками `# %%`)."""
import re
import sys
from pathlib import Path

import nbformat as nbf

root = Path(__file__).resolve().parent.parent
src = (root / "src" / "notebook_src.py").read_text()
nb = nbf.v4.new_notebook()
for part in re.split(r"^# %%", src, flags=re.M)[1:]:
    head, _, body = part.partition("\n")
    body = body.strip("\n")
    if "[markdown]" in head:
        body = re.sub(r'^r?"""\n?', "", body)
        body = re.sub(r'\n?"""$', "", body)
        nb.cells.append(nbf.v4.new_markdown_cell(body))
    else:
        nb.cells.append(nbf.v4.new_code_cell(body))
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = root / "notebook.ipynb"
nbf.write(nb, out)
print("written", out, len(nb.cells), "cells")

if "--run" in sys.argv:
    from nbclient import NotebookClient
    client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(root)}})
    try:
        client.execute()
    finally:
        for cell in nb.cells:   # служебные предупреждения библиотек в сданном ноутбуке не нужны
            if cell.cell_type == "code":
                cell.outputs = [o for o in cell.outputs if not (o.output_type == "stream" and o.name == "stderr")]
        nbf.write(nb, out)
    print("executed")
