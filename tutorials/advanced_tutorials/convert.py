import json
from pathlib import Path

ipynb = "/oak/stanford/groups/akundaje/valehvpa/code/temp_cs221m/pyvene/tutorials/advanced_tutorials/MQNLI_subset.ipynb"
pyfile = Path(ipynb).with_suffix(".py")

with open(ipynb, "r", encoding="utf-8") as f:
    nb = json.load(f)

with open(pyfile, "w", encoding="utf-8") as f:
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue

        lines = []
        for line in cell["source"]:
            stripped = line.lstrip()

            # skip Jupyter magics
            if stripped.startswith("%"):
                continue

            # skip cell magics
            if stripped.startswith("%%"):
                continue

            # skip shell commands
            if stripped.startswith("!"):
                continue

            lines.append(line)

        if not lines:
            continue

        f.write(f"\n# ===== Cell {i} =====\n")
        f.writelines(lines)
        f.write("\n")

print(f"Wrote {pyfile}")