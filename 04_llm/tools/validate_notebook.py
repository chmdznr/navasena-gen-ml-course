#!/usr/bin/env python3
"""Validate a course notebook: valid JSON + every code cell compiles.
Magics/shell lines (starting with ! or %) are treated as `pass`."""
import ast
import json
import re
import sys


def validate(path):
    nb = json.load(open(path, encoding="utf-8"))
    assert nb.get("nbformat") == 4, "nbformat must be 4"
    errs = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        clean = "\n".join(
            "pass" if re.match(r"\s*[!%]", ln) else ln for ln in src.split("\n")
        )
        try:
            ast.parse(clean)
        except SyntaxError as e:
            errs.append(f"  cell {i}: {e}")
    if errs:
        print(f"FAIL {path}\n" + "\n".join(errs))
        return False
    n = sum(c["cell_type"] == "code" for c in nb["cells"])
    print(f"OK   {path}  ({len(nb['cells'])} cells, {n} code)")
    return True


if __name__ == "__main__":
    ok = all(validate(p) for p in sys.argv[1:])
    sys.exit(0 if ok else 1)
