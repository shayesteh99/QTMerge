#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

from treeswift import read_tree_newick


def read_tree(path):
    text = Path(path).read_text().strip()
    return read_tree_newick(text)


def leaf_set(node):
    return frozenset(leaf.label for leaf in node.traverse_leaves())


def unrooted_splits(tree):
    labels = leaf_set(tree.root)
    n = len(labels)
    splits = set()
    for node in tree.traverse_preorder():
        if node.is_root() or node.is_leaf():
            continue
        side = leaf_set(node)
        other = labels - side
        if len(side) < 2 or len(other) < 2:
            continue
        split = side if len(side) < len(other) else other
        if len(side) == len(other) and tuple(sorted(other)) < tuple(sorted(side)):
            split = other
        splits.add(split)
    return labels, splits


def rf_distance(true_path, inferred_path):
    true_labels, true_splits = unrooted_splits(read_tree(true_path))
    inferred_labels, inferred_splits = unrooted_splits(read_tree(inferred_path))
    if true_labels != inferred_labels:
        missing = sorted(true_labels - inferred_labels)
        extra = sorted(inferred_labels - true_labels)
        raise ValueError(f"leaf sets differ: missing={missing[:5]} extra={extra[:5]}")
    rf = len(true_splits - inferred_splits) + len(inferred_splits - true_splits)
    max_rf = 2 * (len(true_labels) - 3)
    return rf, rf / max_rf if max_rf > 0 else 0.0


def run_qtmerge(repo, trees, output, mode, profile_path, astral_threads):
    cmd = [
        sys.executable,
        str(repo / "infer_trees.py"),
        "-t",
        str(trees),
        "--prune",
        "--profile",
        "--astral_threads",
        str(astral_threads),
        "-o",
        str(output),
    ]
    if mode == "adaptive_placement":
        cmd.extend(["--adaptive_quartets", "placement"])
    elif mode == "adaptive_all":
        cmd.extend([
            "--adaptive_quartets", "all",
            "--adaptive_min_trees", "100",
            "--adaptive_step", "50",
            "--adaptive_margin", "0.20",
            "--adaptive_alpha", "0.01",
        ])
    elif mode == "ghost_rescue":
        cmd.extend([
            "--ghost_rescue", "margin",
            "--ghost_rescue_margin", "0.03",
        ])
    elif mode == "adaptive_all_rescue":
        cmd.extend([
            "--adaptive_quartets", "all",
            "--adaptive_min_trees", "100",
            "--adaptive_step", "50",
            "--adaptive_margin", "0.20",
            "--adaptive_alpha", "0.01",
            "--ghost_rescue", "margin",
            "--ghost_rescue_margin", "0.03",
        ])
    elif mode == "adaptive_all_exact_unreliable":
        cmd.extend([
            "--adaptive_quartets", "all",
            "--adaptive_min_trees", "100",
            "--adaptive_step", "50",
            "--adaptive_margin", "0.20",
            "--adaptive_alpha", "0.01",
            "--adaptive_exact_on_unreliable",
        ])
    elif mode == "adaptive_all_margin020":
        cmd.extend([
            "--adaptive_quartets", "all",
            "--adaptive_min_trees", "100",
            "--adaptive_step", "50",
            "--adaptive_margin", "0.20",
            "--adaptive_alpha", "0.01",
        ])
    elif mode == "adaptive_all_exact_unreliable_rescue":
        cmd.extend([
            "--adaptive_quartets", "all",
            "--adaptive_min_trees", "100",
            "--adaptive_step", "50",
            "--adaptive_margin", "0.20",
            "--adaptive_alpha", "0.01",
            "--adaptive_exact_on_unreliable",
            "--ghost_rescue", "margin",
            "--ghost_rescue_margin", "0.03",
        ])

    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, env=env)
    wall = time.perf_counter() - start
    profile_path.write_text(proc.stderr)
    internal_runtime = None
    for line in proc.stdout.splitlines():
        try:
            internal_runtime = float(line.strip())
        except ValueError:
            pass
    return proc.returncode, wall, internal_runtime, proc.stdout, proc.stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="100")
    parser.add_argument("--replicates", nargs="*", default=[f"{i:02d}" for i in range(1, 21)])
    parser.add_argument("--modes", nargs="*", default=["baseline", "adaptive_all"])
    parser.add_argument("--astral_threads", type=int, default=4)
    parser.add_argument("--outdir", default="/tmp/qtmerge_adaptive_benchmark")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    dataset = repo / args.dataset
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for rep in args.replicates:
        rep_dir = dataset / rep
        trees = rep_dir / "truegenetrees"
        truth = rep_dir / "s_tree.trees"
        if not trees.exists() or not truth.exists():
            rows.append({
                "replicate": rep,
                "mode": "skipped",
                "status": "missing_input_or_truth",
            })
            continue

        for mode in args.modes:
            output = outdir / f"{args.dataset}_{rep}_{mode}.trees"
            profile = outdir / f"{args.dataset}_{rep}_{mode}.profile.txt"
            code, wall, internal, stdout, stderr = run_qtmerge(
                repo, trees, output, mode, profile, args.astral_threads
            )
            row = {
                "replicate": rep,
                "mode": mode,
                "status": "ok" if code == 0 else f"failed_{code}",
                "wall_seconds": f"{wall:.3f}",
                "internal_seconds": "" if internal is None else f"{internal:.3f}",
                "rf": "",
                "normalized_rf": "",
                "output": str(output),
                "profile": str(profile),
            }
            if code == 0:
                rf, nrf = rf_distance(truth, output)
                row["rf"] = str(rf)
                row["normalized_rf"] = f"{nrf:.6f}"
            else:
                (outdir / f"{args.dataset}_{rep}_{mode}.stdout.txt").write_text(stdout)
                (outdir / f"{args.dataset}_{rep}_{mode}.stderr.txt").write_text(stderr)
            rows.append(row)
            print(row, flush=True)

    csv_path = outdir / f"{args.dataset}_benchmark.csv"
    fieldnames = [
        "replicate",
        "mode",
        "status",
        "wall_seconds",
        "internal_seconds",
        "rf",
        "normalized_rf",
        "output",
        "profile",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
