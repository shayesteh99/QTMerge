## QTMerge

A fast and scalable quartet-based method to infer large species trees from a set of gene trees.
QTMerge partitions taxa into overlapping subsets, infers a subtree for each subset using ASTRAL,
and then recursively merges the subtrees using quartet signal from the gene trees.

---

## Installation

**Step 1 — install ASTRAL via bioconda** (required external dependency):

```bash
conda install -c bioconda -c conda-forge astral
```

**Step 2 — install QTMerge**:

```bash
pip install .
```

This installs the `qtmerge` command and all Python dependencies (`treeswift`, `numpy`, `scipy`, `statsmodels`).

> **Note:** ASTRAL is a Java application and cannot be installed via pip. The bioconda package above handles the Java runtime and binary automatically.

---

## Basic usage

```bash
qtmerge -t GENE_TREES -o OUTPUT
```

| Argument | Description |
|---|---|
| `-t / --trees` | Gene tree file (one Newick tree per line) |
| `-o / --outfile` | Output file for the inferred species tree (default: `./temp`) |
| `-m / --min_size` | Minimum subset size; `sqrt` (default) or an integer |
| `-s / --seed` | Random seed (default: 1142) |
| `-v / --verbose` | Print progress information |
| `--no-smart-split` | Use random partitioning instead of clade-aware partitioning |

By default QTMerge uses **smart-split**: it computes pairwise distances from the gene trees and
uses hierarchical clustering to group closely related taxa together before calling ASTRAL on each
subset. This generally produces more accurate subtrees and a better final result. Pass
`--no-smart-split` to fall back to random partitioning.

---

## Example

The `Example/` directory contains 1,000 true gene trees (`truegenetrees`) and the true species
tree (`s_tree.trees`) for a 101-taxon dataset.

Infer a species tree and compare it to the true tree:

```bash
qtmerge -t Example/truegenetrees -o Example/qtmerge_tree.trees
```

The output `Example/qtmerge_tree.trees` can be compared to `Example/s_tree.trees` using any
tree-distance tool (e.g., the Robinson–Foulds distance).

---

## `--place`: add new taxa to an existing backbone

Use `--place` when you already have a backbone tree for most taxa and want to extend it with
additional taxa without re-running the full inference from scratch.

```bash
qtmerge -t GENE_TREES --place BACKBONE TAXA_FILE -o OUTPUT
```

- `BACKBONE` — a file containing the backbone species tree (Newick, one line).
- `TAXA_FILE` — a text file with one taxon label per line; these are the taxa to add.

QTMerge infers an ASTRAL subtree for the listed taxa and then merges it into the backbone using
quartet signal.

**Example** — suppose you have a backbone tree for taxa 0–80 and want to add taxa 81–100:

```bash
# backbone.trees  : species tree for taxa 0–80 (one Newick line)
# new_taxa.txt    : one label per line, e.g.:
#   81
#   82
#   ...
#   100

qtmerge -t Example/truegenetrees \
        --place backbone.trees new_taxa.txt \
        -o extended_tree.trees
```

---

## `--merge`: merge two disjoint partial trees

Use `--merge` when you have two separately inferred trees on **disjoint** taxon sets and want to
combine them into a single species tree.

```bash
qtmerge -t GENE_TREES --merge TREE1 TREE2 -o OUTPUT
```

- `TREE1`, `TREE2` — files each containing one Newick tree; the taxon sets must be disjoint.

**Example** — suppose you have inferred a tree for taxa 0–50 (`part_a.trees`) and a tree for
taxa 51–100 (`part_b.trees`), and you want to combine them:

```bash
qtmerge -t Example/truegenetrees \
        --merge part_a.trees part_b.trees \
        -o merged_tree.trees
```

---

## Options summary

```
usage: qtmerge [-h] -t TREES [-s SEED] [-m MIN_SIZE] [-o OUTFILE] [-v]
               [--no-smart-split]
               [--merge TREE1 TREE2 | --place BACKBONE TAXA]

options:
  -h, --help            show this help message and exit
  -t, --trees           Gene tree file (one Newick tree per line)
  -s, --seed            Random seed (default: 1142)
  -m, --min_size        Minimum subset size: "sqrt" or an integer (default: sqrt)
  -o, --outfile         Output file (default: ./temp)
  -v, --verbose         Print progress information
  --no-smart-split      Use random partitioning (disables clade-aware clustering)
  --merge TREE1 TREE2   Merge two disjoint partial trees
  --place BACKBONE TAXA Add new taxa from TAXA file into BACKBONE tree
```
