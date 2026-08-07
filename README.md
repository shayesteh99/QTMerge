## QTMerge
A fast and scalable quartet-based method to infer large species trees from a set of gene trees. 

## Setup
QTMerge requires Python 3.9+ and the ASTRAL command-line executable.

Create an isolated Python environment and install the Python dependencies:
```
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install ASTRAL 4 separately and make the `astral4` executable available on your
`PATH`. If it is installed elsewhere, pass its path with `--astral_cmd` or set
`ASTRAL4_CMD`:
```
python infer_trees.py -t Example/truegenetrees --prune -o Example/qtmerge_tree.trees --astral_cmd /path/to/astral4
```

ASTRAL receives the same seed as `--seed` by default. Override it with
`--astral_seed` or `ASTRAL4_SEED` if needed. ASTRAL4 threading can be set with
`--astral_threads` or `ASTRAL4_THREADS`.

If you use `--start_tree`, QTMerge also expects `nw_prune` from Newick Utilities
to be available on your `PATH`.

QTMerge has an experimental repeated quartet-count query cache. To enable it for
benchmarking, set `QTMERGE_COUNT_CACHE_SIZE`, for example:
```
QTMERGE_COUNT_CACHE_SIZE=20000 python infer_trees.py -t Example/truegenetrees --prune -o Example/qtmerge_tree.trees
```

Use `--profile` to print runtime by major step to stderr.

QTMerge uses the Python exact quartet counter by default. There is an
experimental compiled counter that can be built with:
```
python setup.py build_ext --inplace
```

Use `--counter fast` to force the compiled counter, `--counter auto` to use it
when available, or `--counter validate` to compare both counters during a run.

Use `--adaptive_quartets placement` to enable conservative adaptive quartet
counting during placement, or `--adaptive_quartets all` to apply the same
early-stop rule to merge decisions too. The `all` mode can be much faster, but
should be benchmarked for accuracy before production use. The default adaptive
margin is conservative (`--adaptive_margin 0.20`). Tune with
`--adaptive_min_trees`, `--adaptive_step`, `--adaptive_margin`, and
`--adaptive_alpha`.

Two experimental ghost safeguards are also available:
`--adaptive_exact_on_unreliable` reruns unreliable adaptive decisions with exact
counting before ghosting, and `--ghost_rescue margin` allows low-confidence
ghost-producing decisions to proceed when the dominant topology has enough
margin.

## How to run
You can run QTMerge on a set of gene trees using the command below:
```
python infer_trees.py -t [GENE TREE FILE] --prune -o [OUTPUT FILE]
```
### Example
The example contains a set of 1000 true gene trees and the true species tree. You can run QTMerge on this dataset as:

```
python infer_trees.py -t Example/truegenetrees --prune -o Example/qtmerge_tree.trees
```
You can then compare the output of QTMerge to the true species tree `Example/s_tree.trees`.
