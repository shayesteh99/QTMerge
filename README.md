## QTMerge
A fast and scalable quartet-based method to infer large species trees from a set of gene trees. 

## How to run
You can run QTMerge on a set of gene trees using the command below:
```
python infer_trees.py -t [GENE TREE FILE] -o [OUTPUT FILE]
```
### Example
The example contains a set of 1000 true gene trees and the true species tree. You can run QTMerge on this dataset as:

```
python infer_trees.py -t Example/truegenetrees -o Example/qtmerge_tree.trees
```
You can then compare the output of QTMerge to the true species tree `Example/s_tree.trees`.
