"""
Quartet topology counter on preprocessed trees.

Idea: preprocess each tree once for O(1) LCA queries (Euler tour + sparse-
table RMQ).  At query time, build the induced subtree on the n target leaves
in O(n log n) using the classical "sort by Euler entry, take LCAs of adjacent
pairs" trick, then run the original O(size-of-tree) counting recursion on a
tree of size <= 2n-1.

Total cost:
    preprocessing  : O(N log N) per tree, once
    per query      : O(n log n) per tree (n = number of selected taxa)

Compared to the original O(N) per query, this is a big win whenever you reuse
the same trees with many different taxa selections (or just whenever n << N).
"""

# ----------------------------------------------------------------------------
# Preprocessing: O(1) LCA via Euler tour + sparse-table RMQ
# ----------------------------------------------------------------------------

class PreprocessedTree:
    """
    Wraps a tree (TreeSwift-style API: tree.root, node.child_nodes(),
    node.is_leaf(), node.label) with structures that make LCA O(1).
    """

    __slots__ = (
        "tree", "tin", "tout", "depth",
        "label_to_node", "_euler", "_first", "_st", "_log",
    )

    def __init__(self, tree):
        self.tree = tree
        self.tin = {}            # node -> preorder entry stamp
        self.tout = {}           # node -> exit stamp (for ancestor test)
        self.depth = {}          # node -> depth from root
        self.label_to_node = {}  # leaf label -> node
        self._euler = []         # Euler tour: nodes in order of every visit
        self._first = {}         # node -> first index in Euler tour
        self._build_euler_tour()
        self._build_sparse_table()

    # --- build ------------------------------------------------------------

    def _build_euler_tour(self):
        root = self.tree.root
        self.depth[root] = 0
        self.tin[root] = 0
        self._first[root] = 0
        self._euler.append(root)
        if root.is_leaf():
            self.label_to_node[root.label] = root

        timer = 1
        # iterative DFS that emits the parent on every return (Euler tour)
        stack = [(root, iter(root.child_nodes()))]
        while stack:
            node, it = stack[-1]
            child = next(it, None)
            if child is None:
                self.tout[node] = timer
                timer += 1
                stack.pop()
                if stack:
                    self._euler.append(stack[-1][0])
            else:
                self.depth[child] = self.depth[node] + 1
                self.tin[child] = timer
                timer += 1
                self._first[child] = len(self._euler)
                self._euler.append(child)
                if child.is_leaf():
                    self.label_to_node[child.label] = child
                stack.append((child, iter(child.child_nodes())))

    def _build_sparse_table(self):
        euler = self._euler
        depth = self.depth
        n = len(euler)

        log = [0] * (n + 1)
        for i in range(2, n + 1):
            log[i] = log[i >> 1] + 1
        self._log = log

        K = log[n] + 1 if n > 0 else 1
        st = [None] * K
        st[0] = list(range(n))
        for k in range(1, K):
            half = 1 << (k - 1)
            length = 1 << k
            prev = st[k - 1]
            cur = [0] * (n - length + 1)
            for i in range(n - length + 1):
                a = prev[i]
                b = prev[i + half]
                cur[i] = a if depth[euler[a]] < depth[euler[b]] else b
            st[k] = cur
        self._st = st

    # --- query ------------------------------------------------------------

    def lca(self, u, v):
        if u is v:
            return u
        l, r = self._first[u], self._first[v]
        if l > r:
            l, r = r, l
        k = self._log[r - l + 1]
        a = self._st[k][l]
        b = self._st[k][r - (1 << k) + 1]
        eu = self._euler
        return eu[a] if self.depth[eu[a]] < self.depth[eu[b]] else eu[b]

# ----------------------------------------------------------------------------
# Usage
# ----------------------------------------------------------------------------
# import treeswift
# trees = [treeswift.read_tree_newick(s) for s in newick_strings]
# pre   = preprocess_trees(trees)            # one-time, O(N log N) per tree
# counts = count_all_topos(pre, taxa_list)   # per query, O(n log n) per tree