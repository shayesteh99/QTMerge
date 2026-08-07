import sys
import os
os.environ["PYTHONHASHSEED"] = "0"
import argparse
from collections import OrderedDict
import math
import time
import numpy as np
from treeswift import *
import treeswift
import random
import warnings
warnings.filterwarnings("ignore")
import shutil
import subprocess
import tempfile

from preprocess import PreprocessedTree

try:
	import qtmerge_fast
except ImportError:
	qtmerge_fast = None

ALPHA_QUARTET = 0.05         # significance level
EPSILON_ANOMALY = 0.05       # required margin above 1/3 for the dominant topo
MIN_QUARTETS_FOR_TEST = 1    # only guard against zero-data quartets; the
                             # proportions test handles small-n correctly
                             # (decisive counts like (k,0,0) yield p_val ~= 1
                             # so they're flagged reliable even when k is tiny,
                             # which is what we want for low-discord input)
ASTRAL_CMD = os.environ.get("ASTRAL4_CMD", "astral4")
ASTRAL_SEED = os.environ.get("ASTRAL4_SEED")
ASTRAL_THREADS = int(os.environ.get("ASTRAL4_THREADS", "1"))
COUNT_CACHE_MAXSIZE = int(os.environ.get("QTMERGE_COUNT_CACHE_SIZE", "0"))
ADAPTIVE_QUARTETS = "off"
COUNTER_MODE = os.environ.get("QTMERGE_COUNTER", "python")
ADAPTIVE_MIN_TREES = int(os.environ.get("QTMERGE_ADAPTIVE_MIN_TREES", "500"))
ADAPTIVE_STEP = int(os.environ.get("QTMERGE_ADAPTIVE_STEP", "100"))
ADAPTIVE_MARGIN = float(os.environ.get("QTMERGE_ADAPTIVE_MARGIN", "0.10"))
ADAPTIVE_ALPHA = float(os.environ.get("QTMERGE_ADAPTIVE_ALPHA", "0.01"))
_COUNT_CACHE = OrderedDict()
_COUNT_CACHE_SOURCE_ID = None
PROFILE = False
_PROFILE_TIMINGS = {}
_PROFILE_STACK = []
_PROFILE_COUNTS = {}
_PLACEMENT_DEPTH = 0


def _profile_enter(name):
	if not PROFILE:
		return None
	entry = [name, time.perf_counter(), 0.0]
	_PROFILE_STACK.append(entry)
	return entry


def _profile_exit(name, entry):
	if entry is None:
		return
	elapsed = time.perf_counter() - entry[1]
	self_time = elapsed - entry[2]
	if _PROFILE_STACK and _PROFILE_STACK[-1] is entry:
		_PROFILE_STACK.pop()
	elif entry in _PROFILE_STACK:
		_PROFILE_STACK.remove(entry)
	if _PROFILE_STACK:
		_PROFILE_STACK[-1][2] += elapsed
	calls, inclusive, self_total = _PROFILE_TIMINGS.get(name, (0, 0.0, 0.0))
	_PROFILE_TIMINGS[name] = (calls + 1, inclusive + elapsed, self_total + self_time)


def _profile_count(name, increment=1):
	if not PROFILE:
		return
	_PROFILE_COUNTS[name] = _PROFILE_COUNTS.get(name, 0) + increment


def _print_profile(total_runtime):
	if not PROFILE:
		return
	print("QTMerge profile:", file=sys.stderr)
	print(f"{'step':32s} {'calls':>10s} {'inclusive':>12s} {'self':>12s} {'self_pct':>9s}", file=sys.stderr)
	for name, (calls, inclusive, self_total) in sorted(_PROFILE_TIMINGS.items(), key=lambda item: item[1][2], reverse=True):
		pct = 100.0 * self_total / total_runtime if total_runtime > 0 else 0.0
		print(f"{name:32s} {calls:10d} {inclusive:12.3f} {self_total:12.3f} {pct:8.1f}%", file=sys.stderr)
	if _PROFILE_COUNTS:
		print("\nQTMerge counters:", file=sys.stderr)
		print(f"{'counter':40s} {'count':>12s}", file=sys.stderr)
		for name, count in sorted(_PROFILE_COUNTS.items()):
			print(f"{name:40s} {count:12d}", file=sys.stderr)


def _profiled(name, func):
	def wrapper(*args, **kwargs):
		global _PLACEMENT_DEPTH
		start = _profile_enter(name)
		if name == "find_taxon_placement_new":
			_PLACEMENT_DEPTH += 1
		try:
			return func(*args, **kwargs)
		finally:
			if name == "find_taxon_placement_new":
				_PLACEMENT_DEPTH -= 1
			_profile_exit(name, start)
	return wrapper


def vprint(*x, **kwargs):
	if VERBOSE:
		print(*x, **kwargs)

def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

def __label_tree__(tree_obj, index = 0):
	is_labeled = True
	labels = set()
	for node in tree_obj.traverse_preorder():
		node.edge_length = 1
		if node.is_leaf():
			continue
		if not node.label or node.label in labels or is_number(node.label):
			is_labeled = False
			node.label = 'I' + str(index)
			index += 1
		labels.add(node.label)
	return is_labeled, index

def preprocess_trees(trees):
    """Preprocess a list of trees once; pass the result to count_all_topos."""
    return [PreprocessedTree(t) for t in trees]

def _normal_cdf(x):
	return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _proportions_ztest_smaller(count, nobs, value):
	"""One-sample proportions z-test p-value for alternative='smaller'."""
	if nobs <= 0:
		return 0.0
	proportion = count / nobs
	variance = proportion * (1.0 - proportion) / nobs
	if variance <= 0.0:
		return 1.0 if proportion >= value else 0.0
	z_stat = (proportion - value) / math.sqrt(variance)
	return _normal_cdf(z_stat)

def _require_executable(command, purpose, hint=None):
	executable = command[0] if isinstance(command, (list, tuple)) else command
	if os.path.dirname(executable):
		found = os.path.isfile(executable) and os.access(executable, os.X_OK)
	else:
		found = shutil.which(executable) is not None
	if not found:
		message = f"{purpose} executable '{executable}' was not found. Install it."
		if hint:
			message += f" {hint}"
		raise RuntimeError(message)

def _induced_subtree(pt, leaves):
    """
    Build the induced subtree of `pt.tree` on `leaves` (original tree nodes).

    Returns (preorder_nodes, children_dict, root_node).  Internal nodes of the
    induced tree are exactly the LCAs of adjacent (in preorder) leaf pairs.
    Size is O(n).
    """
    if not leaves:
        return [], {}, None
    if len(leaves) == 1:
        only = leaves[0]
        return [only], {only: []}, only

    tin = pt.tin
    tout = pt.tout

    # Localize LCA components to avoid repeated attribute lookups and method calls
    first = pt._first
    log = pt._log
    st = pt._st
    euler = pt._euler
    depth = pt.depth

    # 1. sort leaves by Euler-tour entry time (== preorder)
    leaves_sorted = sorted(leaves, key=tin.__getitem__)

    # 2. internal nodes of the induced tree = LCAs of consecutive pairs (inlined lca)
    nodes = set(leaves_sorted)
    for i in range(len(leaves_sorted) - 1):
        u = leaves_sorted[i]
        v = leaves_sorted[i + 1]
        if u is not v:
            l, r = first[u], first[v]
            if l > r:
                l, r = r, l
            k = log[r - l + 1]
            a = st[k][l]
            b = st[k][r - (1 << k) + 1]
            nodes.add(euler[a] if depth[euler[a]] < depth[euler[b]] else euler[b])

    # 3. preorder of the induced node set
    preorder = sorted(nodes, key=tin.__getitem__)

    # 4. attach each node to its lowest already-seen ancestor (stack walk).
    #    `top` is an ancestor of `n` iff tin[top] <= tin[n] <= tout[n] <= tout[top].
    children = {n: [] for n in preorder}
    stack = []
    for n in preorder:
        tn = tin[n]
        tn_out = tout[n]
        while stack:
            top = stack[-1]
            if tin[top] <= tn and tn_out <= tout[top]:
                break
            stack.pop()
        if stack:
            children[stack[-1]].append(n)
        stack.append(n)

    return preorder, children, preorder[0]

def copy_subtree(node):
    """Iterative deep copy of the subtree rooted at node."""
    root_copy = Node(label=node.label, edge_length=node.edge_length)
    stack = [(node, root_copy)]
    while stack:
        orig, copy_n = stack.pop()
        for c in orig.children:
            c_copy = Node(label=c.label, edge_length=c.edge_length)
            copy_n.add_child(c_copy)
            stack.append((c, c_copy))
    return root_copy

def copy_tree(tree):
    new_tree = Tree()
    new_tree.root = copy_subtree(tree.root)
    return new_tree

def fast_extract(tree2_pt, labels):
    """Build induced subtree for `labels` in O(K log K) using PreprocessedTree."""
    target_nodes = [tree2_pt.label_to_node[l] for l in labels
                    if l in tree2_pt.label_to_node]
    if len(target_nodes) < 2:
        return tree2_pt.tree.extract_tree_with(labels)
    preorder, children_map, root = _induced_subtree(tree2_pt, target_nodes)
    node_copies = {}
    for n in preorder:
        new_n = Node(label=n.label, edge_length=n.edge_length)
        node_copies[n] = new_n
    for n in preorder:
        new_n = node_copies[n]
        for c in children_map[n]:
            new_n.add_child(node_copies[c])
    result = Tree()
    result.root = node_copies[root]
    return result

def star_tree(taxa, index):
	Leaves = []
	for t in taxa:
		n = Node(label = t, edge_length = 1)
		Leaves.append(n)

	root = Node(label = "I"+str(index), edge_length = 0)
	index += 1
	tree = Tree()
	tree.root = root

	if len(Leaves) < 3:
		for l in Leaves:
			root.add_child(l)
		return tree, index

	parent = Node(label = "I"+str(index), edge_length = 1)
	index += 1
	parent.add_child(Leaves[0])
	parent.add_child(Leaves[1])
	root.add_child(parent)
	root.add_child(Leaves[2])
	return tree, index

def _four_groups(node, working_set, ll_map):
    """Return the (up to 4) leaf-label groups around branch (node -> node.parent).

    For a binary tree these are exactly 4:
      A, B  = the two subtrees directly under `node`
      C     = the sibling subtree(s) of `node` under its parent
      D     = every leaf above the parent (empty when parent is root)

    When parent is the root (common ASTRAL trifurcation), the root's other
    children each become their own group, giving [A, B, C, D] directly.
    When parent is an internal node, C = merged siblings, D = the rest.
    """
    # A, B: one group per child of node
    below = [ll_map[c] for c in node.children]

    parent = node.parent
    if parent.is_root():
        # Each sibling of node under the root is its own group (handles
        # both bifurcating and trifurcating roots cleanly).
        above = [ll_map[c] for c in parent.children if c is not node]
    else:
        # Merge all siblings of node into one group C
        sibling_leaves = []
        for c in parent.children:
            if c is not node:
                sibling_leaves.extend(ll_map[c])
        # D = everything not in A∪B∪C
        covered = set(l for g in below for l in g) | set(sibling_leaves)
        above_parent = [l for l in working_set if l not in covered]
        above = [sibling_leaves, above_parent]

    return below + above


def get_astral_tree_pruned(trees, taxa, index, min_taxa=4):
    """Run ASTRAL on `taxa`, iteratively removing the smallest quadripartition
    around unreliable branches (1 - exp(-l) < 0.05) until no such branches
    remain or fewer than `min_taxa` leaves are left.

    Returns (tree, index, ghost_taxa) where ghost_taxa are the labels removed
    during pruning.  The returned tree always corresponds exactly to the
    surviving (non-ghosted) taxa.

    The original get_astral_tree is preserved and called internally here.
    """
    vprint(*taxa, sep = " ")
    working = list(taxa)
    all_ghosts = set()
    tree = None
    needs_rebuild = False   # True when `working` changed after the last ASTRAL call

    # while len(working) >= min_taxa:
    tree, index, _ = get_astral_tree(trees, working, index)
    # _, index = __label_tree__(tree, index)
    # return tree, index, all_ghosts

    needs_rebuild = False   # tree is now fresh for current `working`

    # Build leaf-label map for quick group lookups
    ll_map, _ = compute_leaf_labels_and_num_leaves(tree)

    # Find all internal branches with 1 - exp(-l) < 0.05
    bad_branches = []
    for n in tree.traverse_internal():
        if n.is_root():
            continue
        l = n.edge_length
        if l and l < 0.05:
        # if l is not None and (1.0 - np.exp(-l)) < 0.1:
            bad_branches.append(n)

        elif n.label and float(n.label) < 0.6:
            bad_branches.append(n)

    for n in bad_branches:
        groups = _four_groups(n, set(working), ll_map)
        smallest = min(groups, key=len)
        all_ghosts.update(smallest)
    

    if len(bad_branches) >= 1:
    	working = [t for t in working if t not in all_ghosts]
    	tree, index, _ = get_astral_tree(trees, working, index)
    vprint(tree)


    vprint(all_ghosts)

    _, index = __label_tree__(tree, index)
    return tree, index, list(all_ghosts)


def _extract_tree_for_astral(tree_or_pt, taxa):
	if isinstance(tree_or_pt, PreprocessedTree):
		return fast_extract(tree_or_pt, taxa)
	return tree_or_pt.extract_tree_with(taxa)


def get_astral_tree(trees, taxa, index = 0):
	# print(*taxa, sep = " ")
	_require_executable(
		ASTRAL_CMD,
		"ASTRAL",
		"Set ASTRAL4_CMD or pass --astral_cmd with the executable path.",
	)
	_prof = _profile_enter("astral_prepare_input")
	f = tempfile.NamedTemporaryFile(mode="w+", delete=False)
	for t in trees:
		pruned = _extract_tree_for_astral(t, taxa)
		f.write(pruned.newick() + "\n")
	f.flush()
	f.close()
	_profile_exit("astral_prepare_input", _prof)

	_prof = _profile_enter("astral_subprocess")
	proc = subprocess.run([
		ASTRAL_CMD,
		"-i", f.name, 
		"--length", "CULength",
		"--seed", str(ASTRAL_SEED),
		"--thread", str(ASTRAL_THREADS),
	], capture_output=True,
	text=True,
	check=True)
	_profile_exit("astral_subprocess", _prof)
	# print(proc.stdout)

	_prof = _profile_enter("astral_parse_output")
	tree = proc.stdout.split("\n")[0]
	# print(tree)
	tree_obj = read_tree_newick(tree)
	_profile_exit("astral_parse_output", _prof)
	# _, index = __label_tree__(tree_obj, index)
	return tree_obj, index, []
	# print(tree)

	nodes = []
	for n in tree_obj.traverse_preorder():
		if n.is_root() or n.is_leaf() or not n.edge_length:
			continue
		l = n.edge_length
		# print(l)
		if 1-np.exp(-l) < 0.05:
			# print(l)
			nodes.append(n)

	ghosts = []
	for n in nodes:
		if n.is_root():
			continue
		sizes = []
		for c in n.child_nodes():
			sizes.append(c.num_nodes(leaves=True, internal=False))

		parent = n.parent
		if parent.is_root():
			sibling = [c for c in parent.child_nodes() if c != n][0]
			for c in sibling.child_nodes():
				sizes.append(c.num_nodes(leaves=True, internal=False))

		else:
			sibling = [c for c in parent.child_nodes() if c != n][0]
			sizes.append(sibling.num_nodes(leaves=True, internal=False))
			sizes.append(len(taxa) - sum(sizes))
		index = np.argmin(sizes)
		if index <2:
			c = n.child_nodes()[index]
			ghosts += [l.label for l in c.traverse_leaves()]
			n.remove_child(c)
			n.contract()
		elif parent.is_root():
			c = sibling.child_nodes()[index-2]
			ghosts += [l.label for l in c.traverse_leaves()]
			sibling.remove_child(c)
			sibling.contract()
		elif index == 2:
			ghosts += [l.label for l in sibling.traverse_leaves()]
			parent.remove_child(sibling)
			parent.contract()
		else:
			parent_labels = [l.label for l in parent.traverse_leaves()]
			ghosts += [l for l in taxa if l not in parent_labels]
			parent.parent = None
			tree_obj.root = parent

	# print(tree_obj)
	_, index = __label_tree__(tree_obj, index)
	# print(tree_obj)
	return tree_obj, index, ghosts


def get_subtree(start_tree, taxa, index):
	_require_executable("nw_prune", "Newick Utilities")
	cmd = ["nw_prune", "-v", start_tree] + [str(t) for t in taxa]

	proc = subprocess.run(
		cmd,
		capture_output=True,
		text=True,
		check=True
	)

	tree = proc.stdout.split("\n")[0]
	tree_obj = read_tree_newick(tree)
	_, index = __label_tree__(tree_obj, index)
	return tree_obj, index

def extract_quartet(trees, taxa):
	if len(taxa) != 4:
		print("not enough taxa!!")
		return
	topos = {}
	for t in trees:
		pruned = t.extract_tree_with(taxa)
		parents = {}
		for l in pruned.traverse_leaves():
			if l.parent not in parents:
				parents[l.parent] = [l.label]
			else:
				parents[l.parent] += [l.label]

		for p in parents:
			if len(parents[p]) == 2:
				if (parents[p][0], parents[p][1]) in topos:
					topos[(parents[p][0], parents[p][1])] += 1
				elif (parents[p][1], parents[p][0]) in topos:
					topos[(parents[p][1], parents[p][0])] += 1
				else:
					rest = [l for l in taxa if l not in parents[p]]
					if (rest[0], rest[1]) in topos:
						topos[(rest[0], rest[1])] += 1
					elif (rest[1], rest[0]) in topos:
						topos[(rest[1], rest[0])] += 1
					else:
						topos[(parents[p][0], parents[p][1])] = 1
	# print(topos)
	p = max(topos, key=topos.get)
	# print([p[0], p[1]], [l for l in taxa if l not in p])
	return [p[0], p[1]], [l for l in taxa if l not in p]
	# return parents[p], [l for l in taxa if l not in parents[p]]

def _add_topos_for_tree(pt, label_assignments, num_quartets):
    ltn = pt.label_to_node
    # Build node→set-index map and target_leaves in one pass using node
    # identity (faster hash than string labels).
    node_to_set = {}
    target_leaves = []
    for label, i in label_assignments:
        node = ltn.get(label)
        if node is not None:
            existing = node_to_set.get(node)
            if existing is None:
                node_to_set[node] = i
                target_leaves.append(node)
            # if already in node_to_set, label appears in multiple sets;
            # keep first assignment (groups are disjoint in normal use)

    if len(target_leaves) < 4:
        return  # no quartet possible

    # 2. induced subtree (size O(n))
    preorder, children, root = _induced_subtree(pt, target_leaves)

    # 3. postorder accumulation of per-set leaf counts
    num_taxa = {n: [0, 0, 0, 0] for n in preorder}
    for n in reversed(preorder):
        kids = children[n]
        if not kids:
            si = node_to_set.get(n)
            if si is not None:
                num_taxa[n][si] = 1
            continue
        counts = num_taxa[n]
        for c in kids:
            cc = num_taxa[c]
            counts[0] += cc[0]
            counts[1] += cc[1]
            counts[2] += cc[2]
            counts[3] += cc[3]

    root_counts = num_taxa[root]

    # 4. main pass — unrolled for the common binary-node case
    nq = num_quartets
    for n in preorder:
        kids = children[n]
        if not kids:
            continue
        n_counts = num_taxa[n]
        out0 = root_counts[0] - n_counts[0]
        out1 = root_counts[1] - n_counts[1]
        out2 = root_counts[2] - n_counts[2]
        out3 = root_counts[3] - n_counts[3]

        if len(kids) == 2:
            # Unrolled binary case: both (a,b) and (b,a) pairs in one block
            a = num_taxa[kids[0]]
            b = num_taxa[kids[1]]
            a0,a1,a2,a3 = a[0],a[1],a[2],a[3]
            b0,b1,b2,b3 = b[0],b[1],b[2],b[3]
            nq[0] += (a0*b1 + b0*a1)*out2*out3
            nq[0] += (a2*a3*b0 + b2*b3*a0)*out1
            nq[0] += (a2*a3*b1 + b2*b3*a1)*out0
            nq[1] += (a0*b2 + b0*a2)*out1*out3
            nq[1] += (a1*a3*b0 + b1*b3*a0)*out2
            nq[1] += (a1*a3*b2 + b1*b3*a2)*out0
            nq[2] += (a0*b3 + b0*a3)*out1*out2
            nq[2] += (a1*a2*b0 + b1*b2*a0)*out3
            nq[2] += (a1*a2*b3 + b1*b2*a3)*out0
        else:
            for c1 in kids:
                a = num_taxa[c1]
                for c2 in kids:
                    if c1 is c2:
                        continue
                    b = num_taxa[c2]
                    nq[0] += a[0]*b[1]*out2*out3 + a[2]*a[3]*b[0]*out1 + a[2]*a[3]*b[1]*out0
                    nq[1] += a[0]*b[2]*out1*out3 + a[1]*a[3]*b[0]*out2 + a[1]*a[3]*b[2]*out0
                    nq[2] += a[0]*b[3]*out1*out2 + a[1]*a[2]*b[0]*out3 + a[1]*a[2]*b[3]*out0

def _adaptive_quartets_enabled():
    return ADAPTIVE_QUARTETS == "all" or (
        ADAPTIVE_QUARTETS == "placement" and _PLACEMENT_DEPTH > 0
    )

def _adaptive_counts_decisive(counts):
    total = sum(counts)
    if total <= 0:
        return False
    sorted_counts = sorted(counts, reverse=True)
    margin = (sorted_counts[0] - sorted_counts[1]) / total
    return margin >= ADAPTIVE_MARGIN and test_p1_equivalence(counts) >= ADAPTIVE_ALPHA

def count_all_topos(preprocessed_trees, taxa_list):
    """
    preprocessed_trees: list returned by preprocess_trees(...)
    taxa_list         : list of 4 collections of taxon labels

    Returns [count(0,1|2,3), count(0,2|1,3), count(0,3|1,2)].
    """
    if len(taxa_list) != 4:
        print("not enough taxa!!")
        return
    if _PLACEMENT_DEPTH > 0:
        _profile_count("count_all_topos_inside_placement")
    else:
        _profile_count("count_all_topos_outside_placement")

    global _COUNT_CACHE_SOURCE_ID
    source_id = id(preprocessed_trees)
    if _COUNT_CACHE_SOURCE_ID is not None and _COUNT_CACHE_SOURCE_ID != source_id:
        _COUNT_CACHE.clear()
    _COUNT_CACHE_SOURCE_ID = source_id

    cache_key = (
        source_id,
        tuple(tuple(group) for group in taxa_list),
    )
    if COUNT_CACHE_MAXSIZE > 0:
        cached = _COUNT_CACHE.get(cache_key)
        if cached is not None:
            _profile_count("count_all_topos_cache_hit")
            _COUNT_CACHE.move_to_end(cache_key)
            return list(cached)
        _profile_count("count_all_topos_cache_miss")

    num_quartets = [0, 0, 0]
    label_assignments = [
        (label, i)
        for i, group in enumerate(taxa_list)
        for label in group
    ]

    adaptive = _adaptive_quartets_enabled()
    min_trees = max(1, min(ADAPTIVE_MIN_TREES, len(preprocessed_trees)))
    step = max(1, ADAPTIVE_STEP)

    use_fast_counter = qtmerge_fast is not None and COUNTER_MODE in ("auto", "fast", "validate")
    if use_fast_counter:
        fast_counts, trees_used = qtmerge_fast.count_all_topos_fast(
            preprocessed_trees,
            taxa_list,
            adaptive,
            min_trees,
            step,
            ADAPTIVE_MARGIN,
            ADAPTIVE_ALPHA,
            EPSILON_ANOMALY,
        )
        if trees_used < len(preprocessed_trees):
            _profile_count("adaptive_quartets_early_stop")
            _profile_count("adaptive_quartets_trees_skipped", len(preprocessed_trees) - trees_used)
        if COUNTER_MODE != "validate":
            num_quartets = fast_counts
            if COUNT_CACHE_MAXSIZE > 0:
                _COUNT_CACHE[cache_key] = tuple(num_quartets)
                if len(_COUNT_CACHE) > COUNT_CACHE_MAXSIZE:
                    _COUNT_CACHE.popitem(last=False)
            return num_quartets

    for tree_index, pt in enumerate(preprocessed_trees, start=1):
        _add_topos_for_tree(pt, label_assignments, num_quartets)
        if (adaptive and tree_index >= min_trees and
            (tree_index == len(preprocessed_trees) or tree_index % step == 0) and
            _adaptive_counts_decisive(num_quartets)):
            _profile_count("adaptive_quartets_early_stop")
            _profile_count("adaptive_quartets_trees_skipped", len(preprocessed_trees) - tree_index)
            break

    if COUNTER_MODE == "validate" and use_fast_counter and list(fast_counts) != num_quartets:
        raise RuntimeError(f"fast counter mismatch: fast={fast_counts} python={num_quartets}")

    if COUNT_CACHE_MAXSIZE > 0:
        _COUNT_CACHE[cache_key] = tuple(num_quartets)
        if len(_COUNT_CACHE) > COUNT_CACHE_MAXSIZE:
            _COUNT_CACHE.popitem(last=False)
    return num_quartets

def is_quartet_reliable(counts,
                        epsilon=EPSILON_ANOMALY,
                        alpha=ALPHA_QUARTET,
                        min_total=MIN_QUARTETS_FOR_TEST):
    """True iff the dominant topology in `counts` is statistically
    distinguishable from being inside the ILS anomaly zone (within `epsilon`
    of 1/3). Mirrors the existing test used in find_taxon_placement.

    Short-circuits we add for noise-free input:
      * counts is None or sum < min_total: not enough data, False.
      * If the dominant count is at least 2x the second-largest (or the
        second-largest is exactly 0), it's decisive — trust it without
        running the proportions test. This is what catches deterministic
        single-tree quartets like (k,0,0) where the test would return a
        big p-value anyway.
    """
    if counts is None:
        return False
    total = sum(counts)
    if total < min_total:
        return False
    sorted_c = sorted(counts, reverse=True)
    if sorted_c[1] == 0 or sorted_c[0] >= 2 * sorted_c[1]:
        return True
    return test_p1_equivalence(counts, epsilon=epsilon) >= alpha


def dominant_topology(counts):
    """Index 0/1/2 of the largest count, with deterministic tie-breaking."""
    if counts[0] >= counts[1] and counts[0] >= counts[2]:
        return 0
    if counts[1] >= counts[0] and counts[1] >= counts[2]:
        return 1
    return 2


def _flatten_taxa(group_or_groups):
    """Helper for ghost collection: accepts a list of labels OR a list of
    lists, returns a flat list of labels. Tree2 subtrees are 'taxa groups'
    in your code (a list of leaf labels under one root child)."""
    out = []
    for x in group_or_groups:
        if isinstance(x, (list, tuple, set)):
            out.extend(x)
        else:
            out.append(x)
    return out

def test_p1_equivalence(counts, epsilon=EPSILON_ANOMALY):
	k = sum(counts)

	if counts[0] >= counts[1] and counts[0] >= counts[2]:
		x1 = counts[0]
	elif counts[1] >= counts[0] and counts[1] >= counts[2]:
		x1 = counts[1]
	else:
		x1 = counts[2]
	return _proportions_ztest_smaller(count=x1, nobs=k, value=1/3+epsilon)


def compute_num_leaves(tree):
	num_leaves = {}
	for n in tree.traverse_postorder():
		if n.is_leaf():
			num_leaves[n] = 1
		else:
			num_leaves[n] = sum(num_leaves[c] for c in n.children)
	return num_leaves


def compute_leaf_labels(tree):
	"""Returns {node: [leaf_labels]} built bottom-up. Avoids repeated traverse_leaves()."""
	leaf_labels = {}
	for n in tree.traverse_postorder():
		if n.is_leaf():
			leaf_labels[n] = [n.label]
		else:
			labels = []
			for c in n.children:
				labels.extend(leaf_labels[c])
			leaf_labels[n] = labels
	return leaf_labels


def compute_leaf_labels_and_num_leaves(tree):
	"""Single-pass version returning (leaf_labels, num_leaves) together."""
	leaf_labels = {}
	num_leaves = {}
	for n in tree.traverse_postorder():
		if n.is_leaf():
			leaf_labels[n] = [n.label]
			num_leaves[n] = 1
		else:
			labels = []
			count = 0
			for c in n.children:
				labels.extend(leaf_labels[c])
				count += num_leaves[c]
			leaf_labels[n] = labels
			num_leaves[n] = count
	return leaf_labels, num_leaves


def find_middle_branch(tree, num_leaves):
	node = tree.root

	if num_leaves[tree.root] <= 4:
		for n in tree.traverse_preorder():
			if not n.is_root() and len(n.children) == 2:
				return n

	while True:
		sizes = {}
		if node.is_root():
			for c in node.children:
				if c.is_leaf():
					sizes[c] = 0
				else:
					sizes[c] = 0
					for cc in c.children:
						# print(cc.label)
						sizes[c] = max(sizes[c], num_leaves[cc])
		elif node.parent.is_root():
			for c in node.children:
				sizes[c] = num_leaves[c]
			for c in tree.root.children:
				if c != node:
					if c.is_leaf():
						sizes[c] = 0
					else:
						for cc in c.children:
							sizes[cc] = num_leaves[cc]
		else:
			for c in node.children:
				sizes[c] = num_leaves[c]
			parent = node.parent
			for c in parent.children:
				if c !=node:
					sizes[c] = num_leaves[c]
			gparent = parent.parent
			sizes[gparent] = num_leaves[tree.root] - sum([sizes[c] for c in sizes])
		max_size = max([sizes[c] for c in sizes])
		largest_is_child = False
		for c in node.children:
			# print(c.label)
			if sizes[c] == max_size:
				largest_is_child = True
				node = c
				break
		if not largest_is_child:
			if node.parent.is_root():
				return node
			return node.parent


def place_taxon(t, tree, num_leaves, s_tree, label):
	node = find_middle_branch(tree, num_leaves)
	prev = None
	while True:
		# print(node.label)
		if node.is_leaf() and prev:
			parent = node.parent
			parent.remove_child(node)
			newparent = Node(label = label, edge_length = 1)
			newleaf = Node(label = t, edge_length = 1)
			newparent.add_child(node)
			newparent.add_child(newleaf)
			parent.add_child(newparent)

			num_leaves[newleaf] = 1
			num_leaves[newparent] = num_leaves[node] + 1
			num_leaves[parent] += 1
			return

		taxa = []
		for c in node.children:
			c_leaves = [l.label for l in c.traverse_leaves()]
			taxa.append(c_leaves)

		parent = node.parent
		for c in parent.children:
			if c != node:
				c_leaves = [l.label for l in c.traverse_leaves()]
				taxa.append(c_leaves)

		taxa.append([t])

		# q = extract_quartet(s_tree, taxa+[t])
		# print(q)
		if [taxa[0],t] in q or [t,taxa[0]] in q:
			dir = "down"
			nextnode = node.children[0]

		elif [taxa[1],t] in q or [t,taxa[1]] in q:
			dir = "down"
			nextnode = node.children[1]

		else:
			if node.parent.is_root():
				dir="down"
				if prev == 'down':
					parent = node.parent
					parent.remove_child(node)
					newparent = Node(label = label, edge_length = 1)
					newleaf = Node(label = t, edge_length = 1)
					newparent.add_child(node)
					newparent.add_child(newleaf)
					parent.add_child(newparent)

					num_leaves[newleaf] = 1
					num_leaves[newparent] = num_leaves[node] + 1
					num_leaves[parent] += 1
					return
				for c in node.parent.children:
					if c != node:
						nextnode = c
						break
			else:
				dir="up"
				nextnode = node.parent
		if prev and prev != dir and not node.parent.is_root():
			parent = node.parent
			parent.remove_child(node)
			newparent = Node(label = label, edge_length = 1)
			newleaf = Node(label = t, edge_length = 1)
			newparent.add_child(node)
			newparent.add_child(newleaf)
			parent.add_child(newparent)

			num_leaves[newleaf] = 1
			num_leaves[newparent] = num_leaves[node] + 1
			num_leaves[parent] += 1
			return
		prev = dir
		node = nextnode


def find_taxon_placement(t, tree, num_leaves, genetrees, test=False, leaf_labels=None):
	node = find_middle_branch(tree, num_leaves)
	if leaf_labels is None:
		leaf_labels = compute_leaf_labels(tree)
	visited = set()
	prev = None
	while True:
		visited.add(node)
		# print(node.label)
		nc = node.children
		if len(nc) == 1 and nc[0] in visited:
			node = node.parent
		elif len(nc) == 1:
			node = nc[0]
		if node.is_leaf() and node.parent in visited:
			return node, 1

		nc = node.children
		taxa = []
		for c in nc:
			taxa.append(leaf_labels[c])

		parent = node.parent
		for c in parent.children:
			if c != node:
				taxa.append(leaf_labels[c])
		# taxa.append([l for l in leaf_labels[tree.root] if l not in leaf_labels[parent]])

		taxa.append([t])
		# q = extract_quartet(genetrees, taxa+[t])
		# print(taxa)
		q = count_all_topos(genetrees, taxa)
		# pval = -1
		if test:
			pval = test_p1_equivalence(q)
			if pval < ALPHA_QUARTET:
				return None, pval
		# print(taxa, q)

		if q[2] > q[1] and q[2] > q[0]:
		# if [taxa[0],t] in q or [t,taxa[0]] in q:
			# dir = "down"
			nextnode = nc[0]

		elif q[1] > q[2] and q[1] > q[0]:
		# elif [taxa[1],t] in q or [t,taxa[1]] in q:
			# dir = "down"
			nextnode = nc[1]

		else:
			if node.parent.is_root():
				# dir="down"
				if node.parent in visited:
				# if prev == 'down':
					return node, 1
				visited.add(node.parent)
				for c in node.parent.children:
					if c != node:
						nextnode = c
						break
			else:
				# dir="up"
				nextnode = node.parent
		# print(nextnode.label)
		# if prev and prev != dir and not node.parent.is_root():
		if nextnode in visited:
			if nextnode == node.parent:
				return node, 1
			return nextnode, 1

		# prev = dir
		node = nextnode



def find_taxon_placement_new(t, tree, num_leaves, genetrees, test=False, leaf_labels=None):
	node = find_middle_branch(tree, num_leaves)
	if leaf_labels is None:
		leaf_labels = compute_leaf_labels(tree)
	root_leaf_labels = leaf_labels[tree.root]
	leaf_label_sets = {n: set(labels) for n, labels in leaf_labels.items()}
	visited = set()
	prev = None
	while True:
		_profile_count("placement_walk_steps")
		visited.add(node)
		vprint(node.label)
		nc = node.children
		if len(nc) == 1 and nc[0] in visited:
			node = node.parent
		elif len(nc) == 1:
			node = nc[0]
		if node.is_leaf() and node.parent in visited:
			return node, 1

		nc = node.children
		taxa = []
		for c in nc:
			taxa.append(leaf_labels[c])

		# parent = node.parent
		# for c in parent.children:
		# 	if c != node:
		# 		taxa.append(leaf_labels[c])
		node_leaf_set = leaf_label_sets[node]
		taxa.append([l for l in root_leaf_labels if l not in node_leaf_set])

		taxa.append([t])
		# q = extract_quartet(genetrees, taxa+[t])
		# print(taxa)
		q = count_all_topos(genetrees, taxa)
		# print(taxa, q)
		pval = -1
		if test:
			pval = test_p1_equivalence(q)
			if pval < ALPHA_QUARTET:
				return None, pval
		vprint(taxa, q, pval)

		if q[2] > q[1] and q[2] > q[0]:
		# if [taxa[0],t] in q or [t,taxa[0]] in q:
			# dir = "down"
			nextnode = nc[0]

		elif q[1] > q[2] and q[1] > q[0]:
		# elif [taxa[1],t] in q or [t,taxa[1]] in q:
			# dir = "down"
			nextnode = nc[1]

		else:
			if node.parent.is_root():
				# dir="down"
				if node.parent in visited:
				# if prev == 'down':
					return node, 1
				visited.add(node.parent)
				for c in node.parent.children:
					if c != node:
						nextnode = c
						break
			else:
				# dir="up"
				nextnode = node.parent
		# print(nextnode.label)
		# if prev and prev != dir and not node.parent.is_root():
		if nextnode in visited:
			if nextnode == node.parent:
				return node, 1
			return nextnode, 1

		# prev = dir
		node = nextnode


def reroot_middle(tree):
	middle_node = None
	num_leaves = compute_num_leaves(tree)
	if len(num_leaves) < 4:
		return tree
	if len(num_leaves) == 7:
		for c in tree.root.children:
			if num_leaves[c] > 2:
				for cc in c.children:
					if len(cc.children) == 2:
						middle_node = cc
						break
		if middle_node is None:
			return tree
	else:
		middle_node = find_middle_branch(tree, num_leaves)
		if num_leaves[middle_node] < 2:
			middle_node = middle_node.parent
		elif num_leaves[middle_node.parent] - num_leaves[middle_node] < 2:
			max_size = max([num_leaves[c] for c in middle_node.children])
			middle_node = [c for c in middle_node.children if num_leaves[c] == max_size][0]

	tree.root.edge_length = None
	root = tree.root
	tree.reroot(middle_node, length = middle_node.edge_length/2)
	root.contract()
	return tree


def create_subtrees(tree, subs, taxa):
	outputs = []
	for i in range(len(subs)):
		t = taxa[(i+1)%len(subs)][0]
		n = subs[i]
		if n[1] == 0:
			node = n[0]
			parent = Node(edge_length=0, label=node.label)
			leaf = Node(edge_length=0, label=t)
			parent.add_child(copy_subtree(node))
			parent.add_child(leaf)
			temp_tree = Tree()
			temp_tree.root = parent
			outputs.append(temp_tree)

		if n[1] == 1:
			node = n[0]
			for c in node.child_nodes():
				node.remove_child(c)
			leaf = Node(edge_length = 0, label = t)
			node.add_child(leaf)
			outputs.append(tree)

	return outputs

def divide_tree(tree, groups):
	outputs = []
	taken = []
	for i in range(len(groups)):
		g = groups[i]
		if len(g) == 1:
			node = g[0]
			node.parent = None
			subtree = Tree()
			subtree.root = node
			outputs.append(subtree)
		elif len(g) == 2:
			label = g[0].parent.label
			if label in taken:
				label = g[1].parent.label
			node = Node(edge_length = 1, label = label)
			taken.append(label)
			subtree = Tree()
			subtree.root = node

			g[0].parent = None
			g[1].parent = None
			node.add_child(g[0])
			node.add_child(g[1])

			# node = g[0].parent
			# node.parent = None
			# subtree = Tree()
			# subtree.root = node
			outputs.append(subtree)

		elif len(g) == 3:
			node = g[0].parent.parent
			for c in node.child_nodes():
				for cc in c.child_nodes():
					if cc not in g:
						if [cc] not in groups[:i]:
							node = cc
							node.parent = None
							subtree = Tree()
							subtree.root = node
							outputs.append(subtree)

							c.remove_child(cc)
							c.contract()
							outputs.append(tree)
							return outputs[::-1]
						else:	
							c.remove_child(cc)
							c.contract()
							outputs.append(tree)
							break

	return outputs



def merge_trees(genetrees, tree1, tree2, placements, ghosts):
    vprint("start merge")
    vprint(tree1)
    vprint(tree2)
    tree1_leaf_labels, tree1_num_leaves = compute_leaf_labels_and_num_leaves(tree1)
    tree2_leaf_count = sum(1 for _ in tree2.traverse_leaves())
    tree2 = reroot_middle(tree2)

    # ---- base cases (unchanged except find_taxon_placement always tests) ----
    if tree1_num_leaves[tree1.root] < 3:
        for n in tree1.traverse_preorder():
            if not n.is_root() and n.edge_length > 0:
                place = n.label
                break
        for l in tree2.traverse_leaves():
            placements[l.label] = place
        return

    if tree2_leaf_count < 4:
        for l in tree2.traverse_leaves():
            place, pval = find_taxon_placement_new(l.label, tree1, tree1_num_leaves,
                                         genetrees, test=True,
                                         leaf_labels=tree1_leaf_labels)
            if place is None:
                vprint([l.label, pval])
                _profile_count("ghost_created_merge_base")
                ghosts.append([l.label, pval])
                continue
            if place.edge_length == 0:
                place = place.parent
            placements[l.label] = place.label
        return

    tree1_copy_root = copy_subtree(tree1.root)
    node = find_middle_branch(tree1, tree1_num_leaves)

    # ---- get 4 representative taxa groups for tree1 ----
    taxa = []
    tree1_subs = []
    for c in node.children:
        taxa.append(tree1_leaf_labels[c])
        tree1_subs.append([c, 0])
    tree1_subs.append([node, 1])
    all_tree1_leaves = tree1_leaf_labels[tree1.root]
    node_leaves_set = set(tree1_leaf_labels[node])
    taxa.append([l for l in all_tree1_leaves if l not in node_leaves_set])
    vprint(taxa)
    vprint([t[0].label for t in tree1_subs])

    # ---- get 4 representative taxa groups for tree2 ----
    tree2_taxa = []
    tree2_subs = []
    tree2_leaf_labels = compute_leaf_labels(tree2)
    for c in tree2.root.children:
        for cc in c.children:
            tree2_taxa.append(tree2_leaf_labels[cc])
            tree2_subs.append([cc, 0])

    vprint(tree2_taxa)
    vprint(tree1_subs)
    vprint("rerooted: ", tree2)

    # ---- assignment step (original logic; no robustness gate) ----
    # is_quartet_reliable cannot live here because count_all_topos returns
    # (0,0,0) on legitimate inputs when create_subtrees' representative-leaf
    # insertions cause label collisions across the 4 groups, and ghosting
    # those would corrupt the recursion structure.
    assignments = []
    p_values = {}
    for t in tree2_taxa:
        to_check = t.copy()
        assignments.append(-1)
        while len(to_check) > 0:
            taxon = to_check.pop(0)
            q = count_all_topos(genetrees, taxa + [[taxon]])
            p_val = test_p1_equivalence(q)
            vprint(taxa + [[taxon]], q, p_val < ALPHA_QUARTET)
            if p_val < ALPHA_QUARTET:
                p_values[taxon] = p_val
                continue
            # if p_val < ALPHA_QUARTET:
            # 	p_values.append(p_val)
            # 	assignments.append(-1)
            elif q[2] >= q[1] and q[2] >= q[0]:
                assignments[-1] = 0
            elif q[1] >= q[2] and q[1] >= q[0]:
                assignments[-1] = 1
            else:
                assignments[-1] = 2
            break

    vprint(assignments)
    counts = {}
    for i, a in enumerate(assignments):
        counts.setdefault(a, []).append(i)

    vprint(counts)

    tree1_subtrees = create_subtrees(tree1, tree1_subs, taxa)

    # ---- "all in one tree1 component" disambiguation ----
    # Exact original logic: do NOT early-bail on an unreliable quartet here.
    # The `is_quartet_reliable` gate would be wrong on this loop because
    # count_all_topos can legitimately return (0,0,0) when create_subtrees'
    # representative leaves cause label collisions across the four groups,
    # and dumping all of tree2 down one branch corrupts the recursion.
    if len(counts) == 1 and len(tree2_taxa) >= 2:
        only = next(iter(counts))
        if only == -1:
            for i in range(4):
                for t in tree2_taxa[i]:
                    _profile_count("ghost_created_merge_all_unassigned")
                    ghosts.append([t, p_values[t]])
            return

        out = [t for i, group in enumerate(taxa) if i != only for t in group]
        which = [1] * len(tree2_taxa)
        for i in range(len(tree2_taxa)):
            rest = [tree2_taxa[j] for j in range(len(tree2_taxa)) if j != i]
            if len(rest) < 3:
                continue
            q = count_all_topos(genetrees, rest + [out])
            if q[0] > q[1] and q[0] > q[2]:
                which[0 + int(i <= 0)] = 0
                which[1 + int(i <= 1)] = 0
            elif q[1] > q[2] and q[1] > q[0]:
                which[0 + int(i <= 0)] = 0
                which[2 + int(i <= 2)] = 0
            elif q[2] > q[1] and q[2] > q[0]:
                which[1 + int(i <= 1)] = 0
                which[2 + int(i <= 2)] = 0

        if sum(which) == 0:
            merge_trees(genetrees, tree1_subtrees[only], tree2,
                        placements, ghosts)
            return

        g = [i for i in range(len(which)) if which[i] == 1][0]
        counts[only].remove(g)
        counts[(only + 1) % 3] = [g]

    # ---- recurse on each (tree1 component, tree2 group) pair ----
    groups = [[tree2_subs[i][0] for i in counts[g]] for g in counts]
    tree2_subtrees = divide_tree(tree2, groups)
    max_group = max(len(counts[g]) for g in counts)

    # print(counts)

    if -1 in counts:
        # if max_group > 1 or len(counts[-1]) > 1:
        for i in counts[-1]:
            # print(i, " added to ghosts")
            for t in tree2_taxa[i]:
                _profile_count("ghost_created_merge_unassigned_group")
                ghosts.append([t, p_values[t]])

        if len(counts[-1]) == 1:
            if max_group == 3:
                for i, g in enumerate(counts):
                    if g != -1:
                        # print(i, "went with ", g)
                        merge_trees(genetrees, tree1_subtrees[g], tree2_subtrees[i],
                            placements, ghosts)
            elif max_group == 2:
                # for i, g in enumerate(counts):
                #     if g == -1:
                #         continue
                #     if len(counts[g]) == 1:
                #         print(i, "went with tree 1")
                #     else:
                #         print(i, "went with ", g)
                for i, g in enumerate(counts):
                    if g == -1:
                        continue
                    if len(counts[g]) == 1:
                        new_tree = Tree()
                        new_tree.root = copy_subtree(tree1_copy_root)
                        merge_trees(genetrees, new_tree, tree2_subtrees[i],
                                    placements, ghosts)
                    else:
                        merge_trees(genetrees, tree1_subtrees[g], tree2_subtrees[i],
                            placements, ghosts)
            else:
                # print(assignments, counts)
                # for i, g in enumerate(counts):
                #     print(i, "went with ", g)
                # i = counts[-1][0]
                # if i % 2 == 0:
                #     g = assignments[i+1]
                # else:
                #     g = assignments[i-1]
                # print(i, "went with ", g)

                for i, g in enumerate(counts):
                    if g == -1:
                        continue
                    merge_trees(genetrees, tree1_subtrees[g], tree2_subtrees[i],
                            placements, ghosts)

                # i = counts[-1][0]
                # if i % 2 == 0:
                #     g = assignments[i+1]
                # else:
                #     g = assignments[i-1]
                # merge_trees(genetrees, tree1_subtrees[g], tree2_subtrees[i],
                #             placements, ghosts)

        else:
            # for i, g in enumerate(counts):
            #     if g != -1:
            #         print(i, "went with tree 1")
            for i, g in enumerate(counts):
                if g != -1:
                    if len(counts[g]) > 1:
                        merge_trees(genetrees, tree1_subtrees[g], tree2_subtrees[i],
                                    placements, ghosts)
                    else:
                        new_tree = Tree()
                        new_tree.root = copy_subtree(tree1_copy_root)
                        merge_trees(genetrees, new_tree, tree2_subtrees[i],
                                    placements, ghosts)

    elif max_group < 3:
        # for i, g in enumerate(counts):
        #     print(i, "went with ", g)
        for i, g in enumerate(counts):
            merge_trees(genetrees, tree1_subtrees[g], tree2_subtrees[i],
                        placements, ghosts)
    else:
        # for i, g in enumerate(counts):
        #     if len(counts[g]) == 1:
        #         print(i, "went with tree 1")
        #     else:
        #         print(i, "went with ", g)

        for i, g in enumerate(counts):
            if len(counts[g]) == 1:
                new_tree = Tree()
                new_tree.root = copy_subtree(tree1_copy_root)
                merge_trees(genetrees, new_tree, tree2_subtrees[i],
                            placements, ghosts)
            else:
                merge_trees(genetrees, tree1_subtrees[g], tree2_subtrees[i],
                        placements, ghosts)


def create_full_tree(rev_placements, tree1, tree2, genetrees, index):
    full_tree = copy_tree(tree1)
    # We'll need to expose ghosts; the original signature doesn't return them,
    # so collect them onto a closure list and have the caller drain it.
    ghosts_collected = []

    tree2_pt = PreprocessedTree(tree2)
    full_tree_ltn = full_tree.label_to_node(selection='all')
    tree1_ltn = tree1.label_to_node(selection='all')
    tree1_leaf_labels = compute_leaf_labels(tree1)
    all_tree1_leaves = tree1_leaf_labels[tree1.root]

    for p in rev_placements:
        vprint(p, rev_placements[p])

        if len(rev_placements[p]) == 1:
            node = full_tree_ltn[p]
            parent = node.parent
            parent.remove_child(node)
            newparent = Node(label='I' + str(index), edge_length=1)
            newleaf = Node(label=rev_placements[p][0], edge_length=1)
            newparent.add_child(node)
            newparent.add_child(newleaf)
            parent.add_child(newparent)
            index += 1

        elif len(rev_placements[p]) == 2:
            node = tree1_ltn[p]
            leaf_down = tree1_leaf_labels[node]
            set_leaves = set(leaf_down)
            leaf_up = [l for l in all_tree1_leaves if l not in set_leaves]
            q = count_all_topos(genetrees,
                                [[i] for i in rev_placements[p]] +
                                [leaf_up, leaf_down])

            # === ROBUSTNESS CHANGE 1 ===
            # Defer placement when the topology of the (anchor, leaf_a, leaf_b)
            # configuration is statistically indistinguishable from random.
            # SAFE for single-tree input: tree1 here is an unmodified ASTRAL
            # backbone (no create_subtrees representative leaves), and the four
            # quartet groups [{leaf_a},{leaf_b},leaf_up,leaf_down] are disjoint,
            # so count_all_topos returns deterministic (k,0,0)-shape counts and
            # is_quartet_reliable returns True — gate is a no-op.
            pval = test_p1_equivalence(q)
            if pval < ALPHA_QUARTET:
                _profile_count("ghost_created_create_full_pair", len(rev_placements[p]))
                ghosts_collected.extend([[p, pval] for p in rev_placements[p]])
                continue

            node = full_tree_ltn[p]
            newleaf1 = Node(label=rev_placements[p][0], edge_length=1)
            newleaf2 = Node(label=rev_placements[p][1], edge_length=1)

            # Original tie-breaking: priority q[0] > q[2] > q[1].
            if q[0] >= q[1] and q[0] >= q[2]:
                newparent = Node(label='I' + str(index), edge_length=1)
                newparent.add_child(newleaf1)
                newparent.add_child(newleaf2)
                index += 1
                parent = node.parent
                parent.remove_child(node)
                newgparent = Node(label='I' + str(index), edge_length=1)
                newgparent.add_child(node)
                newgparent.add_child(newparent)
                parent.add_child(newgparent)
                index += 1
            elif q[2] >= q[1] and q[2] >= q[0]:
                parent = node.parent
                parent.remove_child(node)
                newparent1 = Node(label='I' + str(index), edge_length=1)
                newparent1.add_child(newleaf1)
                newparent1.add_child(node)
                index += 1
                newparent2 = Node(label='I' + str(index), edge_length=1)
                newparent2.add_child(newleaf2)
                newparent2.add_child(newparent1)
                index += 1
                parent.add_child(newparent2)
            else:
                parent = node.parent
                parent.remove_child(node)
                newparent1 = Node(label='I' + str(index), edge_length=1)
                newparent1.add_child(newleaf2)
                newparent1.add_child(node)
                index += 1
                newparent2 = Node(label='I' + str(index), edge_length=1)
                newparent2.add_child(newleaf1)
                newparent2.add_child(newparent1)
                index += 1
                parent.add_child(newparent2)

        else:
            node = tree1_ltn[p]
            leaf_down = tree1_leaf_labels[node]
            set_leaves = set(leaf_down)
            leaf_up = [l for l in all_tree1_leaves if l not in set_leaves]
            subtree = fast_extract(tree2_pt, rev_placements[p])
            subtree_ll, num_leaves = compute_leaf_labels_and_num_leaves(subtree)

            up_node, _ = find_taxon_placement_new(leaf_up[0], subtree, num_leaves,
                                           genetrees, leaf_labels=subtree_ll)
            label = subtree.root.label
            subtree.root.edge_length = None
            root = subtree.root
            subtree.reroot(up_node, length=up_node.edge_length / 2)
            root.contract()
            subtree.root.label = label
            subtree.root.edge_length = 1
            subtree_ll, num_leaves = compute_leaf_labels_and_num_leaves(subtree)
            down_node, _ = find_taxon_placement_new(leaf_down[0], subtree,
                                             num_leaves, genetrees,
                                             leaf_labels=subtree_ll)

            node = full_tree_ltn[p]
            if (up_node == down_node or
                (up_node.parent.is_root() and down_node.parent.is_root())):
                taxa1 = [l.label for l in down_node.traverse_leaves()]
                set_leaves = set(taxa1)
                taxa2 = [l.label for l in subtree.traverse_leaves()
                         if l.label not in set_leaves]
                q = count_all_topos(genetrees,
                                    [leaf_up, leaf_down, taxa1, taxa2])

                # === ROBUSTNESS CHANGE 2 ===
                # Same idea as Change 1: defer all leaves at this anchor when
                # the topology decision is in the ILS anomaly zone. SAFE for
                # single-tree input — all four groups (leaf_up, leaf_down from
                # tree1; taxa1, taxa2 from tree2-extracted subtree) are disjoint
                # and free of create_subtrees representative leaves, so counts
                # are decisive and is_quartet_reliable is True.
                pval = test_p1_equivalence(q)
                if pval < ALPHA_QUARTET:
                    _profile_count("ghost_created_create_full_multi", len(rev_placements[p]))
                    ghosts_collected.extend([[p, pval] for p in rev_placements[p]])
                    continue

                # Original tie-breaking: priority q[0] > q[2] > q[1].
                if q[0] >= q[1] and q[0] >= q[2]:
                    parent = node.parent
                    parent.remove_child(node)
                    newparent = Node(label='I' + str(index), edge_length=1)
                    newparent.add_child(subtree.root)
                    newparent.add_child(node)
                    parent.add_child(newparent)
                    index += 1
                elif q[2] >= q[1] and q[2] >= q[0]:
                    sibling = [c for c in subtree.root.children
                               if c != down_node][0]
                    parent = node.parent
                    parent.remove_child(node)
                    newparent1 = Node(label='I' + str(index), edge_length=1)
                    newparent1.add_child(down_node)
                    newparent1.add_child(node)
                    index += 1
                    newparent2 = Node(label='I' + str(index), edge_length=1)
                    newparent2.add_child(sibling)
                    newparent2.add_child(newparent1)
                    index += 1
                    parent.add_child(newparent2)
                else:
                    sibling = [c for c in subtree.root.children
                               if c != down_node][0]
                    parent = node.parent
                    parent.remove_child(node)
                    newparent1 = Node(label='I' + str(index), edge_length=1)
                    newparent1.add_child(sibling)
                    newparent1.add_child(node)
                    index += 1
                    newparent2 = Node(label='I' + str(index), edge_length=1)
                    newparent2.add_child(down_node)
                    newparent2.add_child(newparent1)
                    index += 1
                    parent.add_child(newparent2)
            else:
                parent = node.parent
                parent.remove_child(node)
                trav = down_node.parent
                trav.remove_child(down_node)
                newparent = Node(label='I' + str(index), edge_length=1)
                newparent.add_child(down_node)
                newparent.add_child(node)
                index += 1
                observed = [down_node]
                while trav is not None:
                    observed.append(trav)
                    child = [c for c in trav.children
                             if c not in observed][0]
                    trav.remove_child(child)
                    newparent2 = Node(label='I' + str(index), edge_length=1)
                    newparent2.add_child(child)
                    newparent2.add_child(newparent)
                    newparent = newparent2
                    trav = trav.parent
                    index += 1
                parent.add_child(newparent)

    # Stash the ghosts collected here so the caller can drain them.
    full_tree._pending_ghosts = ghosts_collected
    return full_tree, index


def infer_tree(leaves, genetrees, index, ghosts=None):
    if ghosts is None:
        ghosts = []
    n = len(leaves)
    if n < 4:
        tree, index = star_tree(leaves, index)
        return tree, index, ghosts

    if n == 4:
        q = count_all_topos(genetrees, [[l] for l in leaves])

        # NEW: if the quartet is uninformative, defer one leaf as a ghost
        # and return a 3-leaf star. The deferred leaf will be re-placed at
        # the end with the full backbone available.
        if not is_quartet_reliable(q):
            # Pick the leaf whose removal leaves the most "balanced" triple.
            # Cheap heuristic: ghost the last leaf. (You can replace this
            # with a smarter rule, e.g. the leaf appearing in the fewest
            # gene trees, but in practice the choice rarely matters because
            # all four are getting deferred to the cleanup pass anyway.)
            ghosts.append(leaves[-1])
            tree, index = star_tree(leaves[:-1], index)
            return tree, index, ghosts

        Nodes = []
        d = dominant_topology(q)
        if d == 0:
            biparts = [[leaves[0], leaves[1]], [leaves[2], leaves[3]]]
        elif d == 1:
            biparts = [[leaves[0], leaves[2]], [leaves[1], leaves[3]]]
        else:
            biparts = [[leaves[0], leaves[3]], [leaves[1], leaves[2]]]
        for bi in biparts:
            n_ = Node(label="I" + str(index), edge_length=1)
            index += 1
            Nodes.append(n_)
            n_.add_child(Node(label=bi[0], edge_length=1))
            n_.add_child(Node(label=bi[1], edge_length=1))
        root = Node(label="I" + str(index), edge_length=0)
        index += 1
        tree = Tree()
        tree.root = root
        for n_ in Nodes:
            root.add_child(n_)
        return tree, index, ghosts

    # ... rest unchanged: divide, recurse, merge_trees, create_full_tree.
    set1 = leaves[:n // 2]
    set2 = leaves[n // 2:]
    tree1, index, ghosts = infer_tree(set1, genetrees, index, ghosts)
    tree2, index, ghosts = infer_tree(set2, genetrees, index, ghosts)
    placements = {}
    merge_trees(genetrees,
                copy_tree(tree1),
                copy_tree(tree2),
                placements, ghosts)
    rev_placements = {}
    for p in placements:
        if placements[p] == tree1.root.children[1].label:
            placements[p] = tree1.root.children[0].label
        rev_placements.setdefault(placements[p], []).append(p)
    full_tree, index = create_full_tree(rev_placements, tree1, tree2,
                                        genetrees, index)
    ghosts.extend(getattr(full_tree, '_pending_ghosts', []))
    return full_tree, index, ghosts



def merge_all_subtrees(input_trees, genetrees, index, ghosts = []):
    if len(input_trees) == 1:
        return input_trees[0], index, ghosts
    new_trees = []
    mid = len(input_trees)//2
    if len(input_trees) % 2 == 1:
        new_trees.append(input_trees[-1])
    # print(len(input_trees))
    for i in range(mid):
        vprint("="*200)
        placements = {}
        tree1 = input_trees[i]
        tree2 = input_trees[i+mid]
        vprint(tree1)
        vprint(tree2)
        merge_trees(genetrees, copy_tree(tree1), copy_tree(tree2), placements, ghosts)
        # print(ghosts)
        vprint(placements)

        rev_placements = {}
        for p in placements:
            if placements[p] == tree1.root.children[1].label:
                placements[p] = tree1.root.children[0].label
            if placements[p] not in rev_placements:
                rev_placements[placements[p]] = []
            rev_placements[placements[p]].append(p)

        vprint(rev_placements)
        full_tree, index = create_full_tree(rev_placements, tree1, tree2, genetrees, index)
        ghosts.extend(getattr(full_tree, '_pending_ghosts', []))
        new_trees.append(full_tree)
        vprint(full_tree)

    return merge_all_subtrees(new_trees, genetrees, index, ghosts)

def main():
	global VERBOSE
	global PROFILE
	global ASTRAL_CMD
	global ASTRAL_SEED
	global ASTRAL_THREADS
	global ADAPTIVE_QUARTETS
	global ADAPTIVE_MIN_TREES
	global ADAPTIVE_STEP
	global ADAPTIVE_MARGIN
	global ADAPTIVE_ALPHA
	global COUNTER_MODE
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-t', '--trees', required=True, help="Input Trees")
	parser.add_argument('-s', '--seed', required=False, default=1142, help="Random Seed")
	parser.add_argument('-m', '--min_size', required=False, default="sqrt", help="Minimum size of each subtree")
	parser.add_argument("-v", "--verbose", action="store_true", help="enable verbose output")
	parser.add_argument('--start_tree', required=False, help="Start tree")
	parser.add_argument('--astral_cmd', default=os.environ.get("ASTRAL4_CMD", "astral4"),
	                    help="ASTRAL executable path or command name")
	parser.add_argument('--astral_seed', default=os.environ.get("ASTRAL4_SEED"),
	                    help="ASTRAL pseudorandom seed; defaults to --seed")
	parser.add_argument('--astral_threads', type=int, default=ASTRAL_THREADS,
	                    help="Number of threads to pass to ASTRAL4")
	parser.add_argument('--profile', action="store_true",
	                    help="Print runtime profiling by major QTMerge step to stderr")
	parser.add_argument('--counter', choices=["auto", "python", "fast", "validate"], default=COUNTER_MODE,
	                    help="Quartet counter implementation")
	parser.add_argument('--adaptive_quartets', choices=["off", "placement", "all"], default=os.environ.get("QTMERGE_ADAPTIVE_QUARTETS", "off"),
	                    help="Use adaptive early-stopping for quartet counts")
	parser.add_argument('--adaptive_min_trees', type=int, default=ADAPTIVE_MIN_TREES,
	                    help="Minimum gene trees to count before adaptive early-stopping")
	parser.add_argument('--adaptive_step', type=int, default=ADAPTIVE_STEP,
	                    help="Adaptive quartet-count checkpoint interval")
	parser.add_argument('--adaptive_margin', type=float, default=ADAPTIVE_MARGIN,
	                    help="Minimum dominant-vs-runner-up count margin fraction for adaptive early-stopping")
	parser.add_argument('--adaptive_alpha', type=float, default=ADAPTIVE_ALPHA,
	                    help="Minimum conservative p-value for adaptive early-stopping")
	parser.add_argument('--prune', action="store_true",
	                    help="Use pruned ASTRAL: iteratively remove the smallest "
	                         "quadripartition of unreliable branches (1-exp(-l)<0.05) "
	                         "before merging; removed taxa become ghosts")
	# parser.add_argument('-i', '--input', required=True, help="Input Trees")
	# parser.add_argument('-g', '--gene_trees', required=True, help="Gene tree file")
	# parser.add_argument('-a', '--annot', required=False, help="Annotation file")
	# parser.add_argument('-n', '--num_genes', required=False, default=1, help="Number of gene trees")

	parser.add_argument('-o', '--outfile', required=False, default='./temp', help="Output file")

	args = parser.parse_args()
	VERBOSE = args.verbose
	PROFILE = args.profile
	ASTRAL_CMD = args.astral_cmd
	ASTRAL_SEED = args.astral_seed if args.astral_seed is not None else args.seed
	ASTRAL_THREADS = args.astral_threads
	COUNTER_MODE = args.counter
	ADAPTIVE_QUARTETS = args.adaptive_quartets
	ADAPTIVE_MIN_TREES = args.adaptive_min_trees
	ADAPTIVE_STEP = args.adaptive_step
	ADAPTIVE_MARGIN = args.adaptive_margin
	ADAPTIVE_ALPHA = args.adaptive_alpha

	random.seed(a=int(args.seed))
	np.random.seed(int(args.seed))
	if args.min_size == "sqrt":
		m = 10
	else:
		m = int(args.min_size)

	# print("+"*300)

	start = time.time()

	_prof = _profile_enter("input_read_parse")
	with open(args.trees, "r") as f:
		trees = f.readlines()
		trees = [read_tree_newick(t) for t in trees]
	_profile_exit("input_read_parse", _prof)
	preprocessed = preprocess_trees(trees)

	_prof = _profile_enter("collect_and_label_leaves")
	leaves = set()
	for t in trees:
		__label_tree__(t)
		leaves |= set([l.label for l in t.traverse_leaves()])
	leaves = sorted(leaves)
	_profile_exit("collect_and_label_leaves", _prof)


	if args.min_size == "sqrt":
		m = max(m, int(np.sqrt(len(leaves))))

	vprint("+" * 300)
	vprint("Creating Subtrees")

	random.shuffle(leaves)
	num_trees = len(leaves) // m
	ghosts = []
	if num_trees <= 1:
		if args.start_tree:
			inferred_tree = read_tree_newick(args.start_tree)
		else:
			inferred_tree, _, ghosts = get_astral_tree(preprocessed, leaves)
			# _, index = __label_tree__(inferred_tree, index)
	else:
		_prof = _profile_enter("create_initial_subtrees")
		index = 0
		input_trees = []
		if args.prune:
			# Dynamic-pool approach: randomly pick m taxa each round, run
			# get_astral_tree_pruned, and put the pruned-off ghosts back into
			# the pool so they get another chance in a later ASTRAL batch.
			# Only the ghosts produced by the LAST batch (when the pool is
			# exhausted) are kept as permanent ghosts for end-placement.
			pool = list(leaves)   # already shuffled
			while len(pool) > m:
				batch = random.sample(pool, m)
				batch_set = set(batch)
				pool = [t for t in pool if t not in batch_set]
				tree, index, g = get_astral_tree_pruned(preprocessed, batch, index)
				_profile_count("ghost_created_prune_reentered", len(g))
				pool = g + pool   # pruned taxa re-enter pool for future batches
				# ghosts += g
				input_trees.append(tree)
			# Last batch: remaining taxa; ghosts here are permanent
			if pool:
				if len(pool) < 4:
					_profile_count("ghost_created_initial_small_pool", len(pool))
					ghosts += [[p, 1] for p in pool]
				else:
					tree, index, g = get_astral_tree_pruned(preprocessed, pool, index)
					_profile_count("ghost_created_prune_final", len(g))
					ghosts += [[p, 1] for p in g]
					input_trees.append(tree)
		else:
			# Original fixed-subset approach (unchanged)
			taxa_subsets = [[] for _ in range(num_trees)]
			for i in range(len(leaves)):
				taxa_subsets[i % num_trees].append(leaves[i])
			for i in range(num_trees):
				if args.start_tree:
					tree, index = get_subtree(args.start_tree, taxa_subsets[i], index)
				else:
					tree, index, g = get_astral_tree(preprocessed, taxa_subsets[i], index)
					_, index = __label_tree__(tree, index)
					_profile_count("ghost_created_initial_astral", len(g))
					ghosts += g
				# print(tree)
				input_trees.append(tree)
		# return
		_profile_exit("create_initial_subtrees", _prof)

		vprint("+" * 300)
		vprint("Merging Subtrees")
		inferred_tree, index, ghosts = merge_all_subtrees(input_trees, preprocessed, index, ghosts)

	# inferred_tree, index, ghosts = infer_tree(leaves, trees, index = 0)
	vprint("+" * 300)
	vprint("Adding Ghost Taxa")
	vprint(ghosts)
	vprint(inferred_tree)

	ghosts = sorted(ghosts, key=lambda t: t[1])[::-1]
	_profile_count("ghost_queue_before_reliable", len(ghosts))
	# ghosts = [g[0] for g in ghosts]
	# print(len(ghosts))

	num_leaves = compute_num_leaves(inferred_tree)

	repeat = True
	_prof = _profile_enter("ghost_placement_reliable")
	while repeat and len(ghosts) > 0: 
		repeat = False
	# for l in ghosts:
		n = len(ghosts)
		_profile_count("ghost_reliable_passes")
		_profile_count("ghost_reliable_pass_queue_total", n)
		for _ in range(n):
			l = ghosts.pop(0)
			_profile_count("ghost_reliable_attempts")
			vprint("=" * 300)
			taxa = l[0]
			pval = l[1]
			vprint(taxa, pval)
			node, pval = find_taxon_placement_new(taxa, inferred_tree, num_leaves, preprocessed, test = True)
			if not node:
				_profile_count("ghost_reliable_retries")
				ghosts.append([taxa,pval])
				continue
			_profile_count("ghost_reliable_placed")
			repeat = True
			vprint(node.label)
			parent = node.parent
			parent.remove_child(node)
			newparent = Node(label = "I"+str(index), edge_length = 1)
			index+=1
			newleaf = Node(label = taxa, edge_length = 1)
			newparent.add_child(node)
			newparent.add_child(newleaf)
			parent.add_child(newparent)

			num_leaves[newleaf] = 1
			num_leaves[newparent] = num_leaves[node] + 1
			num_leaves[parent] += 1

			vprint(inferred_tree)
	_profile_exit("ghost_placement_reliable", _prof)

	vprint("placing the rest:")
	# print(ghosts)
	ghosts = sorted(ghosts, key=lambda t: t[1])[::-1]
	_profile_count("ghost_queue_before_forced", len(ghosts))
	# print(ghosts)
	if len(ghosts) > 0:
		_prof = _profile_enter("ghost_placement_forced")
		vprint(len(ghosts))
		for l in ghosts:
			_profile_count("ghost_forced_attempts")
			vprint("=" * 300)
			taxa = l[0]
			pval = l[1]
			vprint(taxa, pval)
			node, pval = find_taxon_placement_new(taxa, inferred_tree, num_leaves, preprocessed, test = False)
			_profile_count("ghost_forced_placed")
			vprint(node.label)
			parent = node.parent
			parent.remove_child(node)
			newparent = Node(label = "I"+str(index), edge_length = 1)
			index+=1
			newleaf = Node(label = taxa, edge_length = 1)
			newparent.add_child(node)
			newparent.add_child(newleaf)
			parent.add_child(newparent)

			num_leaves[newleaf] = 1
			num_leaves[newparent] = num_leaves[node] + 1
			num_leaves[parent] += 1

			vprint(inferred_tree)
		_profile_exit("ghost_placement_forced", _prof)

	_prof = _profile_enter("write_output")
	inferred_tree.write_tree_newick(args.outfile)
	_profile_exit("write_output", _prof)
	# print(inferred_tree)

	end = time.time()

	print(end - start)
	_print_profile(end - start)



preprocess_trees = _profiled("preprocess_trees", preprocess_trees)
copy_tree = _profiled("copy_tree", copy_tree)
fast_extract = _profiled("fast_extract", fast_extract)
get_astral_tree_pruned = _profiled("get_astral_tree_pruned", get_astral_tree_pruned)
get_astral_tree = _profiled("get_astral_tree", get_astral_tree)
get_subtree = _profiled("get_subtree", get_subtree)
count_all_topos = _profiled("count_all_topos", count_all_topos)
compute_num_leaves = _profiled("compute_num_leaves", compute_num_leaves)
compute_leaf_labels = _profiled("compute_leaf_labels", compute_leaf_labels)
compute_leaf_labels_and_num_leaves = _profiled("compute_leaf_labels_and_num_leaves", compute_leaf_labels_and_num_leaves)
find_taxon_placement_new = _profiled("find_taxon_placement_new", find_taxon_placement_new)
reroot_middle = _profiled("reroot_middle", reroot_middle)
create_subtrees = _profiled("create_subtrees", create_subtrees)
divide_tree = _profiled("divide_tree", divide_tree)
merge_trees = _profiled("merge_trees", merge_trees)
create_full_tree = _profiled("create_full_tree", create_full_tree)
merge_all_subtrees = _profiled("merge_all_subtrees", merge_all_subtrees)


if __name__ == "__main__":
	main()
