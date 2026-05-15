import sys
import os
os.environ["PYTHONHASHSEED"] = "0"
import argparse
import time
import numpy as np
from treeswift import *
import treeswift
import random
import warnings
from statsmodels.stats.proportion import proportions_ztest
warnings.filterwarnings("ignore")
import subprocess
import tempfile

from preprocess import PreprocessedTree

ALPHA_QUARTET = 0.05         # significance level
EPSILON_ANOMALY = 0.05       # required margin above 1/3 for the dominant topo
MIN_QUARTETS_FOR_TEST = 1    # only guard against zero-data quartets; the
                             # proportions test handles small-n correctly
                             # (decisive counts like (k,0,0) yield p_val ~= 1
                             # so they're flagged reliable even when k is tiny,
                             # which is what we want for low-discord input)


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

def get_astral_tree(trees, taxa, index = 0):
	# print(taxa)
	f = tempfile.NamedTemporaryFile(mode="w+", delete=False)
	for t in trees:
		pruned = t.extract_tree_with(taxa)
		f.write(pruned.newick() + "\n")
	f.flush()
	f.close()

	proc = subprocess.run([
		"astral4",
		"-i", f.name
	], capture_output=True,
	text=True,
	check=True)

	tree = proc.stdout.split("\n")[0]
	tree_obj = read_tree_newick(tree)
	# print(tree_obj)
	_, index = __label_tree__(tree_obj, index)
	# print(tree_obj)
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

def count_all_topos(preprocessed_trees, taxa_list):
    """
    preprocessed_trees: list returned by preprocess_trees(...)
    taxa_list         : list of 4 collections of taxon labels

    Returns [count(0,1|2,3), count(0,2|1,3), count(0,3|1,2)].
    """
    if len(taxa_list) != 4:
        print("not enough taxa!!")
        return

    num_quartets = [0, 0, 0]

    for pt in preprocessed_trees:
        ltn = pt.label_to_node
        # Build node→set-index map and target_leaves in one pass using node
        # identity (faster hash than string labels).
        node_to_set = {}
        target_leaves = []
        for i, group in enumerate(taxa_list):
            for label in group:
                node = ltn.get(label)
                if node is not None:
                    existing = node_to_set.get(node)
                    if existing is None:
                        node_to_set[node] = i
                        target_leaves.append(node)
                    # if already in node_to_set, label appears in multiple sets;
                    # keep first assignment (groups are disjoint in normal use)

        if len(target_leaves) < 4:
            continue  # no quartet possible

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

def test_p1_equivalence(counts, epsilon=0.05):
	k = sum(counts)

	if counts[0] >= counts[1] and counts[0] >= counts[2]:
		x1 = counts[0]
	elif counts[1] >= counts[0] and counts[1] >= counts[2]:
		x1 = counts[1]
	else:
		x1 = counts[2]
	z_stat, p_val = proportions_ztest(count=x1, nobs=k, value=1/3+epsilon, alternative='smaller')
	return p_val


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
			return node

		nc = node.children
		taxa = []
		for c in nc:
			taxa.append(leaf_labels[c])

		parent = node.parent
		for c in parent.children:
			if c != node:
				taxa.append(leaf_labels[c])

		taxa.append([t])
		# q = extract_quartet(genetrees, taxa+[t])
		# print(taxa)
		q = count_all_topos(genetrees, taxa)
		if test:
			pval = test_p1_equivalence(q)
			if pval < 0.05:
				return None
		# print(q)

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
					return node
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
				return node
			return nextnode

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
            place = find_taxon_placement(l.label, tree1, tree1_num_leaves,
                                         genetrees, test=True,
                                         leaf_labels=tree1_leaf_labels)
            if place is None:
                ghosts.append(l.label)
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

    # ---- get 4 representative taxa groups for tree2 ----
    tree2_taxa = []
    tree2_subs = []
    tree2_leaf_labels = compute_leaf_labels(tree2)
    for c in tree2.root.children:
        for cc in c.children:
            tree2_taxa.append(tree2_leaf_labels[cc])
            tree2_subs.append([cc, 0])

    vprint("rerooted: ", tree2)

    # ---- assignment step (original logic; no robustness gate) ----
    # is_quartet_reliable cannot live here because count_all_topos returns
    # (0,0,0) on legitimate inputs when create_subtrees' representative-leaf
    # insertions cause label collisions across the 4 groups, and ghosting
    # those would corrupt the recursion structure.
    assignments = []
    for t in tree2_taxa:
        q = count_all_topos(genetrees, taxa + [t])
        if q[2] >= q[1] and q[2] >= q[0]:
            assignments.append(0)
        elif q[1] >= q[2] and q[1] >= q[0]:
            assignments.append(1)
        else:
            assignments.append(2)

    counts = {}
    for i, a in enumerate(assignments):
        counts.setdefault(a, []).append(i)

    tree1_subtrees = create_subtrees(tree1, tree1_subs, taxa)

    # ---- "all in one tree1 component" disambiguation ----
    # Exact original logic: do NOT early-bail on an unreliable quartet here.
    # The `is_quartet_reliable` gate would be wrong on this loop because
    # count_all_topos can legitimately return (0,0,0) when create_subtrees'
    # representative leaves cause label collisions across the four groups,
    # and dumping all of tree2 down one branch corrupts the recursion.
    if len(counts) == 1 and len(tree2_taxa) >= 2:
        only = next(iter(counts))
        out = [t for i, group in enumerate(taxa) if i != only for t in group]
        which = [1] * len(tree2_taxa)
        for i in range(len(tree2_taxa)):
            rest = [tree2_taxa[j] for j in range(len(tree2_taxa)) if j != i]
            if len(rest) < 3:
                continue
            q = count_all_topos(genetrees, rest + [out])
            # Strict '>' to match original — on a tie we leave `which` alone.
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

    if max_group < 3:
        for i, g in enumerate(counts):
            merge_trees(genetrees, tree1_subtrees[g], tree2_subtrees[i],
                        placements, ghosts)
    else:
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
    full_tree = read_tree_newick(tree1.newick())
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
            if not is_quartet_reliable(q):
                ghosts_collected.extend(rev_placements[p])
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

            up_node = find_taxon_placement(leaf_up[0], subtree, num_leaves,
                                           genetrees, leaf_labels=subtree_ll)
            label = subtree.root.label
            subtree.root.edge_length = None
            root = subtree.root
            subtree.reroot(up_node, length=up_node.edge_length / 2)
            root.contract()
            subtree.root.label = label
            subtree.root.edge_length = 1
            subtree_ll, num_leaves = compute_leaf_labels_and_num_leaves(subtree)
            down_node = find_taxon_placement(leaf_down[0], subtree,
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
                if not is_quartet_reliable(q):
                    ghosts_collected.extend(rev_placements[p])
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
                read_tree_newick(tree1.newick()),
                read_tree_newick(tree2.newick()),
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
		merge_trees(genetrees, read_tree_newick(tree1.newick()), read_tree_newick(tree2.newick()), placements, ghosts)
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
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-t', '--trees', required=True, help="Input Trees")
	parser.add_argument('-s', '--seed', required=False, default=1142, help="Random Seed")
	parser.add_argument('-m', '--min_size', required=False, default="sqrt", help="Minimum size of each subtree")
	parser.add_argument("-v", "--verbose", action="store_true", help="enable verbose output")
	# parser.add_argument('-i', '--input', required=True, help="Input Trees")
	# parser.add_argument('-g', '--gene_trees', required=True, help="Gene tree file")
	# parser.add_argument('-a', '--annot', required=False, help="Annotation file")
	# parser.add_argument('-n', '--num_genes', required=False, default=1, help="Number of gene trees")

	parser.add_argument('-o', '--outfile', required=False, default='./temp', help="Output file")

	args = parser.parse_args()
	global VERBOSE
	VERBOSE = args.verbose

	random.seed(a=int(args.seed))
	np.random.seed(int(args.seed))
	if args.min_size == "sqrt":
		m = 10
	else:
		m = int(args.min_size)

	# print("+"*300)

	start = time.time()

	with open(args.trees, "r") as f:
		trees = f.readlines()
		trees = [read_tree_newick(t) for t in trees]
		preprocessed = preprocess_trees(trees) 

	# stree = read_tree_newick("((((65:0.05020870,(((((53:0.01578979,83:0.01578979):0.00767264,9:0.02346243):0.00456874,28:0.02803117):0.01649611,((62:0.02399946,17:0.02399946):0.01252535,(67:0.02908889,85:0.02908889):0.00743592):0.00800246):0.00308367,((((((38:0.02546873,68:0.02546873):0.00572046,((8:0.01187262,75:0.01187262):0.01673997,(48:0.00538162,14:0.00538162):0.02323098):0.00257660):0.00277513,81:0.03396432):0.00251264,(96:0.03554466,72:0.03554466):0.00093230):0.00997261,(87:0.03849819,(56:0.03389305,((63:0.00586389,66:0.00586389):0.01315401,(76:0.01847970,(52:0.00041249,18:0.00041249):0.01806721):0.00053819):0.01487515):0.00460514):0.00795138):0.00023663,((57:0.01969415,2:0.01969415):0.02116783,(49:0.03098632,44:0.03098632):0.00987566):0.00582421):0.00092475):0.00259775):0.00085173,((((((((22:0.03052066,16:0.03052066):0.00333398,93:0.03385465):0.00773149,((82:0.02236742,33:0.02236742):0.01618609,((92:0.01480158,88:0.01480158):0.00773985,(64:0.01642121,78:0.01642121):0.00612022):0.01601208):0.00303263):0.00411416,71:0.04570030):0.00009290,((((59:0.00116533,1:0.00116533):0.01051930,(73:0.01016602,34:0.01016602):0.00151861):0.00312740,7:0.01481204):0.02978257,(((95:0.02959739,(24:0.01473318,23:0.01473318):0.01486421):0.00053564,(12:0.02539142,((41:0.00397066,90:0.00397066):0.01146564,27:0.01543630):0.00995513):0.00474160):0.0143147,(((11:0.00838973,69:0.00838973):0.02280229,(86:0.02378535,3:0.02378535):0.00740667):0.00909810,(((25:0.02780867,55:0.02780867):0.00819224,((43:0.02814893,(46:0.01003381,(21:0.00900962,13:0.00900962):0.00102419):0.01811512):0.00122510,((6:0.00560383,79:0.00560383):0.01007986,29:0.01568369):0.01369034):0.00662688):0.00223613,(51:0.02065393,40:0.02065393):0.01758311):0.00205308):0.00415759):0.00014690):0.00119858):0.00161079,(74:0.04337825,((89:0.03569711,(37:0.02853480,(26:0.00249619,84:0.00249619):0.02603861):0.00716230):0.00480993,((((61:0.00333772,50:0.00333772):0.01837642,39:0.02171414):0.01313987,(31:0.02513686,32:0.02513686):0.00971715):0.00027384,45:0.03512785):0.00537918):0.00287122):0.00402574):0.00189301,(((77:0.00418433,30:0.00418433):0.02798003,(15:0.03001031,((80:0.01436022,97:0.01436022):0.00675751,20:0.02111772):0.00889259):0.00215405):0.01665062,(((((47:0.00607741,54:0.00607741):0.00674951,10:0.01282692):0.00808803,60:0.02091495):0.00636287,(19:0.01966193,4:0.01966193):0.00761588):0.00188079,((35:0.01742653,36:0.01742653):0.00600927,58:0.02343580):0.00572280):0.01965638):0.00048202):0.00037147,((70:0.01429235,94:0.01429235):0.02971623,(42:0.02921620,91:0.02921620):0.01479238):0.00565989):0.00139196):0.00008767,(100:0.03607388,(99:0.03204718,98:0.03204718):0.00402671):0.01507422):0.05114810,0:0.10229621);")
	# __label_tree__(stree)
	# print(stree)

	# num_leaves = compute_num_leaves(stree)
	# place = find_taxon_placement('5', stree, num_leaves, trees)
	# print(place.label)
	# return

	leaves = set()
	for t in trees:
		__label_tree__(t)
		leaves |= set([l.label for l in t.traverse_leaves()])
	leaves = [l for l in leaves]

	if args.min_size == "sqrt":
		m = max(m, int(np.sqrt(len(leaves))))
	vprint(m)

	random.shuffle(leaves)
	num_trees = len(leaves) // m
	if num_trees <= 1:
		inferred_tree, _ = get_astral_tree(trees, leaves)
		ghosts = []
	else:
		index = 0
		taxa_subsets = [[] for _ in range(num_trees)]
		input_trees = []
		for i in range(len(leaves)):
			taxa_subsets[i % num_trees].append(leaves[i])
		for i in range(num_trees):
			tree, index = get_astral_tree(trees, taxa_subsets[i], index)
			input_trees.append(tree)

		inferred_tree, index, ghosts = merge_all_subtrees(input_trees, preprocessed, index)

	# inferred_tree, index, ghosts = infer_tree(leaves, trees, index = 0)
	vprint(ghosts)
	vprint(inferred_tree)

	num_leaves = compute_num_leaves(inferred_tree)
	for l in ghosts:
		node = find_taxon_placement(l, inferred_tree, num_leaves, preprocessed)
		parent = node.parent
		parent.remove_child(node)
		newparent = Node(label = "I"+str(index), edge_length = 1)
		index+=1
		newleaf = Node(label = l, edge_length = 1)
		newparent.add_child(node)
		newparent.add_child(newleaf)
		parent.add_child(newparent)

		num_leaves[newleaf] = 1
		num_leaves[newparent] = num_leaves[node] + 1
		num_leaves[parent] += 1

	inferred_tree.write_tree_newick(args.outfile)
	# print(inferred_tree)

	end = time.time()

	print(end - start)



if __name__ == "__main__":
	main()