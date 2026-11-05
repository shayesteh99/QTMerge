import os
os.environ["PYTHONHASHSEED"] = "0"
import argparse
import time
import numpy as np
from treeswift import *
import random
import warnings
warnings.filterwarnings("ignore")
import subprocess
import tempfile
from statsmodels.stats.proportion import proportions_ztest
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from .preprocess import PreprocessedTree

VERBOSE = False
ALPHA = 0.05


def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def _is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


def _label_tree(tree_obj, index=0):
    is_labeled = True
    labels = set()
    for node in tree_obj.traverse_preorder():
        node.edge_length = 1
        if node.is_leaf():
            continue
        if not node.label or node.label in labels or _is_number(node.label):
            is_labeled = False
            node.label = 'I' + str(index)
            index += 1
        labels.add(node.label)
    return is_labeled, index


def _induced_subtree(pt, leaves):
    if not leaves:
        return [], {}, None
    if len(leaves) == 1:
        only = leaves[0]
        return [only], {only: []}, only
    tin, tout = pt.tin, pt.tout
    leaves_sorted = sorted(leaves, key=lambda n: tin[n])
    nodes = set(leaves_sorted)
    for i in range(len(leaves_sorted) - 1):
        nodes.add(pt.lca(leaves_sorted[i], leaves_sorted[i + 1]))
    preorder = sorted(nodes, key=lambda n: tin[n])
    children = {n: [] for n in preorder}
    stack = []
    for n in preorder:
        while stack:
            top = stack[-1]
            if tin[top] <= tin[n] and tout[n] <= tout[top]:
                break
            stack.pop()
        if stack:
            children[stack[-1]].append(n)
        stack.append(n)
    return preorder, children, preorder[0]


def preprocess_trees(trees):
    return [PreprocessedTree(t) for t in trees]


def count_all_topos(preprocessed_trees, taxa_list):
    if len(taxa_list) != 4:
        return [0, 0, 0]
    label_to_sets = {}
    for i, group in enumerate(taxa_list):
        for label in group:
            label_to_sets.setdefault(label, []).append(i)
    num_quartets = [0, 0, 0]
    for pt in preprocessed_trees:
        target_leaves = [pt.label_to_node[lbl]
                         for lbl in label_to_sets if lbl in pt.label_to_node]
        if len(target_leaves) < 4:
            continue
        preorder, children, root = _induced_subtree(pt, target_leaves)
        num_taxa = {n: [0, 0, 0, 0] for n in preorder}
        for n in reversed(preorder):
            kids = children[n]
            if not kids:
                sets = label_to_sets.get(n.label)
                if sets is not None:
                    row = num_taxa[n]
                    for idx in sets:
                        row[idx] = 1
                continue
            counts = num_taxa[n]
            for c in kids:
                cc = num_taxa[c]
                counts[0] += cc[0]; counts[1] += cc[1]
                counts[2] += cc[2]; counts[3] += cc[3]
        root_counts = num_taxa[root]
        for n in preorder:
            kids = children[n]
            if not kids:
                continue
            nc = num_taxa[n]
            out0 = root_counts[0]-nc[0]; out1 = root_counts[1]-nc[1]
            out2 = root_counts[2]-nc[2]; out3 = root_counts[3]-nc[3]
            for c1 in kids:
                a = num_taxa[c1]
                for c2 in kids:
                    if c1 is c2:
                        continue
                    b = num_taxa[c2]
                    num_quartets[0] += a[0]*b[1]*out2*out3 + a[2]*a[3]*b[0]*out1 + a[2]*a[3]*b[1]*out0
                    num_quartets[1] += a[0]*b[2]*out1*out3 + a[1]*a[3]*b[0]*out2 + a[1]*a[3]*b[2]*out0
                    num_quartets[2] += a[0]*b[3]*out1*out2 + a[1]*a[2]*b[0]*out3 + a[1]*a[2]*b[3]*out0
    return num_quartets


def compute_num_leaves(tree):
    num_leaves = {}
    for n in tree.traverse_postorder():
        if n.is_leaf():
            num_leaves[n] = 1
        else:
            num_leaves[n] = sum(num_leaves[c] for c in n.children)
    return num_leaves


def compute_leaf_labels(tree):
    leaf_labels = {}
    for n in tree.traverse_postorder():
        if n.is_leaf():
            leaf_labels[n] = [n.label]
        else:
            leaf_labels[n] = [l for c in n.children for l in leaf_labels[c]]
    return leaf_labels


def find_middle_branch(tree, num_leaves):
    node = tree.root
    if num_leaves[tree.root] <= 4:
        for n in tree.traverse_preorder():
            if not n.is_root() and len(n.child_nodes()) == 2:
                return n
    while True:
        sizes = {}
        if node.is_root():
            for c in node.child_nodes():
                sizes[c] = 0 if c.is_leaf() else max(num_leaves[cc] for cc in c.child_nodes())
        elif node.parent.is_root():
            for c in node.child_nodes():
                sizes[c] = num_leaves[c]
            for c in tree.root.child_nodes():
                if c != node:
                    if c.is_leaf():
                        sizes[c] = 0
                    else:
                        for cc in c.child_nodes():
                            sizes[cc] = num_leaves[cc]
        else:
            for c in node.child_nodes():
                sizes[c] = num_leaves[c]
            parent = node.parent
            for c in parent.child_nodes():
                if c != node:
                    sizes[c] = num_leaves[c]
            sizes[parent.parent] = num_leaves[tree.root] - sum(sizes.values())
        if not sizes:
            return tree.root
        max_size = max(sizes.values())
        largest_is_child = False
        for c in node.child_nodes():
            if sizes[c] == max_size:
                largest_is_child = True
                node = c
                break
        if not largest_is_child:
            if node.parent.is_root():
                return node
            return node.parent


def reroot_middle(tree):
    num_leaves = compute_num_leaves(tree)
    if len(num_leaves) < 4:
        return tree
    middle_node = None
    if len(num_leaves) == 7:
        for c in tree.root.child_nodes():
            if c.num_nodes(leaves=True, internal=False) > 2:
                for cc in c.child_nodes():
                    if len(cc.child_nodes()) == 2:
                        middle_node = cc
                        break
        if middle_node is None:
            return tree
    else:
        middle_node = find_middle_branch(tree, num_leaves)
        if num_leaves[middle_node] < 2:
            middle_node = middle_node.parent
        elif num_leaves[middle_node.parent] - num_leaves[middle_node] < 2:
            max_size = max(num_leaves[c] for c in middle_node.child_nodes())
            middle_node = [c for c in middle_node.child_nodes()
                           if num_leaves[c] == max_size][0]

    M_leaf_set = set(l.label for l in middle_node.traverse_leaves())
    all_leaves = [l.label for l in tree.traverse_leaves()]
    comp_leaves = [lbl for lbl in all_leaves if lbl not in M_leaf_set]

    if len(comp_leaves) < 2 or len(M_leaf_set) < 2:
        return tree

    tree.root.edge_length = None
    old_root = tree.root
    tree.reroot(middle_node, length=middle_node.edge_length / 2)
    old_root.contract()
    return tree


def test_p1_equivalence(counts, epsilon=0.05):
    k = sum(counts)
    x1 = max(counts)
    _, p_val = proportions_ztest(count=x1, nobs=k, value=1/3+epsilon, alternative='smaller')
    return p_val


def find_taxon_placement_new(t, tree, num_leaves, genetrees, test=False, leaf_labels=None):
    if leaf_labels is None:
        leaf_labels = compute_leaf_labels(tree)
    node = find_middle_branch(tree, num_leaves)
    visited = set()
    while True:
        visited.add(node)
        nc = node.child_nodes()
        if len(nc) < 2:
            if nc and nc[0] in visited:
                node = node.parent
            elif nc:
                node = nc[0]
            else:
                return node, 1.0
            continue
        if node.is_leaf() and node.parent in visited:
            return node, 1.0

        taxa = [leaf_labels[c] for c in nc]
        taxa.append([l for l in leaf_labels[tree.root] if l not in leaf_labels[node]])
        taxa.append([t])

        q = count_all_topos(genetrees, taxa)
        pval = test_p1_equivalence(q)
        if test and pval < ALPHA:
            return None, pval

        if q[2] > q[1] and q[2] > q[0]:
            nextnode = nc[0]
        elif q[1] > q[2] and q[1] > q[0]:
            nextnode = nc[1]
        else:
            if node.parent.is_root():
                if node.parent in visited:
                    return node, pval
                visited.add(node.parent)
                for c in node.parent.child_nodes():
                    if c != node:
                        nextnode = c
                        break
            else:
                nextnode = node.parent
        if nextnode in visited:
            if nextnode == node.parent:
                return node, pval
            return nextnode, pval
        node = nextnode


def create_subtrees(tree, subs, taxa):
    outputs = []
    for i in range(len(subs)):
        t = taxa[(i + 1) % len(subs)][0]
        n = subs[i]
        if n[1] == 0:
            node = n[0]
            subtree = tree.extract_subtree(node)
            parent = Node(edge_length=0, label=node.label)
            leaf = Node(edge_length=0, label=t)
            parent.add_child(node); parent.add_child(leaf)
            subtree.root = parent
            outputs.append(read_tree_newick(subtree.newick()))
        if n[1] == 1:
            node = n[0]
            for c in node.child_nodes():
                node.remove_child(c)
            leaf = Node(edge_length=0, label=t)
            node.add_child(leaf)
            outputs.append(tree)
    return outputs


def divide_tree(tree, groups):
    outputs = []
    taken = []
    for i in range(len(groups)):
        g = groups[i]
        if len(g) == 1:
            node = g[0]; node.parent = None
            subtree = Tree(); subtree.root = node
            outputs.append(subtree)
        elif len(g) == 2:
            label = g[0].parent.label
            if label in taken:
                label = g[1].parent.label
            node = Node(edge_length=1, label=label)
            taken.append(label)
            subtree = Tree(); subtree.root = node
            g[0].parent = None; g[1].parent = None
            node.add_child(g[0]); node.add_child(g[1])
            outputs.append(subtree)
        elif len(g) == 3:
            node = g[0].parent.parent
            for c in node.child_nodes():
                for cc in c.child_nodes():
                    if cc not in g:
                        if [cc] not in groups[:i]:
                            node = cc; node.parent = None
                            subtree = Tree(); subtree.root = node
                            outputs.append(subtree)
                            c.remove_child(cc); c.contract()
                            outputs.append(tree)
                            return outputs[::-1]
                        else:
                            c.remove_child(cc); c.contract()
                            outputs.append(tree)
                            break
        elif len(g) == 4:
            node = g[0].parent.parent
            node.parent = None
            subtree = Tree(); subtree.root = node
            outputs.append(subtree)
    return outputs


def create_full_tree(rev_placements, tree1, tree2, genetrees, index):
    full_tree = read_tree_newick(tree1.newick())

    for p in rev_placements:
        if len(rev_placements[p]) == 1:
            node = full_tree.label_to_node(selection='all')[p]
            parent = node.parent; parent.remove_child(node)
            newparent = Node(label='I' + str(index), edge_length=1); index += 1
            newleaf = Node(label=rev_placements[p][0], edge_length=1)
            newparent.add_child(node); newparent.add_child(newleaf)
            parent.add_child(newparent)

        elif len(rev_placements[p]) == 2:
            node = tree1.label_to_node(selection='all')[p]
            leaf_down = [l.label for l in node.traverse_leaves()]
            leaf_up = [l.label for l in tree1.traverse_leaves()
                       if l.label not in set(leaf_down)]
            q = count_all_topos(genetrees,
                                [[i] for i in rev_placements[p]] + [leaf_up, leaf_down])
            node = full_tree.label_to_node(selection='all')[p]
            newleaf1 = Node(label=rev_placements[p][0], edge_length=1)
            newleaf2 = Node(label=rev_placements[p][1], edge_length=1)
            if q[0] >= q[1] and q[0] >= q[2]:
                newparent = Node(label='I' + str(index), edge_length=1); index += 1
                newparent.add_child(newleaf1); newparent.add_child(newleaf2)
                parent = node.parent; parent.remove_child(node)
                newgparent = Node(label='I' + str(index), edge_length=1); index += 1
                newgparent.add_child(node); newgparent.add_child(newparent)
                parent.add_child(newgparent)
            elif q[2] >= q[1] and q[2] >= q[0]:
                parent = node.parent; parent.remove_child(node)
                np1 = Node(label='I' + str(index), edge_length=1); index += 1
                np1.add_child(newleaf1); np1.add_child(node)
                np2 = Node(label='I' + str(index), edge_length=1); index += 1
                np2.add_child(newleaf2); np2.add_child(np1)
                parent.add_child(np2)
            else:
                parent = node.parent; parent.remove_child(node)
                np1 = Node(label='I' + str(index), edge_length=1); index += 1
                np1.add_child(newleaf2); np1.add_child(node)
                np2 = Node(label='I' + str(index), edge_length=1); index += 1
                np2.add_child(newleaf1); np2.add_child(np1)
                parent.add_child(np2)

        else:
            node = tree1.label_to_node(selection='all')[p]
            leaf_down = [l.label for l in node.traverse_leaves()]
            leaf_up = [l.label for l in tree1.traverse_leaves()
                       if l.label not in set(leaf_down)]
            subtree = tree2.extract_tree_with(rev_placements[p])
            nl = compute_num_leaves(subtree)
            up_node, _ = find_taxon_placement_new(leaf_up[0], subtree, nl, genetrees)
            label = subtree.root.label
            subtree.root.edge_length = None
            old_root = subtree.root
            subtree.reroot(up_node, length=up_node.edge_length / 2)
            old_root.contract()
            subtree.root.label = label
            subtree.root.edge_length = 1
            nl = compute_num_leaves(subtree)
            down_node, _ = find_taxon_placement_new(leaf_down[0], subtree, nl, genetrees)

            node = full_tree.label_to_node(selection='all')[p]
            if up_node == down_node or (up_node.parent.is_root() and down_node.parent.is_root()):
                taxa1 = [l.label for l in down_node.traverse_leaves()]
                taxa2 = [l.label for l in subtree.traverse_leaves()
                         if l.label not in set(taxa1)]
                q = count_all_topos(genetrees, [leaf_up, leaf_down, taxa1, taxa2])
                if q[0] >= q[1] and q[0] >= q[2]:
                    parent = node.parent; parent.remove_child(node)
                    newparent = Node(label='I' + str(index), edge_length=1); index += 1
                    newparent.add_child(subtree.root); newparent.add_child(node)
                    parent.add_child(newparent)
                elif q[2] >= q[1] and q[2] >= q[0]:
                    sibling = [c for c in subtree.root.child_nodes() if c != down_node][0]
                    parent = node.parent; parent.remove_child(node)
                    np1 = Node(label='I' + str(index), edge_length=1); index += 1
                    np1.add_child(down_node); np1.add_child(node)
                    np2 = Node(label='I' + str(index), edge_length=1); index += 1
                    np2.add_child(sibling); np2.add_child(np1)
                    parent.add_child(np2)
                else:
                    sibling = [c for c in subtree.root.child_nodes() if c != down_node][0]
                    parent = node.parent; parent.remove_child(node)
                    np1 = Node(label='I' + str(index), edge_length=1); index += 1
                    np1.add_child(sibling); np1.add_child(node)
                    np2 = Node(label='I' + str(index), edge_length=1); index += 1
                    np2.add_child(down_node); np2.add_child(np1)
                    parent.add_child(np2)
            else:
                parent = node.parent; parent.remove_child(node)
                trav = down_node.parent
                trav.remove_child(down_node)
                newparent = Node(label='I' + str(index), edge_length=1); index += 1
                newparent.add_child(down_node); newparent.add_child(node)
                observed = [down_node]
                while trav is not None:
                    observed.append(trav)
                    child = [c for c in trav.child_nodes() if c not in observed][0]
                    trav.remove_child(child)
                    np2 = Node(label='I' + str(index), edge_length=1); index += 1
                    np2.add_child(child); np2.add_child(newparent)
                    newparent = np2
                    trav = trav.parent
                    index += 1
                parent.add_child(newparent)

    return full_tree, index


def _make_star_tree(taxa, index):
    root = Node()
    root.label = 'I' + str(index); index += 1
    root.edge_length = 0
    for lbl in taxa:
        leaf = Node()
        leaf.label = lbl
        leaf.edge_length = 1
        root.add_child(leaf)
    tree_obj = Tree()
    tree_obj.root = root
    return tree_obj, index


def get_astral_tree(trees, taxa, index=0):
    if len(taxa) < 4:
        return _make_star_tree(taxa, index)
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    for t in trees:
        f.write(t.extract_tree_with(taxa).newick() + '\n')
    f.flush(); f.close()
    try:
        proc = subprocess.run(['astral4', '-i', f.name, '--length', 'CULength'],
                              capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        return _make_star_tree(taxa, index)
    tree_obj = read_tree_newick(proc.stdout.split('\n')[0])
    _, index = _label_tree(tree_obj, index)
    return tree_obj, index


def compute_pairwise_distances(preprocessed_trees, leaves):
    n = len(leaves)
    leaf_to_idx = {l: i for i, l in enumerate(leaves)}
    dist_sum = np.zeros((n, n))
    n_counted = 0

    for pt in preprocessed_trees:
        target = [pt.label_to_node[lbl] for lbl in leaves if lbl in pt.label_to_node]
        if len(target) < 2:
            continue
        n_counted += 1
        preorder, children, root = _induced_subtree(pt, target)

        nl = {}
        idx_sets = {}
        for node in reversed(preorder):
            kids = children[node]
            if not kids:
                nl[node] = 1
                idx_sets[node] = [leaf_to_idx[node.label]]
            else:
                nl[node] = sum(nl[c] for c in kids)
                idx_sets[node] = [i for c in kids for i in idx_sets[c]]

        total = nl[root]

        for node in preorder:
            kids = children[node]
            if len(kids) < 2:
                continue
            frac = nl[node] / total
            for k1 in range(len(kids)):
                s1 = idx_sets[kids[k1]]
                for k2 in range(k1 + 1, len(kids)):
                    s2 = idx_sets[kids[k2]]
                    if len(s1) == 1 and len(s2) == 1:
                        dist_sum[s1[0], s2[0]] += frac
                        dist_sum[s2[0], s1[0]] += frac
                    else:
                        a1 = np.array(s1, dtype=np.intp)
                        a2 = np.array(s2, dtype=np.intp)
                        dist_sum[np.ix_(a1, a2)] += frac
                        dist_sum[np.ix_(a2, a1)] += frac

    if n_counted == 0:
        return np.ones((n, n)) - np.eye(n)
    dist_sum /= n_counted
    np.fill_diagonal(dist_sum, 0.0)
    return dist_sum


def merge_trees(genetrees, tree1, tree2, placements):
    tree1_num_leaves = compute_num_leaves(tree1)

    if tree1.num_nodes(leaves=True, internal=False) < 3:
        for n in tree1.traverse_preorder():
            if not n.is_root() and n.edge_length > 0:
                place = n.label
                break
        for l in tree2.traverse_leaves():
            placements[l.label] = place
        return

    if tree2.num_nodes(leaves=True, internal=False) < 4:
        for l in tree2.traverse_leaves():
            place, _ = find_taxon_placement_new(l.label, tree1, tree1_num_leaves, genetrees, test=False)
            if place.edge_length == 0:
                place = place.parent
            placements[l.label] = place.label
        return

    tree1_copy = tree1.newick()
    node = find_middle_branch(tree1, tree1_num_leaves)

    taxa = []
    tree1_subs = []
    for c in node.child_nodes():
        taxa.append([l.label for l in c.traverse_leaves()])
        tree1_subs.append([c, 0])
    tree1_subs.append([node, 1])
    set_leaves = set(l.label for l in node.traverse_leaves())
    taxa.append([l.label for l in tree1.traverse_leaves() if l.label not in set_leaves])
    L1, L2, L3 = taxa[0], taxa[1], taxa[2]

    tree2_mid = reroot_middle(tree2)
    tree2_taxa = []
    tree2_subs = []
    for c in tree2_mid.root.child_nodes():
        c_children = list(c.child_nodes())
        if not c_children:
            tree2_taxa.append([l.label for l in c.traverse_leaves()])
            tree2_subs.append([c, 0])
        else:
            for cc in c_children:
                tree2_taxa.append([l.label for l in cc.traverse_leaves()])
                tree2_subs.append([cc, 0])

    assignments = []
    for t in tree2_taxa:
        q = count_all_topos(genetrees, taxa + [t])
        if q[2] >= q[1] and q[2] >= q[0]:
            assignments.append(0)
        elif q[1] >= q[2] and q[1] >= q[0]:
            assignments.append(1)
        else:
            assignments.append(2)

    vprint(f"    assignments: {assignments}")

    counts = {}
    for i, a in enumerate(assignments):
        counts.setdefault(a, []).append(i)

    tree1_subtrees = create_subtrees(tree1, tree1_subs, taxa)

    if len(counts) == 1:
        which = [1] * len(tree2_taxa)
        out = [t for i in range(len(taxa)) for t in taxa[i] if i != assignments[0]]
        for i in range(len(tree2_taxa)):
            rest = [tree2_taxa[j] for j in range(len(tree2_taxa)) if j != i]
            q = count_all_topos(genetrees, rest + [out])
            if q[0] > q[1] and q[0] > q[2]:
                which[0 + int(i <= 0)] = 0; which[1 + int(i <= 1)] = 0
            elif q[1] > q[2] and q[1] > q[0]:
                which[0 + int(i <= 0)] = 0; which[2 + int(i <= 2)] = 0
            elif q[2] > q[1] and q[2] > q[0]:
                which[1 + int(i <= 1)] = 0; which[2 + int(i <= 2)] = 0
        if sum(which) == 0:
            i = assignments[0]
            merge_trees(genetrees, tree1_subtrees[i], tree2_mid, placements)
            return
        g = [i for i in range(len(which)) if which[i] == 1][0]
        i = assignments[0]
        counts[i].remove(g)
        counts[(assignments[0] + 1) % 3] = [g]

    groups = [[tree2_subs[i][0] for i in counts[g]] for g in counts]
    tree2_subtrees = divide_tree(tree2_mid, groups)
    max_group = max(len(counts[g]) for g in counts)

    if max_group < 3:
        for idx, g in enumerate(counts):
            merge_trees(genetrees, tree1_subtrees[g], tree2_subtrees[idx], placements)
    else:
        for idx, g in enumerate(counts):
            if len(counts[g]) == 1:
                new_tree = read_tree_newick(tree1_copy)
                merge_trees(genetrees, new_tree, tree2_subtrees[idx], placements)
            else:
                merge_trees(genetrees, tree1_subtrees[g], tree2_subtrees[idx], placements)


def merge_all_subtrees(input_trees, genetrees, index):
    if len(input_trees) == 1:
        return input_trees[0], index

    new_trees = []
    mid = len(input_trees) // 2
    if len(input_trees) % 2 == 1:
        new_trees.append(input_trees[-1])

    for i in range(mid):
        tree1 = input_trees[i]
        tree2 = input_trees[i + mid]
        vprint(f"\n  Merging pair {i}: {tree1.num_nodes(leaves=True)} + "
               f"{tree2.num_nodes(leaves=True)} leaves")

        placements = {}
        merge_trees(genetrees,
                    read_tree_newick(tree1.newick()),
                    read_tree_newick(tree2.newick()),
                    placements)

        rev_placements = {}
        for p, anchor in placements.items():
            if anchor == tree1.root.child_nodes()[1].label:
                anchor = tree1.root.child_nodes()[0].label
            rev_placements.setdefault(anchor, []).append(p)

        full_tree, index = create_full_tree(rev_placements, tree1, tree2, genetrees, index)
        new_trees.append(full_tree)

    return merge_all_subtrees(new_trees, genetrees, index)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--trees',    required=True,
                        help='Input gene tree file (one Newick tree per line)')
    parser.add_argument('-s', '--seed',     default=1142, type=int,
                        help='Random seed')
    parser.add_argument('-m', '--min_size', default='sqrt',
                        help='Minimum subset size; use "sqrt" for sqrt(n) or an integer')
    parser.add_argument('-o', '--outfile',  default='./temp',
                        help='Output file for the inferred species tree')
    parser.add_argument('-v', '--verbose',  action='store_true',
                        help='Print progress information')
    parser.add_argument('--no-smart-split', dest='smart_split', action='store_false',
                        help='Disable clade-aware partitioning (use random partitioning instead)')
    parser.add_argument('--merge', nargs=2, metavar=('TREE1', 'TREE2'),
                        help='Merge two disjoint partial trees using quartet signal from the gene trees')
    parser.add_argument('--place', nargs=2, metavar=('BACKBONE', 'TAXA'),
                        help='Infer an ASTRAL subtree for the taxa in TAXA (one label per line) '
                             'and merge it into the BACKBONE tree')
    parser.set_defaults(smart_split=True)
    args = parser.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    start = time.time()

    with open(args.trees) as f:
        trees = [read_tree_newick(t) for t in f.readlines()]
    preprocessed = preprocess_trees(trees)

    if args.merge:
        with open(args.merge[0]) as f:
            tree1 = read_tree_newick(f.readline().strip())
        with open(args.merge[1]) as f:
            tree2 = read_tree_newick(f.readline().strip())
        index = 0
        _, index = _label_tree(tree1, index)
        _, index = _label_tree(tree2, index)
        vprint(f"Merging {tree1.num_nodes(leaves=True, internal=False)} + "
               f"{tree2.num_nodes(leaves=True, internal=False)} leaves")
        inferred_tree, _ = merge_all_subtrees([tree1, tree2], preprocessed, index)
        with open(args.outfile, 'w') as f:
            f.write(inferred_tree.newick() + '\n')
        print(time.time() - start)
        return

    if args.place:
        with open(args.place[0]) as f:
            backbone = read_tree_newick(f.readline().strip())
        with open(args.place[1]) as f:
            taxa = [l.strip() for l in f if l.strip()]
        index = 0
        _, index = _label_tree(backbone, index)
        vprint(f"Inferring ASTRAL subtree for {len(taxa)} taxa")
        subtree, index = get_astral_tree(trees, taxa, index)
        vprint(f"Merging subtree ({len(taxa)} taxa) into backbone "
               f"({backbone.num_nodes(leaves=True, internal=False)} leaves)")
        inferred_tree, _ = merge_all_subtrees([backbone, subtree], preprocessed, index)
        with open(args.outfile, 'w') as f:
            f.write(inferred_tree.newick() + '\n')
        print(time.time() - start)
        return

    leaves = set()
    for t in trees:
        for l in t.traverse_leaves():
            leaves.add(l.label)
    leaves = list(leaves)

    n = len(leaves)
    if args.min_size == 'sqrt':
        m = max(10, int(np.sqrt(n)))
    else:
        m = int(args.min_size)

    num_trees = max(1, n // m)
    vprint(f"n={n}, m={m}, num_trees={num_trees}")

    random.shuffle(leaves)

    if num_trees <= 1:
        inferred_tree, _ = get_astral_tree(trees, leaves)
    else:
        index = 0
        if args.smart_split:
            vprint("  Computing pairwise gene tree distances...")
            t0 = time.time()
            dist_matrix = compute_pairwise_distances(preprocessed, leaves)
            vprint(f"  Distances computed in {time.time()-t0:.1f}s")
            condensed = squareform(dist_matrix, checks=False)
            Z = linkage(condensed, method='average')
            cluster_labels = fcluster(Z, num_trees, criterion='maxclust')
            subset_dict = {}
            for i, g in enumerate(cluster_labels):
                subset_dict.setdefault(g, []).append(leaves[i])
            subsets = list(subset_dict.values())
            overflow = [t for s in subsets if len(s) < 4 for t in s]
            subsets = [s for s in subsets if len(s) >= 4]
            if overflow:
                if subsets:
                    subsets.sort(key=len, reverse=True)
                    subsets[0].extend(overflow)
                else:
                    subsets = [overflow]
        else:
            subsets = [[] for _ in range(num_trees)]
            for i, leaf in enumerate(leaves):
                subsets[i % num_trees].append(leaf)
        input_trees = []
        for subset in subsets:
            tree_obj, index = get_astral_tree(trees, subset, index)
            input_trees.append(tree_obj)

        vprint(f"\nMerging {len(input_trees)} subtrees")
        inferred_tree, index = merge_all_subtrees(input_trees, preprocessed, index)

    with open(args.outfile, 'w') as f:
        f.write(inferred_tree.newick() + '\n')
    print(time.time() - start)


if __name__ == '__main__':
    main()
