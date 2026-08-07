# cython: boundscheck=False, wraparound=False, initializedcheck=False
from libc.math cimport erf, sqrt


cdef double _normal_cdf(double x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


cdef double _p1_equivalence(list counts, double epsilon):
    cdef long k = counts[0] + counts[1] + counts[2]
    cdef long x1
    cdef double proportion, variance, z_stat
    if k <= 0:
        return 0.0
    if counts[0] >= counts[1] and counts[0] >= counts[2]:
        x1 = counts[0]
    elif counts[1] >= counts[0] and counts[1] >= counts[2]:
        x1 = counts[1]
    else:
        x1 = counts[2]
    proportion = (<double>x1) / k
    variance = proportion * (1.0 - proportion) / k
    if variance <= 0.0:
        return 1.0 if proportion >= 1.0 / 3.0 + epsilon else 0.0
    z_stat = (proportion - (1.0 / 3.0 + epsilon)) / sqrt(variance)
    return _normal_cdf(z_stat)


cdef bint _adaptive_decisive(list counts, double margin_threshold,
                             double alpha, double epsilon):
    cdef long total = counts[0] + counts[1] + counts[2]
    cdef long first, second, value
    cdef double margin
    if total <= 0:
        return False
    first = counts[0]
    second = counts[1]
    if second > first:
        value = first
        first = second
        second = value
    if counts[2] > first:
        second = first
        first = counts[2]
    elif counts[2] > second:
        second = counts[2]
    margin = (<double>(first - second)) / total
    return margin >= margin_threshold and _p1_equivalence(counts, epsilon) >= alpha


cdef int _lca(int u, int v, list first, list flat_log, list flat_st,
              list flat_euler, list flat_depth):
    cdef int l = <int>first[u]
    cdef int r = <int>first[v]
    cdef int tmp, k, a_idx, b_idx, a, b
    if l > r:
        tmp = l
        l = r
        r = tmp
    k = <int>flat_log[r - l + 1]
    a_idx = <int>flat_st[k][l]
    b_idx = <int>flat_st[k][r - (1 << k) + 1]
    a = <int>flat_euler[a_idx]
    b = <int>flat_euler[b_idx]
    if <int>flat_depth[a] < <int>flat_depth[b]:
        return a
    return b


cdef void _add_topos_for_tree(object pt, list label_assignments, list nq):
    cdef dict label_to_id = pt.label_to_id
    cdef list tin = pt._flat_tin
    cdef list tout = pt._flat_tout
    cdef list depth = pt._flat_depth
    cdef list first = pt._flat_first
    cdef list flat_euler = pt._flat_euler
    cdef list flat_st = pt._flat_st
    cdef list flat_log = pt._flat_log
    cdef list target_leaves = []
    cdef list node_to_set = [-1] * len(tin)
    cdef object pair, node_obj, existing
    cdef int label_set, node, i, u, v, lca_node

    for pair in label_assignments:
        node_obj = label_to_id.get(pair[0])
        if node_obj is not None:
            node = <int>node_obj
            if <int>node_to_set[node] == -1:
                node_to_set[node] = <int>pair[1]
                target_leaves.append(node)

    if len(target_leaves) < 4:
        return

    target_leaves.sort(key=tin.__getitem__)
    cdef set node_set = set(target_leaves)
    for i in range(len(target_leaves) - 1):
        u = <int>target_leaves[i]
        v = <int>target_leaves[i + 1]
        if u != v:
            node_set.add(_lca(u, v, first, flat_log, flat_st, flat_euler, depth))

    cdef list preorder = list(node_set)
    preorder.sort(key=tin.__getitem__)
    cdef list idx_by_node = [-1] * len(tin)
    cdef list children = []
    for i in range(len(preorder)):
        idx_by_node[preorder[i]] = i
        children.append([])

    cdef list stack = []
    cdef int n, top, tn, tn_out
    for i in range(len(preorder)):
        n = <int>preorder[i]
        tn = <int>tin[n]
        tn_out = <int>tout[n]
        while stack:
            top = <int>stack[-1]
            if <int>tin[top] <= tn and tn_out <= <int>tout[top]:
                break
            stack.pop()
        if stack:
            children[<int>idx_by_node[<int>stack[-1]]].append(n)
        stack.append(n)

    cdef list num_taxa = []
    cdef list counts, kids, cc, a, b
    cdef int si, c, j, root, out0, out1, out2, out3
    cdef long a0, a1, a2, a3, b0, b1, b2, b3
    for i in range(len(preorder)):
        num_taxa.append([0, 0, 0, 0])
    for i in range(len(preorder) - 1, -1, -1):
        n = <int>preorder[i]
        kids = children[i]
        if not kids:
            si = <int>node_to_set[n]
            if si >= 0:
                num_taxa[i][si] = 1
            continue
        counts = num_taxa[i]
        for j in range(len(kids)):
            c = <int>kids[j]
            cc = num_taxa[<int>idx_by_node[c]]
            counts[0] += cc[0]
            counts[1] += cc[1]
            counts[2] += cc[2]
            counts[3] += cc[3]

    root = <int>preorder[0]
    cdef list root_counts = num_taxa[<int>idx_by_node[root]]
    cdef list n_counts
    for i in range(len(preorder)):
        kids = children[i]
        if not kids:
            continue
        n_counts = num_taxa[i]
        out0 = <int>root_counts[0] - <int>n_counts[0]
        out1 = <int>root_counts[1] - <int>n_counts[1]
        out2 = <int>root_counts[2] - <int>n_counts[2]
        out3 = <int>root_counts[3] - <int>n_counts[3]
        if len(kids) == 2:
            a = num_taxa[<int>idx_by_node[kids[0]]]
            b = num_taxa[<int>idx_by_node[kids[1]]]
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
                a = num_taxa[<int>idx_by_node[c1]]
                for c2 in kids:
                    if c1 == c2:
                        continue
                    b = num_taxa[<int>idx_by_node[c2]]
                    nq[0] += a[0]*b[1]*out2*out3 + a[2]*a[3]*b[0]*out1 + a[2]*a[3]*b[1]*out0
                    nq[1] += a[0]*b[2]*out1*out3 + a[1]*a[3]*b[0]*out2 + a[1]*a[3]*b[2]*out0
                    nq[2] += a[0]*b[3]*out1*out2 + a[1]*a[2]*b[0]*out3 + a[1]*a[2]*b[3]*out0


def count_all_topos_fast(list preprocessed_trees, list taxa_list,
                         bint adaptive, int min_trees, int step,
                         double margin, double alpha, double epsilon):
    cdef list label_assignments = []
    cdef int i, tree_index
    cdef object group, label
    for i, group in enumerate(taxa_list):
        for label in group:
            label_assignments.append((label, i))

    cdef list num_quartets = [0, 0, 0]
    cdef int total_trees = len(preprocessed_trees)
    if min_trees < 1:
        min_trees = 1
    if min_trees > total_trees:
        min_trees = total_trees
    if step < 1:
        step = 1

    for tree_index in range(total_trees):
        _add_topos_for_tree(preprocessed_trees[tree_index], label_assignments, num_quartets)
        if (adaptive and tree_index + 1 >= min_trees and
            (tree_index + 1 == total_trees or (tree_index + 1) % step == 0) and
            _adaptive_decisive(num_quartets, margin, alpha, epsilon)):
            return num_quartets, tree_index + 1
    return num_quartets, total_trees
