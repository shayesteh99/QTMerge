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

def vprint(*x, **kwargs):
	if VERBOSE:
		print(*x, **kwargs)

def __label_tree__(tree_obj):
	is_labeled = True
	i = 0
	labels = set()
	for node in tree_obj.traverse_preorder():
		node.edge_length = 1
		if node.is_leaf():
			continue
		if not node.label or node.label in labels or isinstance(node.label, float): 
			is_labeled = False
			node.label = 'I' + str(i)
			i += 1     
		labels.add(node.label)
	return is_labeled

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

def count_all_topos(trees, taxa_list):
	if len(taxa_list) != 4:
		print("not enough taxa!!")
		return
	taxa = [t for i in range(len(taxa_list)) for t in taxa_list[i]]
	num_quartets = [0,0,0]
	for t in trees:
		tree = t.extract_tree_with(taxa)
		num_taxa = {}

		for n in tree.traverse_postorder():
			num_taxa[n] = [0,0,0,0]
			if n.is_leaf():
				for i in range(4):
					if n.label in taxa_list[i]:
						num_taxa[n][i] = 1
				continue
			for c in n.child_nodes():
				for i in range(4):
					num_taxa[n][i] += num_taxa[c][i]

		root = tree.root

		for n in tree.traverse_postorder():
			if n.is_leaf():
				continue
			for c1 in n.child_nodes():
				for c2 in n.child_nodes():
					if c1 != c2:
						#first topology 0,1|2,3
						num_quartets[0] += num_taxa[c1][0] * num_taxa[c2][1] * (num_taxa[root][2] - num_taxa[n][2]) * (num_taxa[root][3] - num_taxa[n][3]) 
						num_quartets[0] += num_taxa[c1][2] * num_taxa[c1][3] * num_taxa[c2][0] * (num_taxa[root][1] - num_taxa[n][1])
						num_quartets[0] += num_taxa[c1][2] * num_taxa[c1][3] * num_taxa[c2][1] * (num_taxa[root][0] - num_taxa[n][0])

						#second topology 0,2|1,3
						num_quartets[1] += num_taxa[c1][0] * num_taxa[c2][2] * (num_taxa[root][1] - num_taxa[n][1]) * (num_taxa[root][3] - num_taxa[n][3]) 
						num_quartets[1] += num_taxa[c1][1] * num_taxa[c1][3] * num_taxa[c2][0] * (num_taxa[root][2] - num_taxa[n][2])
						num_quartets[1] += num_taxa[c1][1] * num_taxa[c1][3] * num_taxa[c2][2] * (num_taxa[root][0] - num_taxa[n][0])

						#third topology 0,3|1,2
						num_quartets[2] += num_taxa[c1][0] * num_taxa[c2][3] * (num_taxa[root][1] - num_taxa[n][1]) * (num_taxa[root][2] - num_taxa[n][2]) 
						num_quartets[2] += num_taxa[c1][1] * num_taxa[c1][2] * num_taxa[c2][0] * (num_taxa[root][3] - num_taxa[n][3])
						num_quartets[2] += num_taxa[c1][1] * num_taxa[c1][2] * num_taxa[c2][3] * (num_taxa[root][0] - num_taxa[n][0])
	return num_quartets

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
	for n in tree.traverse_preorder():
		num_leaves[n] = n.num_nodes(leaves=True, internal=False)
	return num_leaves


def find_middle_branch(tree, num_leaves):
	node = tree.root

	if tree.num_nodes(leaves=True, internal=False) <= 4:
		for n in tree.traverse_preorder():
			if not n.is_root() and len(n.child_nodes()) == 2:
				return n

	while True:
		sizes = {}
		if node.is_root():
			for c in node.child_nodes():
				if c.is_leaf():
					sizes[c] = 0
				else:
					sizes[c] = 0
					for cc in c.child_nodes():
						# print(cc.label)
						sizes[c] = max(sizes[c], num_leaves[cc])
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
				if c !=node:
					sizes[c] = num_leaves[c]
			gparent = parent.parent
			sizes[gparent] = num_leaves[tree.root] - sum([sizes[c] for c in sizes])
		max_size = max([sizes[c] for c in sizes])
		largest_is_child = False
		for c in node.child_nodes():
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
		for c in node.child_nodes():
			c_leaves = [l.label for l in c.traverse_leaves()]
			taxa.append(c_leaves)

		parent = node.parent
		for c in parent.child_nodes():
			if c != node:
				c_leaves = [l.label for l in c.traverse_leaves()]
				taxa.append(c_leaves)

		taxa.append([t])

		# q = extract_quartet(s_tree, taxa+[t])
		# print(q)
		if [taxa[0],t] in q or [t,taxa[0]] in q:
			dir = "down"
			nextnode = node.child_nodes()[0]

		elif [taxa[1],t] in q or [t,taxa[1]] in q:
			dir = "down"
			nextnode = node.child_nodes()[1]

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
				for c in node.parent.child_nodes():
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


def find_taxon_placement(t, tree, num_leaves, genetrees, test = False):
	# print("find taxon placement", t)
	# print(tree)
	node = find_middle_branch(tree, num_leaves)
	visited = set()
	prev = None
	while True:
		visited.add(node)
		# print(node.label)
		if len(node.child_nodes()) == 1 and node.child_nodes()[0] in visited:
			node = node.parent
		elif len(node.child_nodes()) == 1:
			node = node.child_nodes()[0]
		if node.is_leaf() and node.parent in visited:
			return node

		taxa = []
		for c in node.child_nodes():
			c_leaves = [l.label for l in c.traverse_leaves()]
			taxa.append(c_leaves)
			# l = next(c.traverse_leaves())
			# taxa.append(l.label)

		parent = node.parent
		for c in parent.child_nodes():
			if c != node:
				c_leaves = [l.label for l in c.traverse_leaves()]
				taxa.append(c_leaves)

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
			nextnode = node.child_nodes()[0]

		elif q[1] > q[2] and q[1] > q[0]:
		# elif [taxa[1],t] in q or [t,taxa[1]] in q:
			# dir = "down"
			nextnode = node.child_nodes()[1]

		else:
			if node.parent.is_root():
				# dir="down"
				if node.parent in visited:
				# if prev == 'down':
					return node
				visited.add(node.parent)
				for c in node.parent.child_nodes():
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
		for c in tree.root.child_nodes():
			if c.num_nodes(leaves = True, internal = False) > 2:
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
			max_size = max([num_leaves[c] for c in middle_node.child_nodes()])
			middle_node = [c for c in middle_node.child_nodes() if num_leaves[c] == max_size][0]

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
			subtree = tree.extract_subtree(node)
			parent= Node(edge_length = 0, label = node.label)
			leaf = Node(edge_length = 0, label = t)
			parent.add_child(node)
			parent.add_child(leaf)
			subtree.root = parent
			outputs.append(read_tree_newick(subtree.newick()))

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
	tree1_num_leaves = compute_num_leaves(tree1)
	tree2 = reroot_middle(tree2)
	#base cases

	num_leaves = tree1.num_nodes(leaves=True, internal=False)
	if num_leaves < 3:
		for n in tree1.traverse_preorder():
			if n.edge_length > 0:
				place = n.label
				break
		for l in tree2.traverse_leaves():
			placements[l.label] = place
		return

	num_leaves = tree2.num_nodes(leaves=True, internal=False)
	if num_leaves < 4:
		for l in tree2.traverse_leaves():
			# print(l.label)
			place = find_taxon_placement(l.label, tree1, tree1_num_leaves, genetrees, test = True)
			# print(l.label, place.label)
			if place is None:
				ghosts.append(l.label)
				continue
			if place.edge_length == 0:
				place = place.parent
			placements[l.label] = place.label
		return

	tree1_copy = tree1.newick()
	node = find_middle_branch(tree1, tree1_num_leaves)

	#get three taxa for tree 1
	taxa = []
	tree1_subs = []
	for c in node.child_nodes():
		c_leaves = [l.label for l in c.traverse_leaves()]
		taxa.append(c_leaves)
		# l = next(c.traverse_leaves())
		# taxa.append(l.label)
		tree1_subs.append([c, 0])

	tree1_subs.append([node, 1])
	set_leaves = set([l.label for l in node.traverse_leaves()])
	c_leaves = [l.label for l in tree1.traverse_leaves() if l.label not in set_leaves]
	taxa.append(c_leaves)
	# for c in parent.child_nodes():
	# 	if c != node:
	# 		l = next(c.traverse_leaves())
	# 		taxa.append(l.label)

	# print([n[0].label for n in tree1_subs])
	# print(taxa)

	#get 4 taxa from tree 2
	tree2_taxa = []
	tree2_subs = []
	root = tree2.root
	for c in root.child_nodes():
		for cc in c.child_nodes():
			c_leaves = [l.label for l in cc.traverse_leaves()]
			tree2_taxa.append(c_leaves)
			# l = next(cc.traverse_leaves())
			# tree2_taxa.append(l.label)
			tree2_subs.append([cc, 0])

	vprint("rerooted: ", tree2)
	# vprint(tree2_taxa)

	#assign subtrees
	assignments = []
	for t in tree2_taxa:
		# print(taxa + [t])
		q = count_all_topos(genetrees, taxa + [t])
		# print("q: ", q)
		# q = extract_quartet(genetrees, taxa+[t])
		if q[2] >= q[1] and q[2] >= q[0]:
		# if [taxa[0],t] in q or [t,taxa[0]] in q:
			assignments.append(0)
		elif q[1] >= q[2] and q[1] >= q[0]:
		# elif [taxa[1],t] in q or [t,taxa[1]] in q:
			assignments.append(1)
		elif q[0] >= q[1] and q[0] >= q[2]:
		# elif [taxa[2],t] in q or [t,taxa[2]] in q:
			assignments.append(2)
	# print(assignments)

	counts = {}
	for i in range(len(assignments)):
		if assignments[i] not in counts:
			counts[assignments[i]] = [i]
		else:
			counts[assignments[i]].append(i)

	tree1_subtrees = create_subtrees(tree1, tree1_subs, taxa)

	# print(counts)
	if len(counts) == 1:
		# which = set(tree2_taxa)
		which = [1 for i in range(len(tree2_taxa))]
		# print("all in one component")
		# out = taxa[(assignments[0]+1)%3]
		out = [t for i in range(len(taxa)) for t in taxa[i] if i!= assignments[0]]
		# print(out, tree2_taxa)
		for i in range(len(tree2_taxa)):
			rest = [tree2_taxa[j] for j in range(len(tree2_taxa)) if j!=i]
			q = count_all_topos(genetrees, rest+[out])
			if q[0] > q[1] and q[0] > q[2]:
				which[0 + int(i <= 0)] = 0
				which[1 + int(i <= 1)] = 0
			elif q[1] > q[2] and q[1] > q[0]:
				which[0 + int(i <= 0)] = 0
				which[2 + int(i <= 2)] = 0
			elif q[2] > q[1] and q[2] > q[0]:
				which[1 + int(i <= 1)] = 0
				which[2 + int(i <= 2)] = 0
			# bi = set([b for b in q if out not in b][0])
			# which -= bi
		# print(which)
		if sum(which) == 0:
			i = assignments[0]
			merge_trees(genetrees, tree1_subtrees[i], tree2, placements, ghosts)
			return
		g = [i for i in range(len(which)) if which[i]==1][0]
		i = assignments[0]
		counts[i].remove(g)
		counts[(assignments[0]+1)%3] = [g]
		# print(counts)


	groups = [[tree2_subs[i][0] for i in counts[g]] for g in counts]
	tree2_subtrees = divide_tree(tree2, groups)

	max_group = max([len(counts[g]) for g in counts])

	if max_group < 3:
		i = 0
		for g in counts:
			merge_trees(genetrees, tree1_subtrees[g], tree2_subtrees[i], placements, ghosts)
			i+=1
	else:
		i = 0
		for g in counts:
			if len(counts[g]) == 1:	
				new_tree = read_tree_newick(tree1_copy)
				merge_trees(genetrees, new_tree, tree2_subtrees[i], placements, ghosts)
			else:
				merge_trees(genetrees, tree1_subtrees[g], tree2_subtrees[i], placements, ghosts)
			i+=1
	return


def create_full_tree(rev_placements, tree1, tree2, genetrees, index):
	full_tree = read_tree_newick(tree1.newick())
	for p in rev_placements:
		# print(p, rev_placements[p])
		if len(rev_placements[p]) == 1:
			node = full_tree.label_to_node(selection='all')[p]
			parent = node.parent
			parent.remove_child(node)
			newparent = Node(label = 'I'+str(index), edge_length = 1)
			newleaf = Node(label = rev_placements[p][0], edge_length = 1)
			newparent.add_child(node)
			newparent.add_child(newleaf)
			parent.add_child(newparent)
			index+=1
		elif len(rev_placements[p]) == 2:
			node = tree1.label_to_node(selection='all')[p]
			leaf_down = [l.label for l in node.traverse_leaves()]
			set_leaves = set(leaf_down)
			leaf_up = [l.label for l in tree1.traverse_leaves() if l.label not in set_leaves]
			q = count_all_topos(genetrees, [[i] for i in rev_placements[p]]+[leaf_up, leaf_down])

			node = full_tree.label_to_node(selection='all')[p]
			newleaf1 = Node(label = rev_placements[p][0], edge_length = 1)
			newleaf2 = Node(label = rev_placements[p][1], edge_length = 1)

			if q[0] >= q[1] and q[0] >= q[2]:
			# if [leaf_up.label, leaf_down.label] in q or [leaf_down.label, leaf_up.label] in q:
				newparent = Node(label = 'I'+str(index), edge_length = 1)
				newparent.add_child(newleaf1)
				newparent.add_child(newleaf2)
				index+=1

				parent = node.parent
				parent.remove_child(node)
				newgparent = Node(label = 'I'+str(index), edge_length = 1)
				newgparent.add_child(node)
				newgparent.add_child(newparent)
				parent.add_child(newgparent)
				index+=1

			elif q[2] >= q[1] and q[2] >= q[0]:
			# elif [leaf_down.label, rev_placements[p][0]] in q or [rev_placements[p][0], leaf_down.label] in q:
				parent = node.parent
				parent.remove_child(node)

				newparent1 = Node(label = 'I'+str(index), edge_length = 1)
				newparent1.add_child(newleaf1)
				newparent1.add_child(node)
				index+=1

				newparent2 = Node(label = 'I'+str(index), edge_length = 1)
				newparent2.add_child(newleaf2)
				newparent2.add_child(newparent1)
				index+=1

				parent.add_child(newparent2)	
			else:
				parent = node.parent
				parent.remove_child(node)

				newparent1 = Node(label = 'I'+str(index), edge_length = 1)
				newparent1.add_child(newleaf2)
				newparent1.add_child(node)
				index+=1

				newparent2 = Node(label = 'I'+str(index), edge_length = 1)
				newparent2.add_child(newleaf1)
				newparent2.add_child(newparent1)
				index+=1

				parent.add_child(newparent2)			

		else:
			# print(p, rev_placements[p])
			node = tree1.label_to_node(selection='all')[p]
			leaf_down = [l.label for l in node.traverse_leaves()]
			set_leaves = set(leaf_down)
			leaf_up = [l.label for l in tree1.traverse_leaves() if l.label not in set_leaves]

			subtree = tree2.extract_tree_with(rev_placements[p])
			num_leaves = compute_num_leaves(subtree)
			# print(subtree)
			up_node = find_taxon_placement(leaf_up[0], subtree, num_leaves, genetrees)
			label = subtree.root.label
			subtree.root.edge_length = None
			root = subtree.root
			subtree.reroot(up_node, length = up_node.edge_length/2)
			root.contract()
			subtree.root.label = label
			subtree.root.edge_length = 1
			# index += 1
			num_leaves = compute_num_leaves(subtree)
			down_node = find_taxon_placement(leaf_down[0], subtree, num_leaves, genetrees)

			# print(subtree)

			node = full_tree.label_to_node(selection='all')[p]
			if up_node == down_node or (up_node.parent.is_root() and down_node.parent.is_root()):
				taxa1 = [l.label for l in down_node.traverse_leaves()]
				set_leaves = set(taxa1)
				taxa2 = [l.label for l in subtree.traverse_leaves() if l.label not in set_leaves]

				q = count_all_topos(genetrees, [leaf_up, leaf_down, taxa1, taxa2])

				if q[0] >= q[1] and q[0] >= q[2]:
				# if [leaf_up.label, leaf_down.label] in q or [leaf_down.label, leaf_up.label] in q:
					parent = node.parent
					parent.remove_child(node)
					newparent = Node(label = 'I'+str(index), edge_length = 1)
					newparent.add_child(subtree.root)
					newparent.add_child(node)
					parent.add_child(newparent)
					index += 1

				elif q[2] >= q[1] and q[2] >= q[0]:
				# elif [taxa1.label, leaf_down.label] in q or [leaf_down.label, taxa1.label] in q:
					sibling = [c for c in subtree.root.child_nodes() if c != down_node][0]
					parent = node.parent
					parent.remove_child(node)

					newparent1 = Node(label = 'I'+str(index), edge_length = 1)
					newparent1.add_child(down_node)
					newparent1.add_child(node)
					index+=1

					newparent2 = Node(label = 'I'+str(index), edge_length = 1)
					newparent2.add_child(sibling)
					newparent2.add_child(newparent1)
					index+=1

					parent.add_child(newparent2)	
				else:
					sibling = [c for c in subtree.root.child_nodes() if c != down_node][0]

					parent = node.parent
					parent.remove_child(node)

					newparent1 = Node(label = 'I'+str(index), edge_length = 1)
					newparent1.add_child(sibling)
					newparent1.add_child(node)
					index+=1

					newparent2 = Node(label = 'I'+str(index), edge_length = 1)
					newparent2.add_child(down_node)
					newparent2.add_child(newparent1)
					index+=1

					parent.add_child(newparent2)
			else:
				parent = node.parent
				parent.remove_child(node)

				trav=down_node.parent
				trav.remove_child(down_node)
				newparent = Node(label = 'I'+str(index), edge_length = 1)
				newparent.add_child(down_node)
				newparent.add_child(node)
				index+=1

				observed = [down_node]
				while trav != None:
					observed.append(trav)
					child = [c for c in trav.child_nodes() if c not in observed][0]
					trav.remove_child(child)
					newparent2 = Node(label = 'I'+str(index), edge_length = 1)
					newparent2.add_child(child)
					newparent2.add_child(newparent)
					newparent = newparent2
					trav = trav.parent
					index+=1
				parent.add_child(newparent)
	return full_tree, index


def infer_tree(leaves, genetrees, index, ghosts = []):
	n = len(leaves)
	if n < 4:
		tree, index = star_tree(leaves, index)
		return tree, index, ghosts
	if n == 4:
		# q = extract_quartet(genetrees, leaves)
		q = count_all_topos(genetrees, [[l] for l in leaves])
		Nodes = []
		biparts = []
		if q[0] >= q[1] and q[0] >= q[2]:
			biparts = [[leaves[0],leaves[1]],[leaves[2],leaves[3]]]
		elif q[1] >= q[2] and q[1] >= q[0]:
			biparts = [[leaves[0],leaves[2]],[leaves[1],leaves[3]]]
		else:
			biparts = [[leaves[0],leaves[3]],[leaves[1],leaves[2]]]
		for bi in biparts:
			n = Node(label = "I"+str(index), edge_length = 1)
			index += 1
			Nodes.append(n)
			l1 = Node(label = bi[0], edge_length = 1)
			l2 = Node(label = bi[1], edge_length = 1)
			n.add_child(l1)
			n.add_child(l2)
		root = Node(label = "I"+str(index), edge_length = 0)
		index += 1
		tree = Tree()
		tree.root = root

		for n in Nodes:
			root.add_child(n)
		return tree, index, ghosts

	#divide
	set1 = leaves[:n//2]
	set2 = leaves[n//2:]

	tree1, index, ghosts = infer_tree(set1, genetrees, index, ghosts)
	tree2, index, ghosts = infer_tree(set2, genetrees, index, ghosts)
	vprint("="*200)
	vprint(leaves)
	vprint(tree1)
	vprint(tree2)

	#merge

	placements = {}
	merge_trees(genetrees, read_tree_newick(tree1.newick()), read_tree_newick(tree2.newick()), placements, ghosts)
	rev_placements = {}
	for p in placements:
		if placements[p] == tree1.root.child_nodes()[1].label:
			placements[p] = tree1.root.child_nodes()[0].label
		if placements[p] not in rev_placements:
			rev_placements[placements[p]] = []
		rev_placements[placements[p]].append(p)

	vprint(rev_placements)
	full_tree, index = create_full_tree(rev_placements, tree1, tree2, genetrees, index)
	vprint(full_tree)
	return full_tree, index, ghosts


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-t', '--trees', required=True, help="Input Trees")
	parser.add_argument('-s', '--seed', required=False, default=1142, help="Random Seed")
	parser.add_argument('-k', '--num_quartets', required=False, default=5, help="Number of Samples Quartets")
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
	k = int(args.num_quartets)

	# print("+"*300)

	with open(args.trees, "r") as f:
		trees = f.readlines()
		trees = [read_tree_newick(t) for t in trees]


	start = time.time()

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

	random.shuffle(leaves)
	inferred_tree, index, ghosts = infer_tree(leaves, trees, index = 0)
	vprint(ghosts)
	vprint(inferred_tree)

	num_leaves = compute_num_leaves(inferred_tree)
	for l in ghosts:
		node = find_taxon_placement(l, inferred_tree, num_leaves, trees)
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