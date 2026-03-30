import sys
import os
import argparse
import time
import numpy as np
from treeswift import *
import treeswift
import random
import json

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

def extract_quartet(tree, taxa):
	pruned = tree.extract_tree_with(taxa)
	parents = {}
	for l in pruned.traverse_leaves():
		if l.parent not in parents:
			parents[l.parent] = [l.label]
		else:
			parents[l.parent] += [l.label]

	for p in parents:
		if len(parents[p]) == 2:
			return parents[p], [l for l in taxa if l not in parents[p]]

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

def reroot_middle(tree):
	# rtree = read_tree_newick(tree.newick())
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
	tree.root.edge_length = None
	root = tree.root
	tree.reroot(middle_node, length = middle_node.edge_length/2)
	root.contract()
	# for l in tree.traverse_leaves():
	# 	if l.label == "ROOT":
	# 		parent = l.parent
	# 		parent.remove_child(l)
	# 		parent.contract()
	# 		break

	return tree

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
			l = next(c.traverse_leaves())
			taxa.append(l.label)

		parent = node.parent
		for c in parent.child_nodes():
			if c != node:
				l = next(c.traverse_leaves())
				taxa.append(l.label)

		q = extract_quartet(s_tree, taxa+[t])
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


def find_taxon_placement(t, tree, num_leaves, s_tree):
	node = find_middle_branch(tree, num_leaves)
	prev = None
	while True:
		if node.is_leaf() and prev:
			return node



		taxa = []
		for c in node.child_nodes():
			l = next(c.traverse_leaves())
			taxa.append(l.label)

		parent = node.parent
		for c in parent.child_nodes():
			if c != node:
				l = next(c.traverse_leaves())
				taxa.append(l.label)

		# print(taxa)
		q = extract_quartet(s_tree, taxa+[t])
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
					return node

				for c in node.parent.child_nodes():
					if c != node:
						nextnode = c
						break
			else:
				dir="up"
				nextnode = node.parent
		if prev and prev != dir and not node.parent.is_root():
			return node

		prev = dir
		node = nextnode

def create_subtrees(tree, subs, taxa):
	outputs = []
	for i in range(len(subs)):
		t = taxa[(i+1)%len(subs)]
		n = subs[i]
		if n[1] == 0:
			# node = tree.label_to_node(selection='all')[n[0]]
			node = n[0]
			subtree = tree.extract_subtree(node)
			parent= Node(edge_length = 0, label = node.label)
			leaf = Node(edge_length = 0, label = t)
			parent.add_child(node)
			parent.add_child(leaf)
			# num_leaves[parent] = num_leaves[node] + 1
			# num_leaves[leaf] = 1
			# subtree = Tree()
			subtree.root = parent
			outputs.append(read_tree_newick(subtree.newick()))

		if n[1] == 1:
			node = n[0]
			for c in node.child_nodes():
				node.remove_child(c)
			leaf = Node(edge_length = 0, label = t)
			node.add_child(leaf)
			# num_leaves[node] = 1
			# num_leaves[leaf] = 1
			outputs.append(tree)

	return outputs

def divide_tree(tree, groups):
	outputs = []
	for i in range(len(groups)):
		g = groups[i]
		if len(g) == 1:
			node = g[0]
			node.parent = None
			subtree = Tree()
			subtree.root = node
			outputs.append(subtree)
		elif len(g) == 2:
			node = g[0].parent
			node.parent = None
			subtree = Tree()
			subtree.root = node
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


	

def merge_trees(s_tree, tree1, tree2, placements):
	tree1_num_leaves = compute_num_leaves(tree1)
	tree2 = reroot_middle(tree2)
	# print("="*200)
	# print("tree1")
	# print(tree1)
	# print("tree2")
	# print(tree2)

	#base cases

	num_leaves = tree1.num_nodes(leaves=True, internal=False)
	if num_leaves < 3:
		for n in tree1.traverse_preorder():
			if n.edge_length > 0:
				place = n.label
				break
		for l in tree2.traverse_leaves():
			placements[l.label] = place
		# print(placements)
		return

	num_leaves = tree2.num_nodes(leaves=True, internal=False)
	if num_leaves < 4:
		for l in tree2.traverse_leaves():
			place = find_taxon_placement(l.label, tree1, tree1_num_leaves, s_tree)
			if place.edge_length == 0:
				place = place.parent
			placements[l.label] = place.label
		return

	tree1_copy = tree1.newick()
	node = find_middle_branch(tree1, tree1_num_leaves)
	# print(node.label)
	
	#get three taxa for tree 1
	taxa = []
	tree1_subs = []
	for c in node.child_nodes():
		l = next(c.traverse_leaves())
		taxa.append(l.label)
		tree1_subs.append([c, 0])

	parent = node.parent
	tree1_subs.append([node, 1])
	for c in parent.child_nodes():
		if c != node:
			l = next(c.traverse_leaves())
			taxa.append(l.label)


	#get 4 taxa from tree 2
	tree2_taxa = []
	tree2_subs = []
	root = tree2.root
	for c in root.child_nodes():
		for cc in c.child_nodes():
			l = next(cc.traverse_leaves())
			tree2_taxa.append(l.label)
			tree2_subs.append([cc, 0])

	#assign subtrees
	assignments = []
	for t in tree2_taxa:
		q = extract_quartet(s_tree, taxa+[t])
		if [taxa[0],t] in q or [t,taxa[0]] in q:
			assignments.append(0)
		elif [taxa[1],t] in q or [t,taxa[1]] in q:
			assignments.append(1)
		elif [taxa[2],t] in q or [t,taxa[2]] in q:
			assignments.append(2)

	# print([n[0].label for n in tree1_subs])
	# print([n[0].label for n in tree2_subs])
	# print(assignments)

	counts = {}
	for i in range(len(assignments)):
		if assignments[i] not in counts:
			counts[assignments[i]] = [i]
		else:
			counts[assignments[i]].append(i)

	tree1_subtrees = create_subtrees(tree1, tree1_subs, taxa)

	if len(counts) == 1:
		# leaves = [l.label for l in tree2.traverse_leaves()]
		i = assignments[0]
		merge_trees(s_tree, tree1_subtrees[i], tree2, placements)
		return


	groups = [[tree2_subs[i][0] for i in counts[g]] for g in counts]
	tree2_subtrees = divide_tree(tree2, groups)

	max_group = max([len(counts[g]) for g in counts])

	if max_group < 3:
		i = 0
		for g in counts:
			merge_trees(s_tree, tree1_subtrees[g], tree2_subtrees[i], placements)
			i+=1
	else:
		i = 0
		for g in counts:
			if len(counts[g]) == 1:	
				new_tree = read_tree_newick(tree1_copy)
				merge_trees(s_tree, new_tree, tree2_subtrees[i], placements)
			else:
				merge_trees(s_tree, tree1_subtrees[g], tree2_subtrees[i], placements)
			i+=1

def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-s', '--stree', required=True, help="Input Species Tree")
	parser.add_argument('-i', '--input', required=True, help="Input Trees")
	# parser.add_argument('-g', '--gene_trees', required=True, help="Gene tree file")
	# parser.add_argument('-a', '--annot', required=False, help="Annotation file")
	# parser.add_argument('-n', '--num_genes', required=False, default=1, help="Number of gene trees")
	parser.add_argument('-o', '--outdir', required=False, default='./', help="Output dir")

	args = parser.parse_args()

	with open(args.stree, "r") as f:
		species_tree = read_tree_newick(f.readlines()[0])
		__label_tree__(species_tree)

	leaves = [l.label for l in species_tree.traverse_leaves()]
	taxa = random.sample(leaves, k=4)

	with open(args.input, "r") as f:
		lines = f.readlines()
		tree1 = read_tree_newick(lines[0])
		__label_tree__(tree1)
		tree2 = read_tree_newick(lines[1])
		__label_tree__(tree2)
	# print(tree1)

	# tree2 = reroot_middle(tree2, tree2_num_leaves)
	# tree2_num_leaves = compute_num_leaves(tree2)
	placements = {}
	merge_trees(species_tree, read_tree_newick(tree1.newick()), read_tree_newick(tree2.newick()), placements)
	print(placements)

	rev_placements = {}
	for p in placements:
		if placements[p] not in rev_placements:
			rev_placements[placements[p]] = []
		rev_placements[placements[p]].append(p)

	full_tree = read_tree_newick(tree1.newick())
	t=0
	for p in rev_placements:
		if len(rev_placements[p]) == 1:
			continue
			node = full_tree.label_to_node(selection='all')[p]
			parent = node.parent
			parent.remove_child(node)
			newparent = Node(label = 'p'+str(t), edge_length = 1)
			newleaf = Node(label = rev_placements[p][0], edge_length = 1)
			newparent.add_child(node)
			newparent.add_child(newleaf)
			parent.add_child(newparent)
			t+=1
		elif len(rev_placements[p]) == 2:
			continue
			node = tree1.label_to_node(selection='all')[p]
			leaf_down = next(node.traverse_leaves())
			leaf_up = [l for l in tree1.traverse_leaves() if l not in node.traverse_leaves()][0]
			q = extract_quartet(species_tree, rev_placements[p] + [leaf_up.label, leaf_down.label])
			node = full_tree.label_to_node(selection='all')[p]
			newleaf1 = Node(label = rev_placements[p][0], edge_length = 1)
			newleaf2 = Node(label = rev_placements[p][1], edge_length = 1)
			if [leaf_up.label, leaf_down.label] in q or [leaf_down.label, leaf_up.label] in q:
				newparent = Node(label = 'p'+str(t), edge_length = 1)
				newparent.add_child(newleaf1)
				newparent.add_child(newleaf2)
				t+=1

				parent = node.parent
				parent.remove_child(node)
				newgparent = Node(label = 'p'+str(t), edge_length = 1)
				newgparent.add_child(node)
				newgparent.add_child(newparent)
				parent.add_child(newgparent)
				t+=1

			elif [leaf_down.label, rev_placements[p][0]] in q or [rev_placements[p][0], leaf_down.label] in q:
				parent = node.parent
				parent.remove_child(node)

				newparent1 = Node(label = 'p'+str(t), edge_length = 1)
				newparent1.add_child(newleaf1)
				newparent1.add_child(node)
				t+=1

				newparent2 = Node(label = 'p'+str(t), edge_length = 1)
				newparent2.add_child(newleaf2)
				newparent2.add_child(newparent1)
				t+=1

				parent.add_child(newparent2)	
			else:
				parent = node.parent
				parent.remove_child(node)

				newparent1 = Node(label = 'p'+str(t), edge_length = 1)
				newparent1.add_child(newleaf2)
				newparent1.add_child(node)
				t+=1

				newparent2 = Node(label = 'p'+str(t), edge_length = 1)
				newparent2.add_child(newleaf1)
				newparent2.add_child(newparent1)
				t+=1

				parent.add_child(newparent2)			

		else:
			node = tree1.label_to_node(selection='all')[p]
			leaf_down = next(node.traverse_leaves())
			leaf_up = [l for l in tree1.traverse_leaves() if l not in node.traverse_leaves()][0]
			subtree = tree2.extract_tree_with(rev_placements[p])
			print(subtree)
			print(leaf_up.label, leaf_down.label)
			num_leaves = compute_num_leaves(subtree)
			up_node = find_taxon_placement(leaf_up.label, subtree, num_leaves, species_tree)
			down_node = find_taxon_placement(leaf_down.label, subtree, num_leaves, species_tree)
			print(up_node.label, down_node.label)
			subtree.root.edge_length = None
			root = subtree.root
			subtree.reroot(up_node, length = up_node.edge_length/2)
			root.contract()

			node = full_tree.label_to_node(selection='all')[p]
			if up_node == down_node:
				taxa1 = next(down_node.traverse_leaves())
				taxa2 = [l for l in subtree.traverse_leaves() if l not in down_node.traverse_leaves()][0]
				q = extract_quartet(species_tree, [leaf_up.label, leaf_down.label, taxa1.label, taxa2.label])
				if [leaf_up.label, leaf_down.label] in q or [leaf_down.label, leaf_up.label] in q:
					parent = node.parent
					parent.remove_child(node)
					newparent = Node(label = 'p'+str(t), edge_length = 1)
					newparent.add_child(subtree.root)
					newparent.add_child(node)
					parent.add_child(newparent)
					t += 1
				# elif [taxa1.label, leaf_down.label] in q or [leaf_down.label, taxa1.label] in q:
				else:
					sibling = [c for c in subtree.root.child_nodes() if c != down_node][0]
					print(down_node.label, sibling.label)
				# else:

			print(subtree)
			print(full_tree)

	return

	taxa = [l.label for l in tree2.traverse_leaves()]

	tree1_num_leaves = compute_num_leaves(tree1)
	true_placements = {}
	for i in range(len(taxa)):
		# print("="*200)
		# print(taxa[i])
		place = find_taxon_placement(taxa[i], tree1, tree1_num_leaves, species_tree)
		true_placements[taxa[i]] = place.label
	print(true_placements)


	for t in true_placements:
		if t not in placements:
			print(t, true_placements[t])
		elif placements[t] != true_placements[t]:
			print(t, true_placements[t], placements[t])

	# mid_node = find_middle_branch(tree1, tree1_num_leaves)
	# print(mid_node.label)


if __name__ == "__main__":
	main()  