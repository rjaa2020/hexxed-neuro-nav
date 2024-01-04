import enum

import numpy as np


class GraphTemplate(enum.Enum):
	two_step = "two_step"
	ring = "ring"
	two_way_linear = "two_way_linear"
	linear = "linear"
	t_graph = "t_graph"
	neighborhood = "neighborhood"
	human_a = "human_a"
	human_b = "human_b"
	t_loop = "t_loop"
	variable_magnitude = "variable_magnitude"
	three_arm_bandit = "three_arm_bandit"
	hexxed = "hexxed_graph"


def two_step():
	reward_locs = {3: 1, 4: -1, 5: 0.5, 6: 0.5}
	edges = [[1, 2], [3, 4], [5, 6], [], [], [], []]
	objects = {"rewards": reward_locs}
	return objects, edges


def three_arm_bandit():
	reward_locs = {1: 1, 2: 0.5, 3: -0.5}
	edges = [[1, 2, 3], [], [], []]
	objects = {"rewards": reward_locs}
	return objects, edges


def two_way_linear():
	reward_locs = {4: 1}
	edges = [[0, 1], [0, 2], [1, 3], [2, 4], [3, 4]]
	objects = {"rewards": reward_locs}
	return objects, edges


def ring():
	reward_locs = {4: 1}
	edges = [[1, 5], [0, 2], [1, 3], [2, 4], [3, 5], [4, 0]]
	objects = {"rewards": reward_locs}
	return objects, edges


def linear():
	reward_locs = {5: 1}
	edges = [[1], [2], [3], [4], [5], []]
	objects = {"rewards": reward_locs}
	return objects, edges


def t_graph():
	reward_locs = {5: 1}
	edges = [[1, 0], [2, 1], [3, 4], [5, 3], [6, 4], [], []]
	objects = {"rewards": reward_locs}
	return objects, edges


def neighborhood():
	reward_locs = {14: 1}
	edges = [
		[1, 2, 3, 4],
		[0, 2, 3, 4],
		[5, 1, 0, 4],
		[10, 4, 0, 1],
		[0, 1, 2, 3],
		[2, 6, 8, 9],
		[5, 8, 7, 9],
		[5, 9, 6, 8],
		[7, 6, 5, 9],
		[6, 7, 8, 11],
		[3, 12, 13, 14],
		[9, 12, 13, 14],
		[10, 11, 13, 14],
		[11, 10, 12, 14],
		[10, 12, 11, 13],
	]
	objects = {"rewards": reward_locs}
	return objects, edges


def human_a():
	reward_locs = {4: 10, 5: 1}
	edges = [[2], [3], [4], [5], [], []]
	objects = {"rewards": reward_locs}
	return objects, edges


def human_b():
	reward_locs = {3: 15, 5: 30}
	edges = [[1, 2], [3, 4], [4, 5], [3, 3], [4, 4], [5, 5]]
	objects = {"rewards": reward_locs}
	return objects, edges


def t_loop():
	reward_locs = {12: 1, 11: 1}
	edges = [
		[1, 0],
		[2, 1],
		[3, 4],
		[5, 3],
		[6, 4],
		[7, 5],
		[8, 6],
		[9, 7],
		[10, 8],
		[11, 9],
		[12, 10],
		[0, 11],
		[0, 12],
	]
	objects = {"rewards": reward_locs}
	return objects, edges


def variable_magnitude():
	# Values taken from original author's code availabe here: https://osf.io/ux5rg/
	fmax = 10.0
	sigma = 200
	utility_func = lambda r: (fmax * np.sign(r) * np.abs(r) ** (0.5)) / (
			np.abs(r) ** (0.5) + sigma ** (0.5)
	)
	reward_locs = {
		1: utility_func(0.1),
		2: utility_func(0.3),
		3: utility_func(1.2),
		4: utility_func(2.5),
		5: utility_func(5),
		6: utility_func(10),
		7: utility_func(20),
	}
	edges = [
		[((1, 2, 3, 4, 5, 6, 7), (0.067, 0.090, 0.148, 0.154, 0.313, 0.151, 0.077))],
		[],
		[],
		[],
		[],
		[],
		[],
		[],
	]
	objects = {"rewards": reward_locs}
	return objects, edges

import numpy as np

class HexxedGraph:
	def __init__(self, inner_layers=6, nodes_per_layer=7):
		assert inner_layers > 0, "There must be at least 1 inner layer"
		assert nodes_per_layer > 3, "There must be at least 4 nodes per layer"

		self.inner_layers = inner_layers
		self.nodes_per_layer = nodes_per_layer

		self.graph = []

		node_num = 0
		for layer in range(inner_layers + 2):
			if layer % (inner_layers + 1) == 0:
				self.graph.append([node_num])
				node_num += 1
			else:
				self.graph.append(list(range(node_num, node_num + nodes_per_layer)))
				node_num += nodes_per_layer

		self.flat_graph = []
		for layer in self.graph:
			self.flat_graph += layer

		self.edges = {}
		for n in self.flat_graph:
			self.edges[n] = self.get_edges(n)

		self.rewards = self.get_reward_locs()

	def visualize(self):
		max_width = max(len(str(item)) for row in self.graph for item in row) * (self.nodes_per_layer - 1) + (self.nodes_per_layer - 2) # 6 elements and 5 spaces

		for layer in self.graph:
			if len(layer) == 1:
				print(f"{str(layer[0]):02}".rjust(max_width + 6))
			else:
				print(" ".join(f"{item:>{max_width // self.nodes_per_layer}}" for item in layer[:-1]) + f" |  {layer[-1]:02}    <==  Layer {layer[-1] // self.nodes_per_layer:02}")

	def get_edges(self, n):
		layer = (n // self.nodes_per_layer) + 1                # determine which layer node is in
		pos = n % self.nodes_per_layer                         # determine where in layer node is


		if n == 0:                                        # if starting state
			edges = list(np.arange(1, self.nodes_per_layer))

		elif pos == 0 or n == (self.nodes_per_layer * self.inner_layers) + 1:     # if one of the terminal states or last state
			edges = []

		elif layer == self.inner_layers:                       # if in the final layer
			edges = [self.nodes_per_layer * self.inner_layers + 1]

			if pos == (self.nodes_per_layer - 1):                # if last layer and last node in layer
				edges = np.append(edges, (n+1))

		else:
			edges = np.array([pos - 1, pos, pos + 1]) + (layer * self.nodes_per_layer)

			if pos == 1:                                    # if first node in layer
				edges[0] += (self.nodes_per_layer - 1)
			elif pos == (self.nodes_per_layer - 1):              # if last node in layer
				edges[2] += 1
				edges = np.append(edges, (n+1))

		return list(edges)

	def get_reward_locs(self):
		self.reward_locs = {}

		for n in self.flat_graph:
			if n == self.flat_graph[-1]:
				self.reward_locs[n] = 0
			elif n % self.nodes_per_layer == 0:
				self.reward_locs[n] = (n // self.nodes_per_layer)**2

		return self.reward_locs

def hexxed_graph():
	g = HexxedGraph(inner_layers=1, nodes_per_layer=4)

	edges = list(g.edges.values())
	reward_locs = g.rewards
	objects = {"rewards": reward_locs}

	return objects, edges


template_map = {
	GraphTemplate.two_step          : two_step,
	GraphTemplate.two_way_linear    : two_way_linear,
	GraphTemplate.ring              : ring,
	GraphTemplate.linear            : linear,
	GraphTemplate.t_graph           : t_graph,
	GraphTemplate.neighborhood      : neighborhood,
	GraphTemplate.human_a           : human_a,
	GraphTemplate.human_b           : human_b,
	GraphTemplate.t_loop            : t_loop,
	GraphTemplate.variable_magnitude: variable_magnitude,
	GraphTemplate.three_arm_bandit  : three_arm_bandit,
	GraphTemplate.hexxed            : hexxed_graph
}
