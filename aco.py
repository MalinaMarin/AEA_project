import math
import numpy as np
import random
import copy
from numpy import mean
from parse import ParseFile


class Ant:
    class NodeForDrone:
        def __init__(self, visited_node, starting_node):
            self.distance = None
            self.visited_node = visited_node
            self.recover_node = -1
            self.starting_node = starting_node

    def __init__(self, alpha, beta, no_cities, edges, ph_drone_node, ph_truck_node, ct_customized):
        self.no_cities = no_cities
        self.edges = edges
        self.dist = 0
        self.ct_customized = ct_customized
        self.truck_nodes = None
        self.drone_nodes = None
        self.alpha = alpha
        self.beta = beta

        self.visited_nodes = [0]
        self.ph_drone_node = ph_drone_node
        self.ph_truck_node = ph_truck_node


    def choose_node(self):
        wheel_roulette = 0.0
        wheel_pos = 0.0
        nodes_to_discover = [node for node in range(self.no_cities) if node not in self.visited_nodes]
        for n in nodes_to_discover:
            wheel_roulette += pow(self.edges[self.truck_nodes[-1]][n].ph, self.alpha) * \
                              pow((1 / self.edges[self.truck_nodes[-1]][n].weight), self.beta)
        random_value = random.uniform(0.0, wheel_roulette)
        for n in nodes_to_discover:
            wheel_pos += pow(self.edges[self.truck_nodes[-1]][n].ph, self.alpha) * \
                         pow((1 / self.edges[self.truck_nodes[-1]][n].weight), self.beta)
            if wheel_pos >= random_value:
                return n

    def drone_choice(self, following_city):
        if (len(self.drone_nodes) >= len(self.truck_nodes) - 1 or (
                len(self.drone_nodes) > 0 and self.drone_nodes[-1].recover_node == -1)):
            return False

        truck_prob = pow(self.ph_truck_node[following_city], self.alpha) * \
                          pow((1 / self.edges[self.truck_nodes[-1]][following_city].weight), self.beta)
        drone_prob = pow(self.ph_drone_node[following_city], self.alpha) * \
                          pow((1 / (self.edges[self.truck_nodes[-1]][following_city].weight / self.ct_customized)), self.beta)
        choice = random.uniform(0.0, truck_prob + drone_prob)
        if choice > truck_prob:
            return True
        else:
            return False

    def compute_paths(self):
        self.truck_nodes = [0]
        initial_node = random.randint(1, self.no_cities - 1)
        self.truck_nodes.append(initial_node)
        self.drone_nodes = []
        self.visited_nodes = [0, initial_node]
        while len(self.truck_nodes) + len(self.drone_nodes) < self.no_cities:
            next_node = self.choose_node()
            self.visited_nodes.append(next_node)
            if self.drone_choice(next_node):
                self.drone_nodes.append(self.NodeForDrone(next_node, self.truck_nodes[-1]))
            else:
                if len(self.drone_nodes) > 0 and self.drone_nodes[-1].recover_node == -1:
                    self.drone_nodes[-1].recover_node = next_node
                self.truck_nodes.append(next_node)
        return [self.truck_nodes, self.drone_nodes]

    def compute_truck_dist(self):
        dist = 0.0
        for i in range(len(self.truck_nodes) - 1):
            dist += self.edges[self.truck_nodes[i]][self.truck_nodes[(i + 1) % self.no_cities]].weight
        return dist

    def compute_drone_dist(self):
        if len(self.drone_nodes) > 0 and self.drone_nodes[-1].recover_node == -1:
            self.drone_nodes[-1].recover_node = self.truck_nodes[0]

        for i in range(len(self.drone_nodes)):
            self.drone_nodes[i].dist = self.edges[self.drone_nodes[i].starting_node][self.drone_nodes[i].visited_node].weight / self.ct_customized + \
                                       self.edges[self.drone_nodes[i].visited_node][self.drone_nodes[i].recover_node].weight / self.ct_customized

    def get_distance(self):
        self.dist = 0.0
        self.compute_drone_dist()
        positions = 0
        for i in range(len(self.truck_nodes)):
            if (len(self.drone_nodes) > positions and self.truck_nodes[(i + 1) % len(self.truck_nodes)] ==
                    self.drone_nodes[positions]):
                truck_distance = self.edges[self.truck_nodes[i]][self.truck_nodes[(i + 1) % len(self.truck_nodes)]].weight
                self.dist += max(truck_distance, self.drone_nodes[positions].dist)
                positions += 1
            else:
                self.dist += self.edges[self.truck_nodes[i]][self.truck_nodes[(i + 1) % len(self.truck_nodes)]].weight
        return self.dist


class ACO:
    class Edge:
        def __init__(self, a, b, weight, initial_ph):
            self.a = a
            self.b = b
            self.weight = weight
            self.ph = initial_ph

    def __init__(self, colonysize=100, alpha=1.2, beta=3.1,
                 rho=0.1, ph_depozit_w=5.0, initial_pheromone=1.0, steps=2, cities_no=None,
                 ct_customized=2):
        self.colony_size = colonysize
        self.rho = rho
        self.ph_depozit_w = ph_depozit_w
        self.steps = steps
        self.no_cities = len(cities_no)
        self.nodes = nodes
        self.ct_customized = ct_customized
        self.drone_ph = []
        self.truck_ph = []
        self.edges = [[None] * self.no_cities for _ in range(self.no_cities)]
        for i in range(self.no_cities):
            for j in range(i + 1, self.no_cities):
                self.edges[i][j] = self.edges[j][i] = self.Edge(i, j, math.sqrt(
                    pow(self.nodes[i][0] - self.nodes[j][0], 2.0) + pow(self.nodes[i][1] - self.nodes[j][1], 2.0)),
                                                                initial_pheromone)
        for i in range(self.no_cities):
            self.drone_ph.append(initial_pheromone)
            self.truck_ph.append(initial_pheromone)
        self.ants = [Ant(alpha, beta, self.no_cities, self.edges, self.drone_ph, self.truck_ph,
                         self.ct_customized) for _ in range(self.colony_size)]
        self.best_solution = float("inf")
        self.best_itinerary = None

    def compute_ph(self, tour, distance, weight=1.0):
        pheromone_to_add = self.ph_depozit_w / distance
        for i in range(len(tour[0])):
            self.truck_ph[tour[0][i]] = weight * pheromone_to_add
            self.edges[tour[0][i]][tour[0][(i + 1) % len(tour[0])]].ph += weight * pheromone_to_add
        for i in range(len(tour[1])):
            self.drone_ph[tour[1][i].visited_node] = weight * pheromone_to_add

    def do_aco(self):
        for step in range(self.steps):
            for ant in self.ants:
                self.compute_ph(ant.compute_paths(), ant.get_distance())
                if ant.dist < self.best_solution:
                    self.best_itinerary = [ant.truck_nodes, ant.drone_nodes]
                    self.best_solution = ant.dist
            for i in range(self.no_cities):
                for j in range(i + 1, self.no_cities):
                    self.edges[i][j].ph *= (1.0 - self.rho)
                self.drone_ph[i] *= (1.0 - self.rho)
                self.truck_ph[i] *= (1.0 - self.rho)


colony_size = 100
steps = 70

path = 'uniform/'
data_paths = ["uniform/uniform-80-n50.txt"]
p = ParseFile(data_paths[0])
points = np.array([p.X, p.Y])
nodes = points.T

all_distances = []
best_distance = float("inf")
best_tour = ACO(colonysize=colony_size, cities_no=nodes)
for i in range(30):
    acs = ACO(colonysize=colony_size, cities_no=nodes)
    acs.do_aco()
    all_distances.append(acs.best_solution)
    if acs.best_solution < best_distance:
        best_tour = copy.deepcopy(acs)
        best_distance = acs.best_solution

print(all_distances)
print(mean(all_distances))
print(best_distance)