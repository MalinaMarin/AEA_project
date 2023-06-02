import numpy as np


class GeneticAlgorithm:
    def __init__(self):
        self.popsize = 100
        self.mutation = 0.02
        self.crossover = 0.01
        self.chromosome = 0.02

        self.no_cities = None
        self.max_fitness_values = []
        self.min_eval_values = []

        self.population = None
        self.chromosome_size = None
        self.parents = None

        self.d_positions = [[0, 0], [1, 0], [0, 1]]
        self.best_solution = []

    def set_mutation_drone(self, prob):
        self.mutation_drone = prob

    def set_chromosome(self, prob):
        self.chromosome = prob

    def set_cities(self, cities_nb):
        self.no_cities = cities_nb - 1
        self.chromosome_size = cities_nb + 1

    def set_nb_iterations(self, nb_iterations):
        self.max_it = nb_iterations

    def set_parameters(self, nb_cities):
        self.no_cities = nb_cities - 1
        self.chromosome_size = nb_cities + 1

    def set_mutation(self, prob):
        self.mutation = prob

    def set_popsize(self, prob):
        self.popsize = prob
        self.population = np.array([self.generate_instance() for _ in range(prob)])

    def generate_bits(self):
        d_pos = [[0, 0], [1, 0], [0, 1]]
        drone_final = []
        for i in range(int(self.no_cities / 2)):
            if len(drone_final) == 0:
                drone_final += d_pos[np.random.randint(0, 3)]
            else:
                if sum(drone_final[-2:]) == 0:
                    drone_final += d_pos[1]
                elif drone_final[-1] == 1:
                    tp = np.random.randint(0, 2)
                    if tp == 1:
                        drone_final += d_pos[2]
                    else:
                        drone_final += d_pos[0]
                else:
                    drone_final += d_pos[1]
        if self.no_cities % 2 == 1:
            drone_final.append(0)
        drone_final = [1] + drone_final
        drone_final.append(1)
        return drone_final

    def generate_instance(self):
        cities = np.array([0])
        element = np.arange(1, self.no_cities + 1)
        np.random.shuffle(element)
        cities = np.append(cities, element)
        cities = np.append(cities, [0])
        bits = self.generate_bits()
        return np.array([cities, bits])

    def compute_parents(self, parents):
        result = []
        pairs_size = len(parents) / 2
        while len(result) < pairs_size:
            p_one = np.random.randint(len(parents))
            p_two = np.random.randint(len(parents))
            if p_one != p_two:
                result.append([parents[p_one], parents[p_two]])
                if p_one > p_two:
                    np.delete(parents, p_one)
                    np.delete(parents, p_two)
                else:
                    np.delete(parents, p_two)
                    np.delete(parents, p_one)
        prob = np.random.uniform(0, 1, len(result))
        result = np.array(result)
        return zip(result, prob)

    def cross_over(self, parents):
        children = []
        parents = self.compute_parents(parents)
        for pair, prob in parents:
            if prob > self.crossover:
                crosspoint = np.random.randint(1, self.chromosome_size - 3)
                aux = pair[:, :, 1:pair.shape[2] - 1]
                param = pair.shape[2] - 2
                result = []
                for el in aux[:, 0]:
                    tmp = np.arange(1, param + 1)
                    r = []
                    for i in el:
                        idx = tmp.tolist().index(i)
                        tmp = np.delete(tmp, idx)
                        r.append(idx + 1)
                    result.append(r)

                cross = [result[0][:crosspoint] + result[1][crosspoint:],
                         result[1][:crosspoint] + result[0][crosspoint:]]

                output = []
                for el in cross:
                    tmp = np.arange(1, param + 1)
                    r = []
                    for i in el:
                        idx = tmp[i - 1]
                        tmp = np.delete(tmp, i - 1)
                        r.append(idx)
                    output.append(r)

                pair[:, 0, 1:pair.shape[2] - 1] = output
                for p in pair:
                    children.append(p)
            else:
                for p in pair:
                    children.append(p)
        return np.array(children)

    def mutate(self, elm):
        if np.random.uniform(0, 1) < self.mutation:
            a = np.random.randint(1, self.chromosome_size - 1)
            b = np.random.randint(1, self.chromosome_size - 1)
            tmp = elm[0][a]
            elm[0][a] = elm[0][b]
            elm[0][b] = tmp
        return elm

    def mutation(self, pop):
        for c in range(len(pop)):
            if np.random.uniform(0, 1) < self.chromosome:
                pop[c] = self.mutate(pop[c])

    def get_fitness(self):
        min_fitness = np.inf
        max_fitness = -np.inf
        evaluated = np.zeros(self.popsize)
        for i in range(self.popsize):
            evaluated[i] = self.eval.compute_score(self.population[i])
            if evaluated[i] < min_fitness:
                min_fitness = evaluated[i]
            if evaluated[i] > max_fitness:
                max_fitness = evaluated[i]

        # penalizing big distances
        fitness = max_fitness - evaluated
        self.best_solution = self.population[np.argmin(evaluated)]
        self.min_eval_values.append(evaluated[np.argmin(evaluated)])
        self.max_fitness_values.append(max(fitness))
        return fitness

    def select(self, q, pick):
        for i in range(0, len(q) - 1):
            if q[i] <= pick < q[i + 1]:
                return i

    def selection(self, fitness):
        q = [0.0]
        new_pop = []
        p = fitness / np.sum(fitness)
        for i in range(1, self.popsize + 1):
            q.append(q[i - 1] + p[i - 1])
        q.append(1.1)
        for i in range(self.popsize):
            pos = np.random.uniform(0, 1)
            new_pop.append(self.population[self.select(q, pos)])
        return new_pop

    def set_evaluation_function(self, eval):
        self.eval = eval

    def run(self, i):
        generation = 1

        while generation < self.max_it:
            fitness = self.get_fitness()
            new_pop = self.selection(fitness)
            children = self.cross_over(new_pop)
            self.mutation(children)
            if generation % 5 == 0:
                print(str(i), generation, min(self.min_eval_values), "Fitness:",
                      min(self.max_fitness_values))
            generation += 1
        print(self.best_solution)
        print(min(self.min_eval_values))
        return self.min_eval_values, self.max_fitness_values, self.best_solution
