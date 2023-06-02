import numpy as np
from genetic_algorithm import GeneticAlgorithm
from evaluatesolution import EvaluateSolution
from parse import ParseFile
import multiprocessing

iterations = 50
popsize = 50
samples = 30

GeneticAlgorithm = GeneticAlgorithm()
EvaluateSolution = EvaluateSolution

save_path = 'uniform/'

instance_path = [
    "uniform/uniform-80-n50.txt"
]

p = ParseFile(instance_path[0])

points = np.array([p.X,p.Y])
points = points.T

truck_dist =[]
for i in points:
    d = np.linalg.norm(i - points, axis=1)
    truck_dist.append(d.tolist())
truck_dist = np.array(truck_dist)

d_distances = []
for i in points:
    d = np.linalg.norm(i - points, axis=1) * p.d_speed
    d_distances.append(d.tolist())
d_distances = np.array(d_distances)

no_parameters = points.shape[0]
GeneticAlgorithm.set_evaluation_function(EvaluateSolution(truck_dist, d_distances))
GeneticAlgorithm.set_parameters(no_parameters)
GeneticAlgorithm.set_popsize(popsize)
GeneticAlgorithm.set_nb_iterations(iterations)


if __name__ == '__main__':
    experiments = list(range(samples))
    with multiprocessing.Pool(int(samples)) as p:
        GeneticAlgorithm.set_parameters(no_parameters)
        GeneticAlgorithm.set_popsize(popsize)
        GeneticAlgorithm.set_nb_iterations(iterations)
        result = p.map(GeneticAlgorithm.run, experiments)
    min_eval_values, fitness, best_solution = zip(*result)
    np.savez(save_path + "ga_" + str(no_parameters) + "_eval_vals", min_eval_values)
    np.savez(save_path + "ga_" + str(no_parameters) + "_fitness_vals", fitness)
    np.savez(save_path + "ga_" + str(no_parameters) + "_best_solutions", best_solution)
