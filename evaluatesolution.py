import numpy as np


class EvaluateSolution:
    def __init__(self, truck, drone):
        self.truck = truck
        self.drone = drone

    def compute_score(self, path, show_path=False):
        current_path = [0]
        path = path.astype(int)

        for i in range(path.shape[1]):
            if path[1][i] == 1:
                if path[1][i - 1] == 0:
                    current_path.append(i)
        copy_of_path = current_path[1:].copy()
        copy_of_path.append(0)

        final_road = []
        final_road_types = []
        complete_path = []
        path_type = []
        for i in range(len(current_path) - 1):
            aux = []
            aux_type = []
            info = path[0][current_path[i] + 1: copy_of_path[i]]
            info_t_or_d = path[1][current_path[i] + 1: copy_of_path[i]]

            initial_point_truck = path[0][current_path[i]]
            initial_point_drone = path[0][current_path[i]]

            has_drone = True

            for t in range(len(info_t_or_d)):
                if not has_drone:
                    aux.append([initial_point_truck, info[t]])
                    aux_type.append(0)
                    initial_point_truck = info[t]

                else:
                    has_drone = False
                    aux.append([initial_point_drone, info[t]])
                    initial_point_drone = info[t]
                    aux_type.append(1)

            aux.append([initial_point_drone, path[0][copy_of_path[i]]])
            aux.append([initial_point_truck, path[0][copy_of_path[i]]])
            aux_type.append(1)
            aux_type.append(0)
            final_road_types.append(aux_type)
            final_road.append(aux)
            complete_path += aux
            path_type += aux_type

        d = 0
        for i in range(len(final_road)):
            path = np.array(final_road[i])
            temp = np.array(final_road_types[i])

            truck_position = np.where(temp == 0)

            drone_position = np.where(temp == 1)

            only_drone_pos = path[drone_position]
            only_truck_pos = path[truck_position]

            if len(only_truck_pos) == 1:
                d += max(self.truck[only_truck_pos[0][0], only_truck_pos[0][1]].sum(),
                         self.drone[only_drone_pos[:-1], only_drone_pos[1:]].sum())
            else:
                d += max(self.truck[only_truck_pos[:-1], only_truck_pos[1:]].sum(),
                         self.drone[only_drone_pos[:-1], only_drone_pos[1:]].sum())
        if show_path:
            return d, complete_path, path_type
        return d
