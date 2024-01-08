import numpy as np
from string_art.optimization.greedy_multi_sampling_data_object_l2 import GreedyMultiSamplingDataObjectL2
from scipy.sparse import csr_matrix
from string_art.transformations import PinEdgeTransformer


def optimize_strings_greedy_multi_sampling(img: np.ndarray, importance_map: np.ndarray, A_high_res: csr_matrix, A_low_res: csr_matrix, pin_edge_transformer: PinEdgeTransformer):
    obj = GreedyMultiSamplingDataObjectL2(img, importance_map, A_high_res, A_low_res, pin_edge_transformer)

    iterative_step_size = 1
    iteration = 0
    best_rmse_value = np.inf
    best_num_edges = 0
    num_bad_runs = 0
    max_num_strings = 100000

    for iteration in range(1, max_num_strings+1):
        print(f'Iteration Nr. {iteration}')

        m, i = obj.find_best_string()

        if i is None:
            break

        if isinstance(i, np.ndarray) and i.shape[0] > 1:
            i = np.random.choice(i, 1)

        obj.choose_string_and_update(i)

        rmse = obj.get_rmse_value()

        if rmse < best_rmse_value:
            best_rmse_value = rmse
            best_num_edges = iteration
            num_bad_runs = 0
        else:
            num_bad_runs += 1

        if num_bad_runs == 1000:
            break

    pure_greedy_best_rmse = best_rmse_value
    picked_edges_sequence = obj.picked_edges_sequence[:best_num_edges]

    if iterative_step_size > 0:
        # Try to improve result by iterative greedy approach

        # Remove 'overshoot'
        obj.remove_overshoot(len(obj.picked_edges_sequence) - best_num_edges)

        # Iterative greedy approach
        num_bad_runs = 0
        modes = [True, False]

        removed_edge_indices = np.zeros(iterative_step_size, dtype=int)
        added_edge_indices = np.zeros(iterative_step_size, dtype=int)

        while num_bad_runs < 1000:
            for k in range(2):
                if modes[k]:
                    print('Iterative Removal...')
                else:
                    print('Iterative Addition...')

                obj.set_removal_mode(modes[k])
                for s in range(iterative_step_size):
                    m, i = obj.find_best_string()

                    if i is None:
                        break

                    if i.shape[0] > 1:
                        i = np.random.choice(i, 1)

                    obj.choose_string_and_update(i)

                    if k == 0:
                        # Removal Stage
                        removed_edge_indices[s] = i
                    else:
                        # Adding Stage
                        added_edge_indices[s] = i

            if np.array_equal(added_edge_indices, removed_edge_indices):
                print('INFO: Breaking iterative step due to equal removed and added strings')
                break

            # Remove edges as long as there is an improvement
            obj.set_removal_mode(True)

            condition = True
            min_l2 = obj.diff_to_blank_squared_error_sum

            print('Try to improve by Removal...')
            while condition:
                val, i = obj.find_best_string()
                if val < min_l2:
                    obj.choose_string_and_update(i)
                    min_l2 = obj.diff_to_blank_squared_error_sum
                else:
                    condition = False

            # Add edges as long as there is an improvement
            obj.set_removal_mode(False)

            condition = True
            min_l2 = obj.diff_to_blank_squared_error_sum

            print('Try to improve by Addition...')
            while condition:
                val, i = obj.find_best_string()

                if val < min_l2:
                    obj.choose_string_and_update(i)
                    min_l2 = obj.diff_to_blank_squared_error_sum
                else:
                    condition = False

            rmse = obj.get_rmse_value()

            if rmse < best_rmse_value:
                best_rmse_value = rmse
                num_bad_runs = 0
                picked_edges_sequence = obj.picked_edges_sequence
            else:
                num_bad_runs += 1

    print(f'INFO: Iterative greedy could improve RMSE...\nFrom {pure_greedy_best_rmse}\nTo {best_rmse_value}')

    x = np.zeros_like(obj.x)
    x[picked_edges_sequence] = 1

    return x, picked_edges_sequence
