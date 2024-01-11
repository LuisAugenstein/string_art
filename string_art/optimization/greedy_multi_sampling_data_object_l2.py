import numpy as np
from scipy.sparse import csr_matrix, find
import matplotlib.pyplot as plt
from string_art.optimization.optimized_loss import OptimizedLoss
from string_art.optimization.simple_loss import SimpleLoss

MIN_CIRCLE_LENGTH = 1


class GreedyMultiSamplingDataObjectL2:
    def __init__(self, img: np.ndarray, importance_map: np.ndarray,  A_high_res: csr_matrix, A_low_res: csr_matrix, valid_edges_mask: np.ndarray):
        """
        img: np.shape([low_res, low_res])
        importance_map: np.shape([low_res, low_res])
        A_high_res: np.shape([high_res**2, n_edges])
        A_low_res: np.shape([low_res**2, n_edges])
        """
        self.n_edges = A_high_res.shape[1]
        self.low_res = np.sqrt(A_low_res.shape[0]).astype(int)
        self.high_res = np.sqrt(A_high_res.shape[0]).astype(int)

        self.loss = OptimizedLoss(img, np.ones_like(importance_map), A_high_res, A_low_res)
        self.simple_loss = SimpleLoss(img, np.ones_like(importance_map), A_high_res, self.low_res)
        self.valid_edges_mask = valid_edges_mask

        self.removalMode = False
        self.x = np.zeros(self.n_edges)
        self.picked_edges_sequence = np.zeros(0, dtype=int)

    @property
    def removable_edge_indices(self) -> np.ndarray:
        return np.where(self.x == 1)[0]

    @property
    def addable_edge_indices(self) -> np.ndarray:
        return np.where((self.x == 0) & self.valid_edges_mask)[0]

    def find_best_string(self) -> tuple[np.ndarray, int]:
        f_scores = self.loss.get_f_scores(self.x, mode='remove' if self.removalMode else 'add')
        candidate_edge_indices = self.removable_edge_indices if self.removalMode else self.addable_edge_indices
        i_next_edge = candidate_edge_indices[np.argmin(f_scores[candidate_edge_indices])]
        loss_value = f_scores[i_next_edge]
        print(f'\tF1 when picking edge Nr. {i_next_edge}: {loss_value:16.16f}')
        return loss_value, i_next_edge

    def choose_string_and_update(self, i):
        self.loss.update_f_scores(i)

        if self.removalMode:
            self.x[i] = 0
            self.picked_edges_sequence = self.picked_edges_sequence[self.picked_edges_sequence != i]
        else:
            self.x[i] = 1
            self.picked_edges_sequence = np.hstack((self.picked_edges_sequence.T, [i])).T

        print(f'\tF2 when picking edge Nr. {i}: {np.sum(self.loss.residual**2):16.16f}\n\n')

    def compute_illegal_edge_indices(self, hook, illegal_pins: np.ndarray):
        if hook == illegal_pins:
            return np.zeros(0, dtype=int)

        if illegal_pins.shape[0] == 1:
            illegal_pins = np.column_stack((illegal_pins, illegal_pins))

        lately_visited_indices = np.tile(illegal_pins.T, (self.edges_to_pins.shape[0], 1))
        lately_from = np.any(lately_visited_indices == np.tile(self.edges_to_pins[:, 0], (1, illegal_pins.shape[1])), axis=1)
        lately_to = np.any(lately_visited_indices == np.tile(self.edges_to_pins[:, 1], (1, illegal_pins.shape[1])), axis=1)

        curr = np.tile(hook, (self.edges_to_pins.shape[0], 1))
        curr_from = curr == np.tile(self.edges_to_pins[:, 0], (1, 1))
        curr_to = curr == np.tile(self.edges_to_pins[:, 1], (1, 1))

        res = np.logical_or(np.logical_and(lately_from, curr_to), np.logical_and(lately_to, curr_from))
        k, _, _ = np.where(res)

        return k

    def show_current(self):
        plt.figure(1)
        plt.imshow(np.flipud(np.reshape(1 - self.current_recon, [self.high_res, self.high_res]).T), cmap='gray')
        plt.show()

    def get_rmse_value(self):
        return self.loss.rmse_value

    def set_removal_mode(self, mode):
        if mode != self.removalMode:
            self.removalMode = mode

    def remove_overshoot(self, num_edges):
        self.set_removal_mode(True)
        for k in range(1, num_edges + 1):
            print(f'Removing string {k} of {num_edges}')
            i = self.picked_edges_sequence[-1]

            print(f'\tF1 when picking edge Nr. {i}: {self.f_removing[i]:16.16f}')
            self.choose_string_and_update(i)
