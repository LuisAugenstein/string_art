import numpy as np
from scipy.sparse import csr_matrix, find
import matplotlib.pyplot as plt
from string_art.optimization.losses import Loss

MIN_CIRCLE_LENGTH = 1


class GreedyMultiSamplingDataObjectL2:
    def __init__(self, loss: Loss, valid_edges_mask: np.ndarray):
        """
        img: np.shape([low_res, low_res])
        importance_map: np.shape([low_res, low_res])
        A_high_res: np.shape([high_res**2, n_edges])
        A_low_res: np.shape([low_res**2, n_edges])
        """
        self.loss = loss
        self.valid_edges_mask = valid_edges_mask

        self.removalMode = False
        self.x = np.zeros_like(valid_edges_mask, dtype=int)
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
