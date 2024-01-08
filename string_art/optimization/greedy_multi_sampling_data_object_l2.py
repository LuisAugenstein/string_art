import numpy as np
from string_art.optimization.multi_sample_correspondence_map import multi_sample_correspondence_map
from string_art.optimization.split_apart_string_matrix import split_apart_string_matrix
from scipy.sparse import csr_matrix, find
from string_art.transformations import PinEdgeTransformer, indices_1D_high_res_to_low_res, indices_1D_low_res_to_high_res, indices_1D_to_2D, indices_2D_to_1D
from string_art.preprocessing import create_circular_mask
import matplotlib.pyplot as plt
from string_art.entities import ConnectionType

MIN_CIRCLE_LENGTH = 1


class GreedyMultiSamplingDataObjectL2:
    def __init__(self, img: np.ndarray, super_sampling_factor: int,  min_angle: float, n_pins: int, importance_map: np.ndarray | None,  fabricable_edges: np.ndarray, A_high_res: csr_matrix, A_low_res: csr_matrix):
        """
        img: np.shape([low_res, low_res])
        importance_map: np.shape([low_res, low_res])
        fabricable_edges: np.shape([n_edges], dtype=bool) with n_edges=4*comb(n_pins, 2) stating which of the possible edges are fabricable
        """
        low_res = img.shape[0]
        self.n_pins = n_pins
        self.pin_edge_transformer = PinEdgeTransformer(n_pins, fabricable_edges)
        self.n_edges = A_high_res.shape[1]
        self.low_res = low_res
        self.high_res = low_res * super_sampling_factor
        self.matrixPath = 'matrix_path'

        self.b_native_res = img.T.flatten()
        importance_map = np.ones((low_res, low_res)) if importance_map is None else importance_map
        self.importance_map = importance_map.T.flatten()

        self.currentReconSquare = np.zeros((self.high_res, self.high_res))

        self.fabricable_edges = fabricable_edges
        # Note in the original algorithm the following two lines should be executed with unfiltered matrices, i.e., including the non-fabricable edges
        self.highResEdgePixelIndices, self.highResEdgePixelValues, self.highResCorrespEdgeIndices, self.highResIndexToIndexMap = split_apart_string_matrix(
            A_high_res)
        self.low_res_edge_pixel_indices, self.low_res_edge_pixel_values, self.low_res_corresp_edge_indices, self.lowResIndexToIndexMap = split_apart_string_matrix(
            A_low_res)
        self.A_edge_indices_to_pixel_codes = self.load_a_index_matrices(A_high_res)
        self.A_high_res = A_high_res
        self.A_low_res = A_low_res

        self.reachablePixelsMask = np.zeros(self.high_res * self.high_res, dtype=bool)
        self.reachablePixelsMask[self.highResEdgePixelIndices] = True

        self.reachablePixelsMaskNativeRes = np.zeros(low_res**2, dtype=bool)
        self.reachablePixelsMaskNativeRes[self.low_res_edge_pixel_indices] = True

        if min_angle > 0:
            self.compute_core_zone_pixels(min_angle, low_res)

        self.filter_weight = 1.0 / (super_sampling_factor * super_sampling_factor)

        self.highResReIndexMap = np.zeros(self.high_res**2)
        self.lowResReIndexMap = np.zeros(self.low_res**2)

        self.removalMode = False

        self.allIndices = np.arange(self.high_res) + 1

        self.x = np.zeros((self.n_edges, 1))
        self.pin_count = np.zeros(n_pins)
        self.picked_edges_sequence = np.zeros(0, dtype=int)

        self.stringList = np.zeros((0, 3), dtype=int)

        self.current_pin = 0

        self.init_state_vectors()
        self.init_lately_visited_pins()

    @property
    def current_recon_unclamped(self) -> np.ndarray:
        """np.shape([high_res**2]) with binary values which pixel is used and which one not"""
        return self.A_high_res @ self.x

    @property
    def current_recon(self) -> np.ndarray:
        return np.minimum(1.0, self.current_recon_unclamped)

    @property
    def current_recon_native_res(self) -> np.ndarray:
        corr_map = multi_sample_correspondence_map(self.low_res, self.high_res)
        return corr_map @ self.current_recon

    @property
    def residual(self) -> np.ndarray:
        return self.importance_map.multiply(self.b_native_res - self.current_recon_native_res).A

    @property
    def removable_edge_indices(self) -> np.ndarray:
        return find(self.x)[0]

    def load_a_index_matrices(self, A_high_res: csr_matrix) -> list[tuple[np.ndarray, np.ndarray]]:
        a_edge_indices_to_pixel_codes = []
        for k in range(A_high_res.shape[1]):
            indices, _, val = find(A_high_res[:, k])
            a_edge_indices_to_pixel_codes.append((np.uint32(indices), val))
        return a_edge_indices_to_pixel_codes

    def compute_core_zone_pixels(self, min_angle, low_res):
        fac = np.clip(np.sin(0.5 * (np.pi - min_angle)), 0.0, 1.0)
        if fac < 1:
            mask = create_circular_mask(low_res, 0.5 * fac * low_res)
            pixel_mask = np.ones((low_res, low_res), dtype=bool)
            pixel_mask[~mask] = False
            self.reachablePixelsMaskNativeRes &= pixel_mask.flatten()

    def init_state_vectors(self):
        self.diff_to_blank_squared_errors = self.importance_map.multiply(self.b_native_res).power(2)
        self.diff_to_blank_squared_error_sum = np.sum(self.diff_to_blank_squared_errors)

        self.rmseValue = np.sqrt(self.diff_to_blank_squared_error_sum / self.b_native_res.size)

        W = self.importance_map[self.low_res_edge_pixel_indices].A.squeeze()
        b = self.b_native_res[self.low_res_edge_pixel_indices].A.squeeze()
        c = self.diff_to_blank_squared_errors[self.low_res_edge_pixel_indices].A.squeeze()
        A = self.low_res_edge_pixel_values
        i = self.low_res_corresp_edge_indices

        sum_of_squared_errors_per_edge_adding = np.bincount(i, weights=W*(b - A)**2)
        sum_of_squared_errors_per_edge_removing = np.bincount(i, weights=W*(b + A)**2)
        diff_to_blank_sum_per_edge = np.bincount(i, weights=c)

        self.f_adding = self.diff_to_blank_squared_error_sum - diff_to_blank_sum_per_edge + sum_of_squared_errors_per_edge_adding
        self.f_removing = self.diff_to_blank_squared_error_sum - diff_to_blank_sum_per_edge + sum_of_squared_errors_per_edge_removing

    def init_lately_visited_pins(self):
        self.latelyVisitedPins = np.zeros((1, 0), dtype=int)

    def find_best_string(self) -> tuple[np.ndarray, int]:
        if self.removalMode:
            j = np.argmin(self.f_removing[self.removable_edge_indices])
            i_next_edge = self.removable_edge_indices[j]
            loss_value = self.f_removing[i_next_edge]
        else:
            i_next_edge = np.argmin(self.f_adding)
            loss_value = self.f_adding[i_next_edge]

        print(f'\tF1 when picking edge Nr. {i_next_edge}: {loss_value:16.16f}')
        return loss_value, i_next_edge

    def choose_string_and_update(self, i):
        # Find all relevant indices
        # Performance improvement: Cache the find operation
        edge_pixel_indices, _, _ = find(self.A_high_res[:, i])  # row_indices of pixels that are hit by string i
        native_res_indices = np.unique(indices_1D_high_res_to_low_res(edge_pixel_indices, self.high_res, self.low_res))
        high_res_indices = indices_1D_low_res_to_high_res(native_res_indices, self.low_res, self.high_res).flatten()

        # pre_update_high_res_recon = self.currentRecon[high_res_indices]
        pre_update_high_res_recon_unclamped = self.current_recon_unclamped[high_res_indices]
        pre_update_low_res_recon = self.current_recon_native_res[native_res_indices]

        dif = 1

        if self.removalMode:
            mask = self.removable_edge_indices == i
            if not any(mask):
                raise ValueError(f"Edge {i} can not be removed.\n")
            else:
                dif = -1
                mask = self.stringList[:, 0] == i
                ind = np.where(mask)[0]
                mask[:] = True
                mask[ind[0]] = False
                self.stringList = self.stringList[mask, :]
                self.picked_edges_sequence = self.picked_edges_sequence[mask, :]
        else:
            self.stringList = np.vstack((self.stringList, [i, 0, 0]))
            self.picked_edges_sequence = np.hstack((self.picked_edges_sequence.T, [i])).T

        # Update data structures
        self.x[i] += dif

        print(f'\tF2 when picking edge Nr. {i}: {np.sum(self.residual)**2:16.16f}\n\n')
        # self.show_current()

        pre_update_errors = self.diff_to_blank_squared_errors[native_res_indices]
        self.diff_to_blank_squared_error_sum -= np.sum(pre_update_errors)
        self.diff_to_blank_squared_errors[native_res_indices] = self.residual[native_res_indices]**2
        post_update_errors = self.diff_to_blank_squared_errors[native_res_indices]
        self.diff_to_blank_squared_error_sum += np.sum(post_update_errors)
        self.rmseValue = np.sqrt(self.diff_to_blank_squared_error_sum / self.b_native_res.size)

        self.update_edge_errors(native_res_indices, high_res_indices, pre_update_low_res_recon,
                                pre_update_high_res_recon_unclamped, pre_update_errors, post_update_errors)
        self.update_incidence_vector(i)

    def update_edge_errors(self, low_res_indices, high_res_indices, pre_update_low_res_recon,
                           pre_update_high_res_recon_unclamped, pre_update_errors, post_update_errors):
        # Update non-intersecting pixel positions
        pre = np.sum(pre_update_errors)
        post = np.sum(post_update_errors)

        # First, falsely update all edges and fix intersection errors afterwards
        self.f_adding -= pre - post
        self.f_removing -= pre - post

        sec_mask = np.max(self.lowResIndexToIndexMap[:, low_res_indices], axis=1).A.squeeze()
        trunc_high_res_indices_mask = high_res_indices <= self.highResIndexToIndexMap.shape[1]
        trunc_high_res_indices = high_res_indices[trunc_high_res_indices_mask]
        high_res_sec_mask = np.max(self.highResIndexToIndexMap[:, trunc_high_res_indices], axis=1)

        sec_edge_pix_ind = self.low_res_edge_pixel_indices[sec_mask]

        self.lowResReIndexMap[low_res_indices] = np.arange(low_res_indices.shape[0])
        re_indices = self.lowResReIndexMap[sec_edge_pix_ind]

        pre_at_indices = pre_update_errors[re_indices]
        post_at_indices = post_update_errors[re_indices]

        sec_corr_edge_ind = self.low_res_corresp_edge_indices[sec_mask]

        pre_corr = np.bincount(sec_corr_edge_ind, weights=pre_at_indices, minlength=self.n_edges)
        post_corr = np.bincount(sec_corr_edge_ind, weights=post_at_indices, minlength=self.n_edges)

        # Fix intersection errors
        self.f_adding -= post_corr - pre_corr
        self.f_removing -= post_corr - pre_corr

        # Update intersecting pixel positions
        high_res_sec_edge_pix_ind = self.highResEdgePixelIndices[high_res_sec_mask]
        high_res_sec_edge_pix_val = self.highResEdgePixelValues[high_res_sec_mask]
        high_res_corr_edge_ind = self.highResCorrespEdgeIndices[high_res_sec_mask]
        high_res_to_low_res_ind = indices_1D_high_res_to_low_res(high_res_sec_edge_pix_ind, self.high_res, self.low_res)
        m = int(np.max(high_res_to_low_res_ind))

        output_ids = indices_2D_to_1D(high_res_to_low_res_ind, high_res_corr_edge_ind, m)
        unique_output_ids = np.unique(output_ids)

        filtered_corr_edge_ind, filtered_pix_ind = indices_1D_to_2D(unique_output_ids, m, mode='row-col')
        filtered_re_indices = self.lowResReIndexMap[filtered_pix_ind]

        num_ids = len(unique_output_ids)
        output_re_index = np.arange(1, num_ids + 1)
        lia, locb = np.isin(output_ids, unique_output_ids, assume_unique=True)
        output_re_index = output_re_index[locb]

        self.highResReIndexMap[high_res_indices] = np.arange(1, len(high_res_indices) + 1)
        high_res_re_indices = self.highResReIndexMap[high_res_sec_edge_pix_ind]

        # PRE UPDATE
        pre_update_high_res_recon = np.minimum(1.0, pre_update_high_res_recon_unclamped)

        pre_update_high_res_val = pre_update_high_res_recon[high_res_re_indices]

        pre_update_high_res_val_unclamped = pre_update_high_res_recon_unclamped[high_res_re_indices]

        pre_update_high_res_val_minus_edges = np.minimum(1.0, pre_update_high_res_val_unclamped - high_res_sec_edge_pix_val)
        pre_update_high_res_val_plus_edges = np.minimum(1.0, pre_update_high_res_val_unclamped + high_res_sec_edge_pix_val)

        # Filter pre update highRes edges
        filtered_pre_update_high_res_val = self.filter_weight * np.bincount(output_re_index, weights=pre_update_high_res_val)
        filtered_pre_update_high_res_val_minus_edges = self.filter_weight * \
            np.bincount(output_re_index, weights=pre_update_high_res_val_minus_edges)
        filtered_pre_update_high_res_val_plus_edges = self.filter_weight * \
            np.bincount(output_re_index, weights=pre_update_high_res_val_plus_edges)

        failure_pre_update_per_edge_adding = np.bincount(filtered_corr_edge_ind,
                                                         weights=(self.importance_map[filtered_pix_ind] *
                                                                  (self.b_native_res[filtered_pix_ind] -
                                                                  (pre_update_low_res_recon[filtered_re_indices] -
                                                                   filtered_pre_update_high_res_val +
                                                                   filtered_pre_update_high_res_val_plus_edges))**2),
                                                         minlength=self.n_edges)

        failure_pre_update_per_edge_removing = np.bincount(filtered_corr_edge_ind,
                                                           weights=(self.importance_map[filtered_pix_ind] *
                                                                    (self.b_native_res[filtered_pix_ind] -
                                                                     (pre_update_low_res_recon[filtered_re_indices] -
                                                                      filtered_pre_update_high_res_val +
                                                                      filtered_pre_update_high_res_val_minus_edges))**2),
                                                           minlength=self.n_edges)

        # POST UPDATE
        post_update_high_res_recon = self.current_recon[high_res_indices]
        post_update_high_res_val = post_update_high_res_recon[high_res_re_indices]

        post_update_high_res_recon_unclamped = self.current_recon_unclamped[high_res_indices]
        post_update_high_res_val_unclamped = post_update_high_res_recon_unclamped[high_res_re_indices]

        post_update_high_res_val_minus_edges = np.minimum(1.0, post_update_high_res_val_unclamped - high_res_sec_edge_pix_val)
        post_update_high_res_val_plus_edges = np.minimum(1.0, post_update_high_res_val_unclamped + high_res_sec_edge_pix_val)

        post_update_low_res_recon = self.current_recon_native_res[low_res_indices]

        # Filter post update highRes edges
        filtered_post_update_high_res_val = self.filter_weight * np.bincount(output_re_index, weights=post_update_high_res_val)
        filtered_post_update_high_res_val_minus_edges = self.filter_weight * \
            np.bincount(output_re_index, weights=post_update_high_res_val_minus_edges)
        filtered_post_update_high_res_val_plus_edges = self.filter_weight * \
            np.bincount(output_re_index, weights=post_update_high_res_val_plus_edges)

        failure_post_update_per_edge_adding = np.bincount(filtered_corr_edge_ind,
                                                          weights=(self.importance_map[filtered_pix_ind] *
                                                                   (self.b_native_res[filtered_pix_ind] -
                                                                    (post_update_low_res_recon[filtered_re_indices] -
                                                                     filtered_post_update_high_res_val +
                                                                     filtered_post_update_high_res_val_plus_edges))**2),
                                                          minlength=self.n_edges)

        failure_post_update_per_edge_removing = np.bincount(filtered_corr_edge_ind,
                                                            weights=(self.importance_map[filtered_pix_ind] *
                                                                     (self.b_native_res[filtered_pix_ind] -
                                                                     (post_update_low_res_recon[filtered_re_indices] -
                                                                      filtered_post_update_high_res_val +
                                                                      filtered_post_update_high_res_val_minus_edges))**2),
                                                            minlength=self.n_edges)

        self.f_adding -= failure_pre_update_per_edge_adding - failure_post_update_per_edge_adding
        self.f_removing -= failure_pre_update_per_edge_removing - failure_post_update_per_edge_removing

    def update_incidence_vector(self, edge_index: int):
        dif = -1 if self.removalMode else 1

        for p in self.pin_edge_transformer.edges_to_pins(edge_index):
            self.pin_count[p] += dif

        p1, p2 = self.pin_edge_transformer.edges_to_pins(edge_index)

        self.pin_count[p1] += dif
        self.pin_count[p2] += dif

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
        return self.rmseValue

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
