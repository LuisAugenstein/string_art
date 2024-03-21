import numpy as np
from scipy.sparse import csc_matrix
from typing import Literal
from string_art.optimization.losses.multi_sample_correspondence_map import multi_sample_correspondence_map
from string_art.transformations import indices_1D_high_res_to_low_res, indices_1D_low_res_to_high_res, indices_1D_to_2D, indices_2D_to_1D
from string_art.api import get_np_array_module
import torch


class OptimizedLoss:
    def __init__(self, img: torch.Tensor, importance_map: torch.Tensor, A_high_res: torch.Tensor, A_low_res: torch.Tensor) -> None:
        """
        Parameters
        -
        img:            torch.shape([low_res, low_res]) grayscale image with values between 0 and 1
        importance_map: torch.shape([low_res, low_res]) scalars between 0 and 1 to weight the reconstruction importance of different image regions
        A_high_res:     torch.shape([high_res**2, n_strings])
        A_low_res:      torch.shape([low_res**2, n_strings])
        """
        img = img.numpy()
        importance_map = importance_map.numpy()
        A_high_res = csc_matrix((A_high_res.values(), (A_high_res.indices()[0], A_high_res.indices()[1])), shape=A_high_res.shape)
        A_low_res = csc_matrix((A_low_res.values(), (A_low_res.indices()[0], A_low_res.indices()[1])), shape=A_low_res.shape)

        # define helper variables
        self.xp, self.xipy = get_np_array_module(img)
        xp = self.xp
        n_pixels_high_res, n_pixels_low_res = A_high_res.shape[0], A_low_res.shape[0]
        self.low_res, self.high_res = int(xp.sqrt(n_pixels_low_res)), int(xp.sqrt(n_pixels_high_res))
        self.n_strings = A_high_res.shape[1]

        # move data to GPU
        self.b_native_res = img.flatten()
        self.importance_map = importance_map.flatten()
        self.A_high_res = A_high_res
        self.low_res_col_row_values = self.xipy.sparse.find(A_low_res.T)
        self.high_res_col_row_values = self.xipy.sparse.find(A_high_res.T)
        self.high_res_index_to_index_map = self.__get_index_to_index_map(A_high_res)
        self.low_res_index_to_index_map = self.__get_index_to_index_map(A_low_res)

        self.correspondence_map = self.xipy.sparse.csr_matrix(multi_sample_correspondence_map(self.low_res, self.high_res))

        self.current_recon = xp.zeros(n_pixels_high_res)
        self.current_recon_unclamped = xp.zeros(n_pixels_high_res)
        self.current_recon_native_res = xp.zeros(n_pixels_low_res)

        self.high_res_re_index_map = xp.zeros(n_pixels_high_res, dtype=int)
        self.low_res_re_index_map = xp.zeros(n_pixels_low_res, dtype=int)

        self.diff_to_blank_squared_errors = (self.importance_map * self.b_native_res)**2
        self.diff_to_blank_squared_error_sum = xp.sum(self.diff_to_blank_squared_errors)
        self.rmse_value = xp.sqrt(self.diff_to_blank_squared_error_sum / self.b_native_res.size)
        self.f_adding, self.f_removing = self.__init_f_scores(self.importance_map, self.b_native_res, self.low_res_col_row_values,
                                                              self.diff_to_blank_squared_error_sum, self.n_strings)

    def get_f_scores(self,  mode: Literal['add', 'remove'] = 'add') -> tuple[torch.Tensor, torch.Tensor]:
        return torch.Tensor(self.f_adding) if mode == 'add' else torch.Tensor(self.f_removing)

    def update(self, i_next_string: int, mode: Literal['add', 'remove']) -> np.ndarray:
        self.__choose_string_and_update(i_next_string, 1 if mode == 'add' else -1)

    def __init_f_scores(self, importance_map: np.ndarray, b_native_res: np.ndarray, low_res_row_col_values: csc_matrix, diff_to_blank_squared_error_sum: float, n_strings: int) -> tuple[np.ndarray, np.ndarray]:
        low_res_corresp_edge_indices, low_res_edge_pixel_indices, low_res_edge_pixel_values = low_res_row_col_values
        w = importance_map[low_res_edge_pixel_indices]
        b = b_native_res[low_res_edge_pixel_indices]
        a = low_res_edge_pixel_values
        j = low_res_corresp_edge_indices
        xp = self.xp
        sum_of_squared_errors_per_edge_adding = xp.bincount(j, weights=(w*(b - a))**2, minlength=n_strings)
        sum_of_squared_errors_per_edge_removing = xp.bincount(j, weights=(w*(b + a))**2, minlength=n_strings)
        diff_to_blank_sum_per_edge = xp.bincount(j, weights=(w*b)**2)

        f_adding = sum_of_squared_errors_per_edge_adding - diff_to_blank_sum_per_edge + diff_to_blank_squared_error_sum
        f_removing = diff_to_blank_squared_error_sum - diff_to_blank_sum_per_edge + sum_of_squared_errors_per_edge_removing
        return f_adding, f_removing

    def __choose_string_and_update(self, edge_index: int, dif: int) -> None:
        xp = self.xp
        edge_pixel_indices = self.A_high_res[:, edge_index].indices
        edge_values = self.A_high_res[:, edge_index].data
        low_res_indices = xp.unique(indices_1D_high_res_to_low_res(edge_pixel_indices, self.high_res, self.low_res))
        high_res_indices = indices_1D_low_res_to_high_res(low_res_indices, self.low_res, self.high_res).T.flatten()

        pre_update_high_res_recon_unclamped = self.current_recon_unclamped[high_res_indices]
        pre_update_low_res_recon = self.current_recon_native_res[low_res_indices]

        self.current_recon_unclamped[edge_pixel_indices] += dif * edge_values
        self.current_recon[edge_pixel_indices] = xp.clip(self.current_recon_unclamped[edge_pixel_indices], 0, 1)
        self.current_recon_native_res[low_res_indices] = self.correspondence_map[low_res_indices, :] @ self.current_recon

        pre_update_errors = self.diff_to_blank_squared_errors[low_res_indices]
        self.diff_to_blank_squared_error_sum -= xp.sum(pre_update_errors)
        residual = self.importance_map[low_res_indices] * (self.b_native_res[low_res_indices] - self.current_recon_native_res[low_res_indices])
        self.diff_to_blank_squared_errors[low_res_indices] = residual**2
        post_update_errors = self.diff_to_blank_squared_errors[low_res_indices]
        self.diff_to_blank_squared_error_sum += xp.sum(post_update_errors)
        self.rmse_value = xp.sqrt(self.diff_to_blank_squared_error_sum / self.b_native_res.size)

        self.__update_edge_errors(low_res_indices, high_res_indices, pre_update_low_res_recon,
                                  pre_update_high_res_recon_unclamped, pre_update_errors, post_update_errors)

    def __update_edge_errors(self, low_res_indices, high_res_indices, pre_update_low_res_recon,
                             pre_update_high_res_recon_unclamped, pre_update_errors, post_update_errors):
        xp = self.xp
        # Update non-intersecting pixel positions
        pre = xp.sum(pre_update_errors)
        post = xp.sum(post_update_errors)

        # First, falsely update all edges and fix intersection errors afterwards
        self.f_adding -= pre - post
        self.f_removing -= pre - post
        high_res_sec_mask = self.high_res_index_to_index_map[:, high_res_indices].max(axis=1).A.squeeze().astype(bool)
        # high_res_sec_mask = np.max(self.high_res_index_to_index_map[:, high_res_indices], axis=1).A.squeeze().astype(bool)
        """which values of A_high_res color a pixel of the current string. 
        Of course all the values of the column for the current string do, but others might do as well."""

        sec_mask = self.low_res_index_to_index_map[:, low_res_indices].max(axis=1).A.squeeze().astype(bool)
        low_res_corresp_edge_indices, low_res_edge_pixel_indices, _ = self.low_res_col_row_values
        sec_corr_edge_ind, sec_edge_pix_ind = low_res_corresp_edge_indices[sec_mask], low_res_edge_pixel_indices[sec_mask]

        self.low_res_re_index_map[low_res_indices] = xp.arange(low_res_indices.shape[0])
        re_indices = self.low_res_re_index_map[sec_edge_pix_ind]

        pre_at_indices = pre_update_errors[re_indices]
        post_at_indices = post_update_errors[re_indices]

        pre_corr = xp.bincount(sec_corr_edge_ind, weights=pre_at_indices, minlength=self.n_strings)
        post_corr = xp.bincount(sec_corr_edge_ind, weights=post_at_indices, minlength=self.n_strings)

        # Fix intersection errors
        self.f_adding -= post_corr - pre_corr
        self.f_removing -= post_corr - pre_corr

        # Update intersecting pixel positions
        high_res_corresp_edge_indices, high_res_edge_pixel_indices, high_res_edge_pixel_values = self.high_res_col_row_values
        high_res_sec_edge_pix_ind = high_res_edge_pixel_indices[high_res_sec_mask]
        high_res_sec_edge_pix_val = high_res_edge_pixel_values[high_res_sec_mask]
        high_res_corr_edge_ind = high_res_corresp_edge_indices[high_res_sec_mask]
        high_res_to_low_res_ind = indices_1D_high_res_to_low_res(high_res_sec_edge_pix_ind, self.high_res, self.low_res)
        m = int(xp.max(high_res_to_low_res_ind)+1)

        output_ids = indices_2D_to_1D(high_res_to_low_res_ind, high_res_corr_edge_ind, m)
        unique_output_ids = xp.unique(output_ids)
        filtered_corr_edge_ind, filtered_pix_ind = indices_1D_to_2D(unique_output_ids, m, mode='row-col').T
        filtered_re_indices = self.low_res_re_index_map[filtered_pix_ind]

        _, output_re_index = xp.unique(output_ids, return_inverse=True)

        self.high_res_re_index_map[high_res_indices] = xp.arange(len(high_res_indices))
        high_res_re_indices = self.high_res_re_index_map[high_res_sec_edge_pix_ind]

        # PRE UPDATE input: pre_update_high_res_recon_unclamped, high_res_re_indices, high_res_sec_edge_pix_val
        pre_update_high_res_recon = xp.minimum(1.0, pre_update_high_res_recon_unclamped)
        pre_update_high_res_val_unclamped = pre_update_high_res_recon_unclamped[high_res_re_indices]

        pre_update_high_res_val = pre_update_high_res_recon[high_res_re_indices]
        pre_update_high_res_val_minus_edges = xp.minimum(1.0, pre_update_high_res_val_unclamped - high_res_sec_edge_pix_val)
        pre_update_high_res_val_plus_edges = xp.minimum(1.0, pre_update_high_res_val_unclamped + high_res_sec_edge_pix_val)

        # Filter pre update highRes edges
        super_sampling_factor = self.high_res // self.low_res
        filter_weight = 1.0 / (super_sampling_factor * super_sampling_factor)
        filtered_pre_update_high_res_val = filter_weight * xp.bincount(output_re_index, weights=pre_update_high_res_val)
        filtered_pre_update_high_res_val_minus_edges = filter_weight * \
            xp.bincount(output_re_index, weights=pre_update_high_res_val_minus_edges)
        filtered_pre_update_high_res_val_plus_edges = filter_weight * \
            xp.bincount(output_re_index, weights=pre_update_high_res_val_plus_edges)

        failure_pre_update_per_edge_adding = xp.bincount(filtered_corr_edge_ind,
                                                         weights=(self.importance_map[filtered_pix_ind] *
                                                                  (self.b_native_res[filtered_pix_ind] -
                                                                  (pre_update_low_res_recon[filtered_re_indices] -
                                                                   filtered_pre_update_high_res_val +
                                                                   filtered_pre_update_high_res_val_plus_edges))**2),
                                                         minlength=self.n_strings)

        failure_pre_update_per_edge_removing = xp.bincount(filtered_corr_edge_ind,
                                                           weights=(self.importance_map[filtered_pix_ind] *
                                                                    (self.b_native_res[filtered_pix_ind] -
                                                                     (pre_update_low_res_recon[filtered_re_indices] -
                                                                      filtered_pre_update_high_res_val +
                                                                      filtered_pre_update_high_res_val_minus_edges))**2),
                                                           minlength=self.n_strings)

        # POST UPDATE
        post_update_high_res_recon = self.current_recon[high_res_indices]
        post_update_high_res_val = post_update_high_res_recon[high_res_re_indices]

        post_update_high_res_recon_unclamped = self.current_recon_unclamped[high_res_indices]
        post_update_high_res_val_unclamped = post_update_high_res_recon_unclamped[high_res_re_indices]

        post_update_high_res_val_minus_edges = xp.minimum(1.0, post_update_high_res_val_unclamped - high_res_sec_edge_pix_val)
        post_update_high_res_val_plus_edges = xp.minimum(1.0, post_update_high_res_val_unclamped + high_res_sec_edge_pix_val)

        post_update_low_res_recon = self.current_recon_native_res[low_res_indices]

        # Filter post update highRes edges
        filtered_post_update_high_res_val = filter_weight * xp.bincount(output_re_index, weights=post_update_high_res_val)
        filtered_post_update_high_res_val_minus_edges = filter_weight * \
            xp.bincount(output_re_index, weights=post_update_high_res_val_minus_edges)
        filtered_post_update_high_res_val_plus_edges = filter_weight * \
            xp.bincount(output_re_index, weights=post_update_high_res_val_plus_edges)

        failure_post_update_per_edge_adding = xp.bincount(filtered_corr_edge_ind,
                                                          weights=(self.importance_map[filtered_pix_ind] *
                                                                   (self.b_native_res[filtered_pix_ind] -
                                                                    (post_update_low_res_recon[filtered_re_indices] -
                                                                     filtered_post_update_high_res_val +
                                                                     filtered_post_update_high_res_val_plus_edges))**2),
                                                          minlength=self.n_strings)

        failure_post_update_per_edge_removing = xp.bincount(filtered_corr_edge_ind,
                                                            weights=(self.importance_map[filtered_pix_ind] *
                                                                     (self.b_native_res[filtered_pix_ind] -
                                                                     (post_update_low_res_recon[filtered_re_indices] -
                                                                      filtered_post_update_high_res_val +
                                                                      filtered_post_update_high_res_val_minus_edges))**2),
                                                            minlength=self.n_strings)

        self.f_adding -= failure_pre_update_per_edge_adding - failure_post_update_per_edge_adding
        self.f_removing -= failure_pre_update_per_edge_removing - failure_post_update_per_edge_removing

    def __get_index_to_index_map(self, A: csc_matrix) -> csc_matrix:
        """
        Parameters
        -
        A: np.shape([n_pixels, n_strings])    values between 0 and 1 indicate how much a pixel i is darkened if edge j is active.

        Returns
        -
        index_to_index_map: np.shape([n_values_in_A, n_pixels]) binary matrix which contains a single 1 in each row (i,edge_pixel_indices[i]) and otherwise 0. 
        """
        xp = self.xp
        n_values_in_A = A.indices.shape[0]
        data = xp.ones(n_values_in_A)
        rows = xp.arange(n_values_in_A)
        index_to_index_map = self.xipy.sparse.csc_matrix((data, (rows, A.indices)), shape=(n_values_in_A, A.shape[0]))
        return index_to_index_map
