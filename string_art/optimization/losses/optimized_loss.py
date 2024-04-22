import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from typing import Literal
from string_art.optimization.losses.high_res_to_low_res_matrix import high_res_to_low_res_matrix
from string_art.transformations import indices_1D_high_res_to_low_res, indices_1D_low_res_to_high_res, indices_1D_to_2D, indices_2D_to_1D
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

        A_high_res = A_high_res.to_sparse_csc()
        A_low_res = A_low_res.to_sparse_csc()
        self.A_high_res = A_high_res
        self.A_low_res = A_low_res

        # define helper variables
        n_pixels_high_res, n_pixels_low_res = A_high_res.shape[0], A_low_res.shape[0]
        self.low_res, self.high_res = int(np.sqrt(n_pixels_low_res)), int(np.sqrt(n_pixels_high_res))
        self.n_strings = A_high_res.shape[1]

        # move data to GPU
        self.b_native_res = img.flatten()
        self.importance_map = importance_map.flatten()

        ccol = A_low_res.ccol_indices()
        self.low_res_col_row_values = (
            torch.cat([j*torch.ones(ccol[j+1]-ccol[j]) for j in range(self.n_strings)]).int().numpy(),
            torch.cat([A_low_res.row_indices()[ccol[j]:ccol[j+1]] for j in range(self.n_strings)]).int().numpy(),
            torch.cat([A_low_res.values()[ccol[j]:ccol[j+1]] for j in range(self.n_strings)]).numpy()
        )
        ccol = A_high_res.ccol_indices()
        self.high_res_col_row_values = (
            torch.cat([j*torch.ones(ccol[j+1]-ccol[j]) for j in range(self.n_strings)]).int().numpy(),
            torch.cat([A_high_res.row_indices()[ccol[j]:ccol[j+1]] for j in range(self.n_strings)]).int().numpy(),
            torch.cat([A_high_res.values()[ccol[j]:ccol[j+1]] for j in range(self.n_strings)]).numpy()
        )

        low_res_index_to_index_map2 = self.__get_index_to_index_map(A_low_res)
        high_res_index_to_index_map2 = self.__get_index_to_index_map(A_high_res)
        self.high_res_index_to_index_map = csc_matrix(
            (high_res_index_to_index_map2.values(),
             (high_res_index_to_index_map2.indices()[0],
             high_res_index_to_index_map2.indices()[1])), shape=high_res_index_to_index_map2.shape)
        self.low_res_index_to_index_map = csc_matrix((
            low_res_index_to_index_map2.values(),
            (low_res_index_to_index_map2.indices()[0],
             low_res_index_to_index_map2.indices()[1])), shape=low_res_index_to_index_map2.shape)

        h2l = high_res_to_low_res_matrix(self.low_res, self.high_res)
        self.h2l = csr_matrix((h2l.values(), (h2l.indices()[0], h2l.indices()[1])), shape=(n_pixels_low_res, n_pixels_high_res))

        self.current_recon = np.zeros(n_pixels_high_res)
        self.current_recon_unclamped = np.zeros(n_pixels_high_res)
        self.current_recon_native_res = np.zeros(n_pixels_low_res)

        self.high_res_re_index_map = np.zeros(n_pixels_high_res, dtype=int)
        self.low_res_re_index_map = np.zeros(n_pixels_low_res, dtype=int)

        self.diff_to_blank_squared_errors = (self.importance_map * self.b_native_res)**2
        self.diff_to_blank_squared_error_sum = np.sum(self.diff_to_blank_squared_errors)
        self.rmse_value = np.sqrt(self.diff_to_blank_squared_error_sum / self.b_native_res.size)
        self.f_adding, self.f_removing = self.__init_f_scores(self.importance_map, self.b_native_res, self.low_res_col_row_values,
                                                              self.diff_to_blank_squared_error_sum, self.n_strings)

    def get_f_scores(self,  mode: Literal['add', 'remove'] = 'add') -> tuple[torch.Tensor, torch.Tensor]:
        return torch.Tensor(self.f_adding) if mode == 'add' else torch.Tensor(self.f_removing)

    def update(self, i_next_string: torch.Tensor, mode: Literal['add', 'remove']) -> torch.Tensor:
        """
        Updates the f_scores after choosing the next string.

        Parameters
        -
        i_next_string: torch.shape([], int)  index of the next string to choose
        """
        self.__choose_string_and_update(i_next_string.item(), 1 if mode == 'add' else -1)

    def __init_f_scores(self, importance_map: np.ndarray, b_native_res: np.ndarray, low_res_row_col_values: csc_matrix, diff_to_blank_squared_error_sum: float, n_strings: int) -> tuple[np.ndarray, np.ndarray]:
        low_res_corresp_edge_indices, low_res_edge_pixel_indices, low_res_edge_pixel_values = low_res_row_col_values
        w = importance_map[low_res_edge_pixel_indices]
        b = b_native_res[low_res_edge_pixel_indices]
        a = low_res_edge_pixel_values
        j = low_res_corresp_edge_indices
        sum_of_squared_errors_per_edge_adding = np.bincount(j, weights=(w*(b - a))**2, minlength=n_strings)
        sum_of_squared_errors_per_edge_removing = np.bincount(j, weights=(w*(b + a))**2, minlength=n_strings)
        diff_to_blank_sum_per_edge = np.bincount(j, weights=(w*b)**2)

        f_adding = sum_of_squared_errors_per_edge_adding - diff_to_blank_sum_per_edge + diff_to_blank_squared_error_sum
        f_removing = diff_to_blank_squared_error_sum - diff_to_blank_sum_per_edge + sum_of_squared_errors_per_edge_removing
        return f_adding, f_removing

    def __choose_string_and_update(self, edge_index: int, mode: int) -> None:
        ccol = self.A_high_res.ccol_indices()
        edge_pixel_indices = self.A_high_res.row_indices()[ccol[edge_index]:ccol[edge_index+1]].int().numpy()
        edge_values = self.A_high_res.values()[ccol[edge_index]:ccol[edge_index+1]].numpy()

        low_res_indices = np.unique(indices_1D_high_res_to_low_res(edge_pixel_indices, self.high_res, self.low_res))
        high_res_indices = indices_1D_low_res_to_high_res(low_res_indices, self.low_res, self.high_res).T.flatten()

        pre_update_high_res_recon_unclamped = self.current_recon_unclamped[high_res_indices]
        pre_update_low_res_recon = self.current_recon_native_res[low_res_indices]

        self.current_recon_unclamped[edge_pixel_indices] += mode * edge_values
        self.current_recon[edge_pixel_indices] = np.clip(self.current_recon_unclamped[edge_pixel_indices], 0, 1)
        self.current_recon_native_res[low_res_indices] = self.h2l[low_res_indices, :] @ self.current_recon

        pre_update_errors = self.diff_to_blank_squared_errors[low_res_indices]
        self.diff_to_blank_squared_error_sum -= np.sum(pre_update_errors)
        residual = self.importance_map[low_res_indices] * (self.b_native_res[low_res_indices] - self.current_recon_native_res[low_res_indices])
        self.diff_to_blank_squared_errors[low_res_indices] = residual**2
        post_update_errors = self.diff_to_blank_squared_errors[low_res_indices]
        self.diff_to_blank_squared_error_sum += np.sum(post_update_errors)
        self.rmse_value = np.sqrt(self.diff_to_blank_squared_error_sum / self.b_native_res.size)

        self.__update_edge_errors(low_res_indices, high_res_indices, pre_update_low_res_recon,
                                  pre_update_high_res_recon_unclamped, pre_update_errors, post_update_errors)

    def __update_edge_errors(self, low_res_indices, high_res_indices, pre_update_low_res_recon,
                             pre_update_high_res_recon_unclamped, pre_update_errors, post_update_errors):
        # Update non-intersecting pixel positions
        pre = np.sum(pre_update_errors)
        post = np.sum(post_update_errors)

        # First, falsely update all edges and fix intersection errors afterwards
        self.f_adding -= pre - post
        self.f_removing -= pre - post
        high_res_sec_mask = self.high_res_index_to_index_map[:, high_res_indices].max(axis=1).A.squeeze().astype(bool)
        # high_res_sec_mask = np.max(self.high_res_index_to_index_map[:, high_res_indices], axis=1).A.squeeze().astype(bool)
        """which values of A_high_res color a pixel of the current string. 
        Of course all the values of the column for the current string do, but others might do as well."""

        sec_mask = self.low_res_index_to_index_map[:, low_res_indices].max(axis=1).A.squeeze().astype(bool)
        low_res_string_indices, low_res_pixel_indices, _ = self.low_res_col_row_values
        sec_corr_edge_ind, sec_edge_pix_ind = low_res_string_indices[sec_mask], low_res_pixel_indices[sec_mask]

        ccol = self.A_low_res.ccol_indices()

        self.low_res_re_index_map[low_res_indices] = np.arange(low_res_indices.shape[0])
        re_indices = self.low_res_re_index_map[sec_edge_pix_ind]

        pre_at_indices = pre_update_errors[re_indices]
        post_at_indices = post_update_errors[re_indices]

        pre_corr = np.bincount(sec_corr_edge_ind, weights=pre_at_indices, minlength=self.n_strings)
        post_corr = np.bincount(sec_corr_edge_ind, weights=post_at_indices, minlength=self.n_strings)

        # Fix intersection errors
        self.f_adding -= post_corr - pre_corr
        self.f_removing -= post_corr - pre_corr

        # Update intersecting pixel positions
        high_res_corresp_edge_indices, high_res_edge_pixel_indices, high_res_edge_pixel_values = self.high_res_col_row_values
        high_res_sec_edge_pix_ind = high_res_edge_pixel_indices[high_res_sec_mask]
        high_res_sec_edge_pix_val = high_res_edge_pixel_values[high_res_sec_mask]
        high_res_corr_edge_ind = high_res_corresp_edge_indices[high_res_sec_mask]
        high_res_to_low_res_ind = indices_1D_high_res_to_low_res(high_res_sec_edge_pix_ind, self.high_res, self.low_res)
        m = int(np.max(high_res_to_low_res_ind)+1)

        output_ids = indices_2D_to_1D(high_res_to_low_res_ind, high_res_corr_edge_ind, m)
        unique_output_ids = np.unique(output_ids)
        filtered_corr_edge_ind, filtered_pix_ind = indices_1D_to_2D(unique_output_ids, m, mode='row-col').T
        filtered_re_indices = self.low_res_re_index_map[filtered_pix_ind]

        _, output_re_index = np.unique(output_ids, return_inverse=True)

        self.high_res_re_index_map[high_res_indices] = np.arange(len(high_res_indices))
        high_res_re_indices = self.high_res_re_index_map[high_res_sec_edge_pix_ind]

        # PRE UPDATE input: pre_update_high_res_recon_unclamped, high_res_re_indices, high_res_sec_edge_pix_val
        pre_update_high_res_recon = np.minimum(1.0, pre_update_high_res_recon_unclamped)
        pre_update_high_res_val_unclamped = pre_update_high_res_recon_unclamped[high_res_re_indices]

        pre_update_high_res_val = pre_update_high_res_recon[high_res_re_indices]
        pre_update_high_res_val_minus_edges = np.minimum(1.0, pre_update_high_res_val_unclamped - high_res_sec_edge_pix_val)
        pre_update_high_res_val_plus_edges = np.minimum(1.0, pre_update_high_res_val_unclamped + high_res_sec_edge_pix_val)

        # Filter pre update highRes edges
        super_sampling_factor = self.high_res // self.low_res
        filter_weight = 1.0 / (super_sampling_factor * super_sampling_factor)
        filtered_pre_update_high_res_val = filter_weight * np.bincount(output_re_index, weights=pre_update_high_res_val)
        filtered_pre_update_high_res_val_minus_edges = filter_weight * \
            np.bincount(output_re_index, weights=pre_update_high_res_val_minus_edges)
        filtered_pre_update_high_res_val_plus_edges = filter_weight * \
            np.bincount(output_re_index, weights=pre_update_high_res_val_plus_edges)

        failure_pre_update_per_edge_adding = np.bincount(filtered_corr_edge_ind,
                                                         weights=(self.importance_map[filtered_pix_ind] *
                                                                  (self.b_native_res[filtered_pix_ind] -
                                                                  (pre_update_low_res_recon[filtered_re_indices] -
                                                                   filtered_pre_update_high_res_val +
                                                                   filtered_pre_update_high_res_val_plus_edges))**2),
                                                         minlength=self.n_strings)

        failure_pre_update_per_edge_removing = np.bincount(filtered_corr_edge_ind,
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

        post_update_high_res_val_minus_edges = np.minimum(1.0, post_update_high_res_val_unclamped - high_res_sec_edge_pix_val)
        post_update_high_res_val_plus_edges = np.minimum(1.0, post_update_high_res_val_unclamped + high_res_sec_edge_pix_val)

        post_update_low_res_recon = self.current_recon_native_res[low_res_indices]

        # Filter post update highRes edges
        filtered_post_update_high_res_val = filter_weight * np.bincount(output_re_index, weights=post_update_high_res_val)
        filtered_post_update_high_res_val_minus_edges = filter_weight * \
            np.bincount(output_re_index, weights=post_update_high_res_val_minus_edges)
        filtered_post_update_high_res_val_plus_edges = filter_weight * \
            np.bincount(output_re_index, weights=post_update_high_res_val_plus_edges)

        failure_post_update_per_edge_adding = np.bincount(filtered_corr_edge_ind,
                                                          weights=(self.importance_map[filtered_pix_ind] *
                                                                   (self.b_native_res[filtered_pix_ind] -
                                                                    (post_update_low_res_recon[filtered_re_indices] -
                                                                     filtered_post_update_high_res_val +
                                                                     filtered_post_update_high_res_val_plus_edges))**2),
                                                          minlength=self.n_strings)

        failure_post_update_per_edge_removing = np.bincount(filtered_corr_edge_ind,
                                                            weights=(self.importance_map[filtered_pix_ind] *
                                                                     (self.b_native_res[filtered_pix_ind] -
                                                                     (post_update_low_res_recon[filtered_re_indices] -
                                                                      filtered_post_update_high_res_val +
                                                                      filtered_post_update_high_res_val_minus_edges))**2),
                                                            minlength=self.n_strings)

        self.f_adding -= failure_pre_update_per_edge_adding - failure_post_update_per_edge_adding
        self.f_removing -= failure_pre_update_per_edge_removing - failure_post_update_per_edge_removing

    def __get_index_to_index_map(self, A: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        -
        A: torch.sparse_csc([n_pixels, n_strings]) values between 0 and 1 indicate how much a pixel i is darkened if string j is active.

        Returns
        -
        index_to_index_map: torch.sparse([n_values_in_A, n_pixels]) binary matrix which contains a single 1 in each row (i,pixel_indices[i]) and otherwise 0. 
        """
        n_values_in_A = A.row_indices().shape[0]
        data = torch.ones(n_values_in_A)
        rows = torch.arange(n_values_in_A)
        ccol = A.ccol_indices()
        A_rows = torch.cat([A.row_indices()[ccol[j]:ccol[j+1]] for j in range(A.shape[1])])
        indices = torch.stack([rows, A_rows])
        index_to_index_map = torch.sparse_coo_tensor(indices, data, size=(n_values_in_A, A.shape[0]))
        return index_to_index_map.coalesce()
