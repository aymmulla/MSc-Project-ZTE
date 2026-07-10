import numpy as np
import matplotlib.pyplot as plt
import sigpy as sp
import jax as jx
from . import gridding_hankel


def sig_val_thresholding_jax_soft(data, zero_thresh):
    U, Sigma, Vh = jx.numpy.linalg.svd(data, full_matrices=False)
    thresh = Sigma.max() * zero_thresh
    S_reduced = jx.numpy.maximum(Sigma - thresh, 0)
    return U, S_reduced, Vh

def sig_val_thresholding_jax(data, zero_thresh):
    U, Sigma, Vh = jx.numpy.linalg.svd(data, full_matrices=False)
    thresh = Sigma.max() * zero_thresh
    S_reduced = jx.numpy.where(Sigma > thresh, Sigma, 0)
    return U, S_reduced, Vh

def sig_val_thresholding(data, zero_thresh):
    U, Sigma, Vh = np.linalg.svd(data, full_matrices=False)
    max_val = np.max(Sigma)
    thresh = max_val*zero_thresh
    arr = []
    for i in range(len(Sigma)): ##Sigma[Sigma < thresh] = 0
        if Sigma[i] > thresh:
            arr.append(Sigma[i])
        else: arr.append(0)
    S_reduced = np.array(arr)
    return(U, S_reduced, Vh)

def svd_recon(u, s_reduced, vh):
    return ((u*s_reduced) @ vh)


def LORAKS_loop(n_iters, window_size, zero_thresh, cartesian_inputkspace, dtg_mask):
    ksp_forhankel = cartesian_inputkspace.copy()
    iter_count = 0
    deltas = []
    k_prev = None
    while iter_count < n_iters:
        hankel_matrix, n_coils, Numx, Numy = gridding_hankel.hankel(kspace=ksp_forhankel, w=window_size)
        U, S_reduced, Vh = sig_val_thresholding_jax(data=hankel_matrix, zero_thresh=zero_thresh)
        data_recon = (U * S_reduced) @ Vh

        kspace_cart_coils_recon = gridding_hankel.hankel_H_averaged(data_recon, n_coils=ksp_forhankel.shape[0], Nx=ksp_forhankel.shape[1], Ny=ksp_forhankel.shape[2], w=window_size)
        kspace_cart_coils_consistent = kspace_cart_coils_recon.copy()

        for coil in range(ksp_forhankel.shape[0]):
            kspace_cart_coils_consistent[coil, dtg_mask] = cartesian_inputkspace[coil, dtg_mask]
        ksp_forhankel = kspace_cart_coils_consistent.copy()

        vec = ksp_forhankel[:, ~dtg_mask]

        if k_prev is not None:
            deltas.append(np.linalg.norm(vec - k_prev) / np.linalg.norm(k_prev))
        k_prev = vec.copy()
        iter_count += 1

    output_kspace = ksp_forhankel.copy()
    return(output_kspace, deltas)