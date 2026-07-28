import numpy as np
import matplotlib.pyplot as plt
import sigpy as sp
import jax as jx
import time
from . import gridding_hankel
from . import zeropadding


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

def sig_val_thresholding_jax(data, zero_thresh):
    U, Sigma, Vh = jx.numpy.linalg.svd(data, full_matrices=False)
    thresh = Sigma.max() * zero_thresh
    S_reduced = np.where(Sigma > thresh, Sigma, 0)
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


def LORAKS_loop(n_iters, window_size, zero_thresh, cartesian_inputkspace, dtg_mask, stride, im_dim, enlarged_kspace, inner_mask, inner_start, inner_end):
    ksp_forhankel = cartesian_inputkspace.copy()
    iter_count = 0
    deltas = []
    k_prev = None
    while iter_count < n_iters:
        hankel_matrix, n_coils, Numx, Numy = gridding_hankel.hankel_2(kspace=ksp_forhankel, w=window_size, s = stride)
        U, S_reduced, Vh = sig_val_thresholding_jax(data=hankel_matrix, zero_thresh=zero_thresh)
        data_recon = (U * S_reduced) @ Vh

        kspace_cart_coils_recon = gridding_hankel.hankel_H_averaged_2(data_recon, n_coils=ksp_forhankel.shape[0], Nx=ksp_forhankel.shape[1], Ny=ksp_forhankel.shape[2], w=window_size, s=stride)
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
    filled_ksp = zeropadding.rebuild(output_kspace=output_kspace, inner_mask = inner_mask, inner_start = inner_start, inner_end = inner_end, enlarged_kspace=enlarged_kspace, resize_x = im_dim, resize_y = im_dim)
    im_grid0 = sp.ifft(filled_ksp, axes=(-2, -1))
    im_0 = np.sum(np.abs(im_grid0)**2, axis=0)**0.5

    output_kspace = ksp_forhankel.copy()
    return(output_kspace, im_0 ,deltas)







def softimpute_ALS(X_H, M_H, rank, lamda, n_iters):
    I = np.eye(rank)
    m, n = np.shape(X_H)
    U = np.random.randn(m, rank) + 1j * np.random.randn(m, rank)
    V = np.random.randn(n, rank) + 1j * np.random.randn(n, rank)

    D = I.copy()

    A = np.dot(U, D)
    B = np.dot(V, D)
    iter_count = 0

    ABt    = np.empty((m, n), dtype=np.complex64)
    X_star = np.empty((m, n), dtype=np.complex64)

    Bh = B.conj().T
    np.matmul(A, Bh, out=ABt)

    iter_times = []
    t_total_start = time.perf_counter()
    print("hit iters")
    while iter_count < n_iters:
        t_start = time.perf_counter()

        np.copyto(X_star, ABt)
        np.copyto(X_star, X_H, where=M_H) 

        A = X_star @ B @ np.linalg.inv(Bh @ B + lamda*I)

        np.matmul(A, Bh, out=ABt)
        np.copyto(X_star, X_H, where=M_H)

        Ah = A.conj().T
        B = (Ah @ X_star).conj().T @ np.linalg.inv(Ah @ A + lamda*I)


        Bh = B.conj().T
        np.matmul(A, Bh, out=ABt)
        iter_count += 1

        t_elapsed = time.perf_counter() - t_start
        iter_times.append(t_elapsed)

    t_total = time.perf_counter() - t_total_start
    t_mean = np.mean(iter_times)

    return (ABt, t_total, t_mean, iter_times)



def softimpute_ALS_ortho(X_H, M_H, rank, lamda, n_iters, seed):
        I = np.eye(rank)
        m,n = np.shape(X_H)
        rng = np.random.default_rng(seed)
        U = rng.standard_normal((m, rank)) + 1j * rng.standard_normal((m, rank))
        V = np.zeros((n, rank),dtype=np.complex64)
        U, _ = np.linalg.qr(U)
        D = I.copy()


        A = np.dot(U,D)
        B = np.dot(V,D)
        iter_count = 0
        ABt = A @ B.conj().T
        
        while iter_count < n_iters:
            X_star = np.where(M_H, X_H, ABt)
            B = X_star.conj().T @ A @ np.linalg.inv(A.conj().T @ A + lamda*I)
            ABt = A @ B.conj().T
    
            X_star = np.where(M_H, X_H, ABt)
            A = X_star @ B @ np.linalg.inv(B.conj().T @ B + lamda*I)
            ABt = A @ B.conj().T
            iter_count += 1
        return(ABt)


def LORAKS_imputeals(n_iters, window_size, cartesian_inputkspace, dtg_mask, rank, lamda, stride):
    ksp_forhankel = cartesian_inputkspace.copy()
    ksp_zerod = ksp_forhankel * dtg_mask
    hankel_matrix, n_coils, Numx, Numy = gridding_hankel.hankel_2(kspace=ksp_zerod, w=window_size, s=stride)
    mask_coiled = np.broadcast_to(dtg_mask, (cartesian_inputkspace.shape))
    masked_hankel, *_ = gridding_hankel.hankel(mask_coiled, window_size)
    masked_hankel = np.real(masked_hankel) > 0.5
    
    
    filled_hankel, *_ = softimpute_ALS_ortho(X_H = hankel_matrix, M_H = masked_hankel, rank = rank, lamda = lamda, n_iters=n_iters)

    kspace_cart_coils_recon = gridding_hankel.hankel_H_averaged_2(filled_hankel, n_coils=ksp_forhankel.shape[0], Nx=ksp_forhankel.shape[1], Ny=ksp_forhankel.shape[2], w=window_size, s=stride)
    kspace_cart_coils_recon = np.where(dtg_mask, cartesian_inputkspace, kspace_cart_coils_recon)
    output_kspace = kspace_cart_coils_recon.copy()

    return output_kspace




def LORAKS_imputeals_ortho(n_iters, window_size, cartesian_inputkspace, dtg_mask, rank, lamda, stride, im_dim, enlarged_kspace, seed, inner_mask, inner_start, inner_end):
    ksp_forhankel = cartesian_inputkspace.copy()
    ksp_zerod = ksp_forhankel * dtg_mask
    hankel_matrix, n_coils, Numx, Numy = gridding_hankel.hankel_2(kspace=ksp_zerod, w=window_size, s=stride)
    mask_coiled = np.broadcast_to(dtg_mask, (cartesian_inputkspace.shape))
    masked_hankel, *_ = gridding_hankel.hankel_2(kspace=mask_coiled, w=window_size, s=stride)
    masked_hankel = np.real(masked_hankel) > 0.5
    
    
    filled_hankel = softimpute_ALS_ortho(X_H = hankel_matrix, M_H = masked_hankel, rank = rank, lamda = lamda, n_iters=n_iters,seed=seed)

    kspace_cart_coils_recon = gridding_hankel.hankel_H_averaged_2(filled_hankel, n_coils=ksp_forhankel.shape[0], Nx=ksp_forhankel.shape[1], Ny=ksp_forhankel.shape[2], w=window_size, s=stride)
    kspace_cart_coils_recon = np.where(dtg_mask, cartesian_inputkspace, kspace_cart_coils_recon)
    output_kspace = kspace_cart_coils_recon.copy()
    filled_ksp = zeropadding.rebuild(output_kspace=output_kspace, inner_mask = inner_mask, inner_start = inner_start, inner_end = inner_end, enlarged_kspace=enlarged_kspace, resize_x = im_dim, resize_y = im_dim)
    im_grid0 = sp.ifft(filled_ksp, axes=(-2, -1))
    im_0 = np.sum(np.abs(im_grid0)**2, axis=0)**0.5

    return (filled_ksp, im_0, masked_hankel)