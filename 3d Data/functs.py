import numpy as np
import time
import sigpy as sp
import sys
from pathlib import Path
sys.path.insert(0,'/Users/ayman/Desktop/MSc Project Local/MSc-Project-ZTE')
import aymansigmri as asm
import jax as jx



def hankel(kspace, w):
    # kspace: (N_c, X, Y, Z)
    N_c = kspace.shape[0]
    N_x = kspace.shape[1] - w + 1
    N_y = kspace.shape[2] - w + 1
    N_z = kspace.shape[3] - w + 1
    data_matrix = []
    for c in range(N_c):
        onecoil_matrix = np.empty((w*w*w, N_x*N_y*N_z), dtype=np.complex64)
        col_index = 0
        for k in range(N_z):        # z outermost
            for i in range(N_y):
                for j in range(N_x):    # x innermost, matching your 2D ordering
                    mat = kspace[c, j:j+w, i:i+w, k:k+w]
                    onecoil_matrix[:, col_index] = mat.reshape(-1, order='F')
                    col_index += 1
        data_matrix.append(onecoil_matrix)
    return np.vstack(data_matrix)

def hankel_H_averaged(data_matrix, n_coils, Nx, Ny, Nz, w):
    kspace = []
    split_arr = np.array(np.split(data_matrix, n_coils))

    n_win_x = Nx - w + 1
    n_win_y = Ny - w + 1
    n_win_z = Nz - w + 1

    for c in range(n_coils):
        singlecoil = split_arr[c]
        transposed = singlecoil.T          # one row per window

        recon = np.zeros((Nx, Ny, Nz), dtype=np.complex64)
        counts = np.zeros((Nx, Ny, Nz), dtype=np.float32)

        for index in range(len(transposed)):
            win = transposed[index].reshape((w, w, w), order='F')

            j = index % n_win_x
            i = (index // n_win_x) % n_win_y
            k = index // (n_win_x * n_win_y)

            recon[j:j+w, i:i+w, k:k+w] += win
            counts[j:j+w, i:i+w, k:k+w] += 1

        kspace.append(recon / counts)
    return np.array(kspace)

def OLDsoftimpute_ALS_time(X_H, M_H, rank, lamda, n_iters):
    I = np.eye(rank)
    m, n = np.shape(X_H)
    U = np.random.randn(m, rank) + 1j * np.random.randn(m, rank)
    V = np.random.randn(n, rank) + 1j * np.random.randn(n, rank)

    D = I.copy()

    A = np.dot(U, D)
    B = np.dot(V, D)
    iter_count = 0
    ABt = A @ B.conj().T

    iter_times = []
    t_total_start = time.perf_counter()

    while iter_count < n_iters:
        t_start = time.perf_counter()

        X_star = np.where(M_H, X_H, ABt)
        X_star1H = X_star.copy() ##removed
        A = X_star @ B @ np.linalg.inv(B.conj().T @ B + lamda*I)
        ABt = A @ B.conj().T
        X_star = np.where(M_H, X_H, ABt)
        B = X_star.conj().T @ A @ np.linalg.inv(A.conj().T @ A + lamda*I)
        ABt = A @ B.conj().T
        iter_count += 1

        t_elapsed = time.perf_counter() - t_start
        iter_times.append(t_elapsed)

    t_total = time.perf_counter() - t_total_start
    t_mean = np.mean(iter_times)

    return (ABt, X_star1H, t_total, t_mean, iter_times)




def softimpute_ALS_time(X_H, M_H, rank, lamda, n_iters):
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


def LORAKS_loop_timing(n_iters, window_size, zero_thresh, cartesian_inputkspace , inner_mask, start, end, zeropadded,dtg_mask, im_dim):
    ksp_forhankel = cartesian_inputkspace.copy()
    iter_count = 0
    deltas = []
    k_prev = None
    timings = {"hankel": [], "svd": [], "unlift": [], "consistency": [], "iter_total": [], "total": []}
    hankel_size = []
    t_loop_start = time.perf_counter()
    while iter_count < n_iters:
        t0 = time.perf_counter()

        hankel_matrix= hankel(kspace=ksp_forhankel, w=window_size)
        t1 = time.perf_counter()

        U, S_reduced, Vh = asm.sig_val_thresholding_jax(data=hankel_matrix, zero_thresh=zero_thresh)
        data_recon = (U * S_reduced) @ Vh
        data_recon = jx.block_until_ready(data_recon)
        t2 = time.perf_counter()

        kspace_cart_coils_recon = hankel_H_averaged(
            data_recon, n_coils=ksp_forhankel.shape[0],
            Nx=ksp_forhankel.shape[1], Ny=ksp_forhankel.shape[2],
            w=window_size)
        t3 = time.perf_counter()

        kspace_cart_coils_consistent = np.where(dtg_mask, cartesian_inputkspace, kspace_cart_coils_recon)
        ksp_forhankel = np.asarray(kspace_cart_coils_consistent)  # ensure NumPy, on-host
        t4 = time.perf_counter()

        vec = ksp_forhankel[:, ~dtg_mask]
        if k_prev is not None:
            deltas.append(np.linalg.norm(vec - k_prev) / np.linalg.norm(k_prev))
        k_prev = vec.copy()

        timings["hankel"].append(t1 - t0)
        timings["svd"].append(t2 - t1)
        timings["unlift"].append(t3 - t2)
        timings["consistency"].append(t4 - t3)
        timings["iter_total"].append(t4 - t0)
        hankel_size.append(hankel_matrix.shape)

        iter_count += 1
    t_loop_end = time.perf_counter()
    timings["total"].append(t_loop_end - t_loop_start)
    output_kspace = ksp_forhankel.copy()
    filled_ksp = asm.rebuild(output_kspace=output_kspace, inner_mask=inner_mask,inner_start=start, inner_end=end,enlarged_kspace=zeropadded,resize_x=im_dim, resize_y=im_dim)
    im_grid0 = sp.ifft(filled_ksp, axes=(-2, -1))
    im_0 = np.sum(np.abs(im_grid0)**2, axis=0)**0.5

    return (im_0, filled_ksp, deltas, timings, hankel_size)

def LORAKS_imputeals(n_iters, window_size, cartesian_inputkspace, inner_mask, start, end, dtg_mask, zeropadded,rank, lamda, im_dim):
    ksp_forhankel = cartesian_inputkspace.copy()
    ksp_zerod = ksp_forhankel * dtg_mask
    hankel_matrix = hankel(kspace=ksp_zerod, w=window_size)
    mask_coiled = np.broadcast_to(dtg_mask, (cartesian_inputkspace.shape))
    masked_hankel = hankel(kspace=mask_coiled, w=window_size)
    masked_hankel = np.real(masked_hankel) > 0.5
    
    
    filled_hankel, t_total, t_mean, iter_times = softimpute_ALS_time(X_H = hankel_matrix, M_H = masked_hankel, rank = rank, lamda = lamda, n_iters=n_iters)

    kspace_cart_coils_recon = hankel_H_averaged(filled_hankel, n_coils=ksp_forhankel.shape[0], Nx=ksp_forhankel.shape[1], Ny=ksp_forhankel.shape[2], w=window_size)
    kspace_cart_coils_recon = np.where(dtg_mask, cartesian_inputkspace, kspace_cart_coils_recon)
    output_kspace = kspace_cart_coils_recon.copy()
    filled_ksp = asm.rebuild(output_kspace=output_kspace, inner_mask = inner_mask, inner_start = start, inner_end = end, enlarged_kspace=zeropadded, resize_x = im_dim, resize_y = im_dim)
    im_grid0 = sp.ifft(filled_ksp, axes=(-2, -1))
    im_0 = np.sum(np.abs(im_grid0)**2, axis=0)**0.5

    return (im_0, filled_ksp, masked_hankel, t_total, t_mean, iter_times)