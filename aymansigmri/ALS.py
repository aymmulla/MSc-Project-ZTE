import numpy as np
from . import gridding_hankel

def softimpute_ALS(X_H, M_H, rank, lamda, n_iters):
        I = np.eye(rank)
        m,n = np.shape(X_H)
        U = np.random.randn(m, rank) + 1j * np.random.randn(m, rank)
        V = np.random.randn(n, rank) + 1j * np.random.randn(n, rank)

        D = I.copy()


        A = np.dot(U,D)
        B = np.dot(V,D)
        iter_count = 0
        ABt = A @ B.conj().T
        
        while iter_count < n_iters:
            X_star = np.where(M_H, X_H, ABt)
            X_star1H = X_star.copy()
            A = X_star @ B @ np.linalg.inv(B.conj().T @ B + lamda*I)
            ABt = A @ B.conj().T
            X_star = np.where(M_H, X_H, ABt)
            B = X_star.conj().T @ A @ np.linalg.inv(A.conj().T @ A + lamda*I)
            ABt = A @ B.conj().T
            iter_count += 1
        return(ABt, X_star1H)

def LORAKS_imputeals(n_iters, window_size, cartesian_inputkspace, dtg_mask, rank, lamda, stride):
    ksp_forhankel = cartesian_inputkspace.copy()
    ksp_zerod = ksp_forhankel * dtg_mask
    hankel_matrix, n_coils, Numx, Numy = gridding_hankel.hankel_2(kspace=ksp_zerod, w=window_size, s=stride)
    mask_coiled = np.broadcast_to(dtg_mask, (cartesian_inputkspace.shape))
    masked_hankel, *_ = gridding_hankel.hankel(mask_coiled, window_size)
    masked_hankel = np.real(masked_hankel) > 0.5
    
    
    filled_hankel, X_star1 = softimpute_ALS(X_H = hankel_matrix, M_H = masked_hankel, rank = rank, lamda = lamda, n_iters=n_iters)

    kspace_cart_coils_recon = gridding_hankel.hankel_H_averaged_2(filled_hankel, n_coils=ksp_forhankel.shape[0], Nx=ksp_forhankel.shape[1], Ny=ksp_forhankel.shape[2], w=window_size, s=stride)
    kspace_cart_coils_recon = np.where(dtg_mask, cartesian_inputkspace, kspace_cart_coils_recon)
    output_kspace = kspace_cart_coils_recon.copy()

    return output_kspace, masked_hankel