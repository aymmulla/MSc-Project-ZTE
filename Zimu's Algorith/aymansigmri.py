import matplotlib.pyplot as plt
import sigpy as sp
import pandas as pd
import sigpy.plot as pl
import numpy as np


#First 4 funtions used to get toy ZTE style data

def spoke_split(kspacedata, coords):
    halfwaypoint = kspacedata.shape[0] // 2
    
    spoke_a = np.flip(kspacedata[:halfwaypoint], axis=0)
    spoke_b = kspacedata[halfwaypoint:]
    
    coords_a = np.flip(coords[:halfwaypoint, :], axis=0)
    coords_b = coords[halfwaypoint:, :]
    
    return spoke_a, spoke_b, coords_a, coords_b

def get_spoke_b_index(spoke_index, n_spokes):
    return n_spokes + spoke_index

def spoke_remove(kspacedata, coord_data, unsampling):
    return kspacedata[:,::unsampling], coord_data[::unsampling]

def point_remove(kspacedata, coord_data, n):
    return kspacedata[:,:,n:], coord_data[:,n:,:]



def generate_zte_data_OLD(ksp, coord,n_missing,undersampling_factor=1):
#basic zte data no extra info, useful for visualisation
    n_spokes, n_points, n_dims = coord.shape
    n_coils = ksp.shape[0]
    half = n_points // 2

    ksp_new = np.zeros((n_coils,n_spokes * 2, half), dtype=ksp.dtype)
    coord_new = np.zeros((n_spokes * 2, half, n_dims), dtype=coord.dtype)
    
    for coil_index in range(n_coils):
        for spoke_index in range(n_spokes):
            spoke_a, spoke_b, coords_a, coords_b = spoke_split(ksp[coil_index,spoke_index], coord[spoke_index])
            
            b_index = get_spoke_b_index(spoke_index, n_spokes)

            ksp_new[coil_index, spoke_index] = spoke_a
            ksp_new[coil_index,b_index] = spoke_b

            coord_new[spoke_index] = coords_a
            coord_new[b_index] = coords_b

    ksp_new, coord_new = point_remove(kspacedata=ksp_new, coord_data=coord_new, n=n_missing)
    ksp_new, coord_new = spoke_remove(kspacedata=ksp_new, coord_data=coord_new, unsampling=undersampling_factor)

    return ksp_new, coord_new


def generate_zte_data(ksp, coord,n_missing,undersampling_factor=1):
#basic zte data no extra info, useful for visualisation
    n_spokes, n_points, n_dims = coord.shape
    n_coils = ksp.shape[0]
    half = n_points // 2

    ksp_new = np.zeros((n_coils,n_spokes * 2, half), dtype=ksp.dtype)
    coord_new = np.zeros((n_spokes * 2, half, n_dims), dtype=coord.dtype)
    
    for coil_index in range(n_coils):
        for spoke_index in range(n_spokes):
            spoke_a, spoke_b, coords_a, coords_b = spoke_split(ksp[coil_index,spoke_index], coord[spoke_index])
            
            b_index = get_spoke_b_index(spoke_index, n_spokes)

            ksp_new[coil_index, spoke_index] = spoke_a
            ksp_new[coil_index,b_index] = spoke_b

            coord_new[spoke_index] = coords_a
            coord_new[b_index] = coords_b
    
    coord_full = coord_new.copy()
    ksp_full = ksp_new.copy()
    ksp_full[:, :, :n_missing] = 0


    ksp_new, coord_new = point_remove(kspacedata=ksp_new, coord_data=coord_new, n=n_missing)
    ksp_new, coord_new = spoke_remove(kspacedata=ksp_new, coord_data=coord_new, unsampling=undersampling_factor)

    coord_full = coord_full[::undersampling_factor]
    ksp_full = ksp_full[:, ::undersampling_factor, :]

    n_spokes_full = coord_full.shape[0]
    n_points_full = coord_full.shape[1]
    acquired_mask = np.ones((n_spokes_full, n_points_full), dtype=bool)
    acquired_mask[:, :n_missing] = False

    return ksp_new, coord_new, ksp_full, coord_full, acquired_mask




def gridding_one_coil(kspace_data, coord):
    dcf = np.sqrt(coord[..., 0]**2 + coord[..., 1]**2)
    shape=(len(coord[1]), len(coord[1]))
    #coord_reindexed = coord + np.array(shape) // 2 
    kspace_cartesian = sp.gridding(kspace_data * dcf, coord, shape=shape,kernel='kaiser_bessel')
    kspace_cartesian = np.fft.fftshift(kspace_cartesian)
    return kspace_cartesian


def gridding_operator(kspace_data, coord):
    n_coils = kspace_data.shape[0]
    shape=(len(coord[1]), len(coord[1]))
    kspace_cartesian = np.zeros((n_coils, *shape), dtype=complex)
    
    dcf = np.sqrt(coord[..., 0]**2 + coord[..., 1]**2)
    

    
    for coil in range(n_coils):
        kspace_cartesian[coil] = sp.gridding(kspace_data[coil] * dcf, coord, shape=shape, kernel='kaiser_bessel')
        kspace_cartesian[coil] = np.fft.fftshift(kspace_cartesian[coil])
    
    return kspace_cartesian


def gridding_operator_H(cartesian_data,coord):
    n_coils = cartesian_data.shape[0]
    spokes_points = coord.shape[:-1]
    kspace_data = np.zeros((n_coils, *spokes_points), dtype=complex)
    
    for coil in range(n_coils):
        cartesian_data[coil] = np.fft.ifftshift(cartesian_data[coil])
        kspace_data[coil] = sp.interpolate(cartesian_data[coil], coord)
    
    return kspace_data


def nufft_gridding(kspace, coords):
    n_coils = kspace.shape[0]
    n_samples = kspace.shape[-1]
    dcf = np.sqrt(coords[..., 0]**2 + coords[..., 1]**2)
    imagegrid = sp.nufft_adjoint(kspace * dcf, coords, oshape=(n_coils, n_samples, n_samples))
    gridded_data = sp.fft(imagegrid, axes=(-2, -1))
    return gridded_data


def nufft_degridding(cartesian_kspace, coords):
    n_coils = cartesian_kspace.shape[0]
    image = sp.ifft(cartesian_kspace, axes=(-2, -1))
    radial_kspace = sp.nufft(image, coord=coords)
    return radial_kspace

def hankel(kspace, w):
    x = 0
    y = 0
    c = 0
    N_x = kspace.shape[1] - w + 1 
    N_y = kspace.shape[2] - w + 1
    N_c = kspace.shape[0]
    data_matrix = []
    for k in range(N_c):
        onecoil_matrix = np.empty((w*w,N_x* N_y),dtype=np.complex64)
        col_index = 0
        for i in range(N_y):        
            for j in range(N_x):    
                mat = kspace[c,x:x+w, y:y+w]
                col = mat.T.reshape(-1)
                onecoil_matrix[:, col_index] = col
                x += 1
                col_index += 1
            x = 0
            y += 1
        data_matrix.append(onecoil_matrix)
        y = 0
        c += 1
    return(np.vstack(data_matrix), N_c, N_x, N_y)


def hankel_H(data_matrix, n_coils, Nx, Ny, w=3):
    kspace = []
    split_arr = np.array(np.split(data_matrix, n_coils))

    n_win_x = Nx - w + 1
    n_win_y = Ny - w + 1

    for k in range(len(split_arr)):
        #n_windows = split_arr.shape[1]

        singlecol = split_arr[k]
        transposed = singlecol.T
        windows = []
        for j in range(len(transposed)):
            window = transposed[j].reshape(w,w)
            windows.append(window)
        windows_full = np.array(windows)
            

        
        recon = np.zeros((Nx, Ny), dtype=np.complex64)

        for index, win in enumerate(windows_full): 
            i = (index // n_win_y)
            j = (index % n_win_y)
            recon[i:i+w, j:j+w] = win
        kspace.append(recon.T)
    return(np.array(kspace))



def hankel_H_averaged(data_matrix, n_coils, Nx, Ny, w=3):
    kspace = []
    split_arr = np.array(np.split(data_matrix, n_coils))

    n_win_x = Nx - w + 1
    n_win_y = Ny - w + 1

    for k in range(len(split_arr)):
        singlecol = split_arr[k]
        transposed = singlecol.T
        windows = []
        for j in range(len(transposed)):
            window = transposed[j].reshape(w,w)
            windows.append(window)
        windows_full = np.array(windows)
        
        recon = np.zeros((Nx, Ny), dtype=np.complex64)
        counts = np.zeros((Nx, Ny), dtype=np.float32)

        for index, win in enumerate(windows_full): 
            i = (index // n_win_y)
            j = (index % n_win_y)
            recon[i:i+w, j:j+w] += win
            counts[i:i+w, j:j+w] += 1

        recon = recon / counts
        kspace.append(recon.T)
    return np.array(kspace)


def sig_val_thresholding(data, zero_thresh):
    U, Sigma, Vh = np.linalg.svd(data, full_matrices=False)
    max_val = np.max(Sigma)
    thresh = max_val*zero_thresh
    arr = []
    for i in range(len(Sigma)):
        if Sigma[i] > thresh:
            arr.append(Sigma[i])
        else: arr.append(0)
    S_reduced = np.array(arr)
    return(U, S_reduced, Vh)

def svd_recon(u, s_reduced, vh):
    return ((u*s_reduced) @ vh)



def data_consistency(kspace_radial, kspace_radial_acquired, mask):
    kspace_consistent = kspace_radial.copy()
    kspace_consistent[:, mask] = kspace_radial_acquired[:, mask]
    return kspace_consistent

