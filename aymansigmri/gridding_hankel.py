import sigpy as sp
import numpy as np

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