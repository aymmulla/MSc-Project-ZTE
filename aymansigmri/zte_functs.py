import sigpy as sp
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



def data_consistency(kspace_radial, kspace_radial_acquired, mask):
    kspace_consistent = kspace_radial.copy()
    kspace_consistent[:, mask] = kspace_radial_acquired[:, mask]
    return kspace_consistent


def cartesian_data_consistency(kspace_cartsvd, kspace_cartacquired, halfmask_width):
    size = kspace_cartsvd[0,0].shape[0]
    cart_mask = np.ones((size, size),dtype=bool)
    
    cen_x, cen_y = size // 2, size // 2
    cart_mask[cen_x - halfmask_width:cen_x + halfmask_width, cen_y - halfmask_width: cen_y + halfmask_width] = False

    cartesian_kspace_consistent = kspace_cartsvd.copy()

    for coil in range(kspace_cartsvd.shape[0]):
        cartesian_kspace_consistent[coil][cart_mask] = kspace_cartacquired[coil][cart_mask]

    return cartesian_kspace_consistent, cart_mask



def generate_zte_data_trial(ksp, coord,n_missing,undersampling_factor=1):

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
    ksp_groundtruth = ksp_new.copy()
    ksp_full[:, :, :n_missing] = 0


    ksp_new, coord_new = point_remove(kspacedata=ksp_new, coord_data=coord_new, n=n_missing)
    ksp_new, coord_new = spoke_remove(kspacedata=ksp_new, coord_data=coord_new, unsampling=undersampling_factor)

    coord_full = coord_full[::undersampling_factor]
    ksp_full = ksp_full[:, ::undersampling_factor, :]

    n_spokes_full = coord_full.shape[0]
    n_points_full = coord_full.shape[1]
    acquired_mask = np.ones((n_spokes_full, n_points_full), dtype=bool)
    acquired_mask[:, :n_missing] = False

    return ksp_new, coord_new, ksp_full, coord_full, acquired_mask, ksp_groundtruth, n_points_full