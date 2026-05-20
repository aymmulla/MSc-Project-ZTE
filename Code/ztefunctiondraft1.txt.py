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
    return kspacedata[::unsampling], coord_data[::unsampling]

def point_remove(kspacedata, coord_data, n):
    return kspacedata[:,n:], coord_data[:,n:,:]


def generate_zte_data(ksp, coord,n_missing,undersampling_factor):

    n_spokes, n_points, n_dims = coord.shape
    half = n_points // 2

    ksp_new = np.zeros((n_spokes * 2, half), dtype=ksp.dtype)
    coord_new = np.zeros((n_spokes * 2, half, n_dims), dtype=coord.dtype)

    for spoke_index in range(n_spokes):
        spoke_a, spoke_b, coords_a, coords_b = spoke_split(ksp[spoke_index], coord[spoke_index])
        
        b_index = get_spoke_b_index(spoke_index, n_spokes)

        ksp_new[spoke_index] = spoke_a
        ksp_new[b_index] = spoke_b

        coord_new[spoke_index] = coords_a
        coord_new[b_index] = coords_b

    ksp_new, coord_new = point_remove(kspacedata=ksp_new, coord_data=coord_new, n=n_missing)
    ksp_new, coord_new = spoke_remove(kspacedata=ksp_new, coord_data=coord_new, unsampling=undersampling_factor)

    return ksp_new, coord_new

zte_radial_kspace, zte_radial_coords = generate_zte_data(ksp=ksp_combined, coord=coord,n_missing=1, undersampling_factor=1)

dcf = np.sqrt(zte_radial_coords[..., 0]**2 + zte_radial_coords[..., 1]**2)
zte_img_grid = sp.nufft_adjoint(zte_radial_kspace * dcf, zte_radial_coords)

reduced_spokes_img = np.abs(zte_img_grid)
pl.ImagePlot(reduced_spokes_img, title='test')