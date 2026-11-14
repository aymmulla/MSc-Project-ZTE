import matplotlib.pyplot as plt
import sigpy as sp
import numpy as np

def zero_padding(cart_kspace,resize_x, resize_y):
    n_coils = cart_kspace.shape[0]
    image_grid = sp.ifft(cart_kspace)
    enlarged_image_grid = sp.resize(image_grid, [n_coils,resize_x, resize_y])
    enlarged_cartesian_kspace = sp.fft(enlarged_image_grid, axes=(-2, -1))
    return enlarged_cartesian_kspace


def inner_portion(enlarged_kspace, inner_sidelen):
    """Take an inner square of kspace to speed up the hankel loop

        Args:
            enlarged_kspace: Zero padded kspace in cartesian coordinates (n_coils, nx, ny)
            inner_sidelen: Side legnth of the inner square, divided by 2 to find from central point outward

        Returns:
            isolated_kspace: Inner kspace array
            isolation_mask: Mask of the inner region to use when 'jigsawing' the transformed array back in
            Start: xy index of the enlarged array that the inner region starts at
            End: xy index of the enlarged array that the inner region ends at
    """
    cx, cy = enlarged_kspace.shape[1] // 2, enlarged_kspace.shape[2] // 2
    N = inner_sidelen/2
    start, end = int(cx-N), int(cx+N)
    isolation_mask = np.ones(enlarged_kspace.shape[1:], dtype=int)
    isolation_mask[start:end, start:end] = 0
    isolated_kspace = enlarged_kspace[:, start:end, start:end]
    
    return(isolated_kspace, start, end)

def jigsaw(output_kspace, start, end, enlarged_kspace):
    ### NOTE: There is a better way to do this without using start and end
    """Put the transormed region of kspace back into the zero padded region
        Args:
            output_kspace: Transformed SVD thresh kspace region
            isolation_mask: Mask of the inner region; typically output of asm.inner_portion
            Start: xy index of the enlarged array that the inner region starts at; typically output of asm.inner_portion 
            End: xy index of the enlarged array that the inner region ends at; typically output of asm.inner_portion
            enlarged_kspace: Zero padded kspace in cartesian coordinates (n_coils, nx, ny)

        Returns:
            recombined: Zero padded kspace with transfomed inner region 'jigsawed' back in
    """
    #processed_isolated_kspace = output_kspace.copy()
    #recombined = enlarged_kspace * isolation_mask ### NOTE: Need to improve this
    recombined = enlarged_kspace.copy()
    recombined[:, start:end, start:end] = output_kspace
    return(recombined)

def rebuild(output_kspace, inner_start, inner_end, enlarged_kspace, im_dim):
    recombined = jigsaw(output_kspace=output_kspace, start=inner_start, end=inner_end, enlarged_kspace=enlarged_kspace)
    filled_ksp = zero_padding(recombined, im_dim, im_dim)
    return(filled_ksp, recombined)




def zeropadding3d(data, resize_factor, input_kspace=True, output_kspace=True):
    """Zero pad of kspace
    
            Args:
                data: Dataset in cartesian coordinates (n_coils, nx, ny)
                im_dim: Requested output dimensions
                input_kpsace: True if data is a kspace array
                output_kpsace: True if wanting kspace output
    
            Returns:
                Zeropadded data of requested domain
        """
    # NOTE: Give options to input and output kspace in order to save computation
    n_coils = data.shape[0]
    resizex = data.shape[1]*resize_factor
    resizey = data.shape[2]*resize_factor
    resizez = data.shape[3]*resize_factor

    if not all(float(r).is_integer() for r in (resizex, resizey, resizez)):
        print(f"Warning: non-integer resize dims: {resizex}, {resizey}, {resizez}")

    resizex, resizey, resizez = int(resizex), int(resizey), int(resizez)
    print('starting resize')
    if input_kspace:
        enlarged_image_grid = sp.resize(sp.ifft(data,axes=(-3, -2, -1)), [n_coils,resizex, resizey, resizez])
        print('fft done input zp')
    else:
        enlarged_image_grid = sp.resize(data, [n_coils,resizex, resizey, resizez])
    if output_kspace:
        enlarged_cartesian_kspace = sp.fft(enlarged_image_grid, axes=(-3, -2, -1))
        print('fft done output zp')
        return enlarged_cartesian_kspace
    else:
        return enlarged_image_grid





def inner_portion3d(enlarged_kspace, inner_sidelen):
    """Take an inner square of kspace to speed up the hankel loop

        Args:
            enlarged_kspace: Zero padded kspace in cartesian coordinates (n_coils, nx, ny)
            inner_sidelen: Side legnth of the inner square, divided by 2 to find from central point outward

        Returns:
            isolated_kspace: Inner kspace array
            isolation_mask: Mask of the inner region to use when 'jigsawing' the transformed array back in
            Start: xy index of the enlarged array that the inner region starts at
            End: xy index of the enlarged array that the inner region ends at
    """
    cx, cy, cz = enlarged_kspace.shape[1] // 2, enlarged_kspace.shape[2] // 2, enlarged_kspace.shape[3] // 2
    N = inner_sidelen/2
    start, end = int(cx-N), int(cx+N)
    isolation_mask = np.ones(enlarged_kspace.shape[1:], dtype=int)
    isolation_mask[start:end, start:end, start:end] = 0
    isolated_kspace = enlarged_kspace[:, start:end, start:end, start:end]
    
    return(isolated_kspace, isolation_mask, start, end)

def jigsaw3d(output_kspace, isolation_mask, start, end, enlarged_kspace):
    ### NOTE: There is a better way to do this without using start and end
    """Put the transormed region of kspace back into the zero padded region
        Args:
            output_kspace: Transformed SVD thresh kspace region
            isolation_mask: Mask of the inner region; typically output of asm.inner_portion
            Start: xy index of the enlarged array that the inner region starts at; typically output of asm.inner_portion 
            End: xy index of the enlarged array that the inner region ends at; typically output of asm.inner_portion
            enlarged_kspace: Zero padded kspace in cartesian coordinates (n_coils, nx, ny)

        Returns:
            recombined: Zero padded kspace with transfomed inner region 'jigsawed' back in
    """
    #processed_isolated_kspace = output_kspace.copy()
    #recombined = enlarged_kspace * isolation_mask ### NOTE: Need to improve this
    #recombined[:, start:end, start:end, start:end] = output_kspace
    enlarged_kspace[:, start:end, start:end, start:end] = output_kspace
    return(enlarged_kspace)

def rebuild3d(output_kspace, inner_mask, inner_start, inner_end, enlarged_kspace, im_dim):
    recombined = jigsaw3d(output_kspace=output_kspace, isolation_mask = inner_mask, start=inner_start, end=inner_end, enlarged_kspace=enlarged_kspace)
    filled_ksp = zero_padding3d(recombined, im_dim, im_dim,im_dim)
    return(filled_ksp)
