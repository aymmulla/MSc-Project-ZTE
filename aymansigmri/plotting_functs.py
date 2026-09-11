import numpy as np
import matplotlib.pyplot as plt
import sigpy as sp
import sigpy.plot as pl


def plot_result(initial_cart, post_cart, initial_zerop, post_zerop, lamda, rank):
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(30, 16))


    im1 = np.sum(np.abs(sp.ifft(initial_cart))**2, axis=0)**0.5
    p1 = ax[0, 0].imshow(im1, cmap='gray', origin='lower')
    ax[0, 0].set_title(f'Pre', fontsize=24)
    fig.colorbar(p1, ax=ax[0, 0])


    im2 = np.sum(np.abs(sp.ifft(post_cart))**2, axis=0)**0.5
    p1_soft = ax[0, 1].imshow(im2, cmap='gray', origin='lower')
    ax[0, 1].set_title(f'Post, rank: {rank}, lamda: {lamda}', fontsize=24)
    fig.colorbar(p1_soft, ax=ax[0, 1])


    p2 = ax[0, 2].imshow(im1-im2, cmap='gray', origin='lower')
    ax[0, 2].set_title('difference', fontsize=24)
    fig.colorbar(p2, ax=ax[0, 2])



    im1k = ax[1, 0].imshow(np.abs(initial_zerop[0]), cmap='viridis', origin='lower')
    fig.colorbar(im1k, ax=ax[1, 0], label='kspace value')
    ax[1, 0].set_title(f'Pre loop', fontsize=24)
    ax[1, 0].set_xlabel('kx')
    ax[1, 0].set_ylabel('ky')
    ax[1, 0].axis('equal')
    ax[1, 0].set_xlim(100/2 - 6, 100/2 + 6)
    ax[1, 0].set_ylim(100/2 - 6, 100/2 + 6)
    ax[1, 0].xaxis.set_minor_locator(plt.MultipleLocator(1, offset=0.5))
    ax[1, 0].yaxis.set_minor_locator(plt.MultipleLocator(1, offset=0.5))
    ax[1, 0].grid(visible=True, which='minor', linewidth=1)
    ax[1, 0].xaxis.set_major_locator(plt.MultipleLocator(1))
    ax[1, 0].yaxis.set_major_locator(plt.MultipleLocator(1))


    im1k_soft = ax[1, 1].imshow(np.abs(post_zerop[0]), cmap='viridis', origin='lower')
    fig.colorbar(im1k_soft, ax=ax[1, 1], label='kspace value')
    ax[1, 1].set_title(f'Post Loop, rank: {rank}, lamda: {lamda}', fontsize=24)
    ax[1, 1].set_xlabel('kx')
    ax[1, 1].set_ylabel('ky')
    ax[1, 1].axis('equal')
    ax[1, 1].set_xlim(100/2 - 6, 100/2 + 6)
    ax[1, 1].set_ylim(100/2 - 6, 100/2 + 6)
    ax[1, 1].xaxis.set_minor_locator(plt.MultipleLocator(1, offset=0.5))
    ax[1, 1].yaxis.set_minor_locator(plt.MultipleLocator(1, offset=0.5))
    ax[1, 1].grid(visible=True, which='minor', linewidth=1)
    ax[1, 1].xaxis.set_major_locator(plt.MultipleLocator(1))
    ax[1, 1].yaxis.set_major_locator(plt.MultipleLocator(1))


    diff = initial_zerop - post_zerop

    im2k = ax[1, 2].imshow(np.abs(diff[0]), cmap='viridis', origin='lower')
    fig.colorbar(im2k, ax=ax[1, 2], label='kspace value')
    ax[1, 2].set_title('difference', fontsize=24)
    ax[1, 2].set_xlabel('kx')
    ax[1, 2].set_ylabel('ky')
    ax[1, 2].axis('equal')
    ax[1, 2].set_xlim(100/2 - 6, 100/2 + 6)
    ax[1, 2].set_ylim(100/2 - 6, 100/2 + 6)
    ax[1, 2].xaxis.set_minor_locator(plt.MultipleLocator(1, offset=0.5))
    ax[1, 2].yaxis.set_minor_locator(plt.MultipleLocator(1, offset=0.5))
    ax[1, 2].grid(visible=True, which='minor', linewidth=1)
    ax[1, 2].xaxis.set_major_locator(plt.MultipleLocator(1))
    ax[1, 2].yaxis.set_major_locator(plt.MultipleLocator(1))

    plt.tight_layout()
    plt.show()


def plot_mask(isolated_kspace, mask, zte_radial_coords, zte_radial_kspace, innersidelen, sidelencart, sidelenrad, n, eps=0.3):
    
    all_coords = zte_radial_coords.reshape(-1, 2)

    # radius of the dead-time gap = distance of the nth point along a spoke from origin
    gap_radius_rad = np.linalg.norm(zte_radial_coords[0, n-1]) + eps
    gap_radius_cart = 2 * np.linalg.norm(zte_radial_coords[0, n-1]) + eps

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 8))
    
    im = ax[0].imshow(np.abs(isolated_kspace[0]), cmap='viridis', origin='lower')
    fig.colorbar(im, ax=ax[0], label='kspace value')
    ax[0].set_title('Cartesian')
    ax[0].set_xlabel('kx')
    ax[0].set_ylabel('ky')
    ax[0].set_xlim(innersidelen/2 - sidelencart,innersidelen/2 + sidelencart)
    ax[0].set_ylim(innersidelen/2 - sidelencart,innersidelen/2 + sidelencart)
    ax[0].xaxis.set_minor_locator(plt.MultipleLocator(1, offset=0.5))
    ax[0].yaxis.set_minor_locator(plt.MultipleLocator(1, offset=0.5))
    ax[0].grid(visible=True, which='minor', linewidth=1)
    ax[0].xaxis.set_major_locator(plt.MultipleLocator(1))
    ax[0].yaxis.set_major_locator(plt.MultipleLocator(1))
    ax[0].add_patch(plt.Circle((innersidelen/2, innersidelen/2), gap_radius_cart,
                               fill=False, ec='red', lw=4, ls='--', zorder=10, label='Dead-time gap'))
    ax[0].legend()

    yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[0]]
    mask_flat = mask.ravel().astype(bool)
    ax[1].scatter(xx.ravel()[mask_flat], yy.ravel()[mask_flat], c='yellow', s=300, marker='s', zorder=3, label='mask True')
    ax[1].scatter(xx.ravel()[~mask_flat], yy.ravel()[~mask_flat], c='purple', s=300, marker='s', zorder=3, label='mask False')
    ax[1].set_title('Centre mask')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].axis('equal')
    ax[1].set_xlim(innersidelen/2 - sidelencart,innersidelen/2 + sidelencart)
    ax[1].set_ylim(innersidelen/2 - sidelencart,innersidelen/2 + sidelencart)
    ax[1].xaxis.set_minor_locator(plt.MultipleLocator(1, offset=0.5))
    ax[1].yaxis.set_minor_locator(plt.MultipleLocator(1, offset=0.5))
    ax[1].grid(visible=True, which='minor', linewidth=1)
    ax[1].xaxis.set_major_locator(plt.MultipleLocator(1))
    ax[1].yaxis.set_major_locator(plt.MultipleLocator(1))
    ax[1].add_patch(plt.Circle((innersidelen/2, innersidelen/2), gap_radius_cart,
                               fill=False, ec='red', lw=4, ls='--', zorder=10, label='Dead-time gap'))
    ax[1].legend()

    im2 = ax[2].scatter(all_coords[:, 0],all_coords[:, 1], c=np.abs(zte_radial_kspace[1]), cmap='viridis')
    fig.colorbar(im2, ax=ax[2], label='kspace value')
    ax[2].set_title('Radial')
    ax[2].set_xlabel('kx')
    ax[2].set_ylabel('ky')
    ax[2].axis('equal')
    ax[2].xaxis.set_minor_locator(plt.MultipleLocator(1, offset=0.5))
    ax[2].yaxis.set_minor_locator(plt.MultipleLocator(1, offset=0.5))
    ax[2].grid(visible=True, which='minor', linewidth=1)
    ax[2].xaxis.set_major_locator(plt.MultipleLocator(1))
    ax[2].yaxis.set_major_locator(plt.MultipleLocator(1))
    ax[2].set_xlim(0 - (sidelenrad//2), 0 + (sidelenrad//2))
    ax[2].set_ylim(0 - sidelenrad//2, 0 + (sidelenrad//2))
    ax[2].add_patch(plt.Circle((0, 0), gap_radius_rad,
                               fill=False, ec='red', lw=4, ls='--', zorder=10, label='Dead-time gap'))
    ax[2].legend()

    plt.show()

def im_recon_zte(kspace, radial_coords):
    dcf = (radial_coords[...,0]**2 + radial_coords[...,1]**2)**0.5
    im_grid = sp.nufft_adjoint(kspace* dcf, radial_coords)
    img_rss = np.sum(np.abs(im_grid)**2, axis=0)**0.5
    pl.ImagePlot(img_rss)


def im_recon_cart(kspace):
    grid_recon = sp.ifft(kspace)
    im_rss_recon = np.sum(np.abs(grid_recon)**2, axis=0)**0.5
    pl.ImagePlot(im_rss_recon)


def plotdiff(fig, index1, index2, imagearr, axname, numims, cmap='RdBu_r', fontsize=18, colorbar=True):
    if not index2 > numims-1:
        diff = (np.abs(np.abs(imagearr[index1]) - np.abs(imagearr[index2]))) / np.max(np.abs(imagearr[index1]))
        diff_plot = axname[index1, index2].imshow(diff, cmap)
        cb = fig.colorbar(diff_plot, ax=axname[index1, index2], fraction=0.046, format='{x:.1%}')
        if colorbar:
            cb.ax.tick_params(labelsize=fontsize)
        else:
            cb.ax.set_visible(False)

def plotrowdiffs(fig, rownum, imagearr, axname, numims, fontsize=18, colorbar=True):
    for jj in range(numims):
        if not rownum == rownum+jj:
            plotdiff(fig, rownum, rownum+jj, imagearr, axname, numims, fontsize=fontsize, colorbar=colorbar)

def diff_matrix(imdict, title=None, fontsize=18, savepath=None, vmax=None,
                im_colorbar=True, diff_colorbar=True):
    titles, imarray = list(imdict.keys()), list(imdict.values())
    n = len(imarray)
    fig, axs = plt.subplots(n, n, figsize=(4*n, 3.2*n))
    for i in range(n):
        for j in range(n):
            axs[i, j].axis('off')
        im = axs[i, i].imshow(np.abs(imarray[i]), cmap='gray', vmax=vmax)
        cb = fig.colorbar(im, ax=axs[i, i], fraction=0.046)
        if im_colorbar:
            cb.ax.tick_params(labelsize=fontsize)
        else:
            cb.ax.set_visible(False)
    for ii in range(n):
        plotrowdiffs(fig, ii, imarray, axs, n, fontsize=fontsize, colorbar=diff_colorbar)
    if title is not None:
        fig.suptitle(title, fontsize=plt.rcParams['figure.titlesize'])

    fig.subplots_adjust(left=0.06, right=0.98, top=0.98, bottom=0.02,
                        wspace=0.32, hspace=0)

    for i in range(n):
        pos = axs[i, i].get_position()
        ycenter = pos.y0 + pos.height / 2
        fig.text(0.02, ycenter, f'{titles[i]}', fontsize=fontsize,
                 rotation=90, va='center', ha='center')

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()

def plot_planes(data, title, savefig=None, inputtype_kspace=True):
    if inputtype_kspace:
        im_grid = sp.ifft(data)
        print('fft done')
    else: 
        im_grid = data
    rss = np.sum(np.abs(im_grid)**2, axis=0)**0.5

    nx, ny, nz = rss.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14)

    for ax, (sl, sl_title) in zip(axes, [
        (rss[nx // 2, :, :], 'x mid-slice'),
        (rss[:, ny // 2, :], 'y mid-slice'),
        (rss[:, :, nz // 2], 'z mid-slice'),
    ]):
        img = ax.imshow(np.abs(sl), cmap='gray')
        ax.set_title(sl_title)
        ax.axis('off')
        fig.colorbar(img, ax=ax, fraction=0.046)

    plt.tight_layout()
    if savefig is not None:
        fig.savefig(savefig, dpi=300, bbox_inches='tight')
    plt.show()



def plotdiff2(fig, index1, index2, imagearr, axname, numims, cmap='RdBu_r',
             diff_vmax=None):
    if index2 > numims-1 or index2 <= index1:
        return None
    diff = np.abs(np.abs(imagearr[index1]) - np.abs(imagearr[index2])) / np.max(np.abs(imagearr[index1]))
    return axname[index1, index2].imshow(diff, cmap, vmin=0, vmax=diff_vmax)

def plotrowdiffs2(fig, rownum, imagearr, axname, numims, diff_vmax=None):
    mappable = None
    for jj in range(numims):
        m = plotdiff2(fig, rownum, rownum+jj, imagearr, axname, numims, diff_vmax=diff_vmax)
        if m is not None:
            mappable = m
    return mappable

def diff_matrix2(imdict, title=None, fontsize=18, savepath=None,
                gray_vmax=None, diff_vmax=None):
    titles, imarray = list(imdict.keys()), list(imdict.values())
    n = len(imarray)

    if diff_vmax is None:
        dmax = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.abs(np.abs(imarray[i]) - np.abs(imarray[j])) / np.max(np.abs(imarray[i]))
                dmax = max(dmax, float(np.nanmax(d)))
        diff_vmax = dmax

    if gray_vmax is None:
        gray_vmax = max(float(np.nanmax(np.abs(im))) for im in imarray)

    fig, axs = plt.subplots(n, n, figsize=(4*n, 3.2*n))
    for i in range(n):
        for j in range(n):
            axs[i, j].axis('off')

    diff_mappable = None
    for ii in range(n):
        m = plotrowdiffs2(fig, ii, imarray, axs, n, diff_vmax=diff_vmax)
        if m is not None:
            diff_mappable = m

    gray_mappable = None
    for i in range(n):
        gray_mappable = axs[i, i].imshow(np.abs(imarray[i]), cmap='gray', vmin=0, vmax=gray_vmax)

    if title is not None:
        fig.suptitle(title, fontsize=plt.rcParams['figure.titlesize'])

    fig.subplots_adjust(left=0.06, right=0.86, top=0.98, bottom=0.02,
                        wspace=0.1, hspace=0.1)



    cbar_bottom = 0.02
    cbar_height = 0.96  # 0.98 - 0.02

    cax_gray = fig.add_axes([0.885, cbar_bottom, 0.015, cbar_height])
    cbar_gray = fig.colorbar(gray_mappable, cax=cax_gray)
    cbar_gray.ax.tick_params(labelsize=fontsize)

    cax_diff = fig.add_axes([0.965, cbar_bottom, 0.015, cbar_height])
    cbar_diff = fig.colorbar(diff_mappable, cax=cax_diff, format='{x:.1%}')
    cbar_diff.ax.tick_params(labelsize=fontsize)

    #cbar_gray.set_label('Magnitude', fontsize=fontsize)
    #cbar_diff.set_label('Difference', fontsize=fontsize)

    x_title = axs[0, 0].get_position().x0 - 0.01
    for i in range(n):
        pos = axs[i, i].get_position()
        ycenter = pos.y0 + pos.height / 2
        fig.text(x_title, ycenter, rf'{titles[i]}', fontsize=fontsize,
                 rotation=90, va='center', ha='center')

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()