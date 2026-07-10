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


def plot_mask(isolated_kspace, mask, zte_radial_coords, zte_radial_kspace, innersidelen, sidelencart, sidelenrad):
    
    all_coords = zte_radial_coords.reshape(-1, 2)

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


    yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[0]]
    im1 = ax[1].scatter(xx.ravel(), yy.ravel(), c=mask.ravel().astype(int), cmap='viridis', s=300, marker='s', zorder=3)
    fig.colorbar(im1, ax=ax[1], label='mask')
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


def plotdiff(fig, index1, index2, imagearr, axname, numims, cmap='RdBu_r'):
    if not index2>numims-1:
        diff = (np.abs(np.abs(imagearr[index1]) - np.abs(imagearr[index2])))/ np.max(np.abs(imagearr[index1]))
        diff_plot = axname[index1, index2].imshow(diff, cmap)
        fig.colorbar(diff_plot, ax=axname[index1, index2], fraction=0.046, format='{x:.1%}')
    
def plotrowdiffs(fig, rownum, imagearr, axname, numims):
    for jj in range(numims):
        if not rownum == rownum+jj:
            plotdiff(fig,rownum, rownum+jj,imagearr, axname, numims)

def diff_matrix(imdict):
    titles, imarray = list(imdict.keys()), list(imdict.values())
    n = len(imarray)
    n = len(imarray)
    fig, axs = plt.subplots(n, n, figsize=(4*n, 4*n))
    for i in range(n):
        for j in range(n):
            axs[i, j].axis('off')
        im = axs[i,i].imshow(np.abs(imarray[i]), cmap='gray')
        axs[i, i].set_title(f'{titles[i]}')
        fig.colorbar(im,ax=axs[i,i], fraction=0.046)
    for ii in range(n):
        plotrowdiffs(fig, ii, imarray, axs, n)

    plt.tight_layout()
    plt.show()