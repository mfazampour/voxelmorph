from typing import List

import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec


def set_axs_attribute(axs):
    for ax in list(axs.flatten()):
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)


def fill_subplots(img: torch.Tensor, axs, img_name='', fontsize=6, cmap='gray',
                  fig: plt.Figure=None, show_colorbar=False):
    if cmap == 'gray':  # map image to 0...1
        img = (img + img.min())/(img.max() - img.min())
    elif cmap is None:  # cliping data to 0...255
        img[img < 0] = 0
        img[img > 255] = 255

    shape = img.shape[-3:]
    img1 = img[0, :, int(shape[0] / 2), :, :].permute(dims=(1, 2, 0)).squeeze().numpy()
    axs[0].imshow(img1, cmap=cmap)
    axs[0].set_title(f'{img_name} central slice \n in sagittal view', fontsize=fontsize)
    img1 = img[0, :, :, int(shape[1] / 2), :].permute(dims=(1, 2, 0)).squeeze().numpy()
    axs[1].imshow(img1, cmap=cmap)
    axs[1].set_title(f'{img_name} central slice \n in coronal view', fontsize=fontsize)
    img2 = img[0, :, :, :, int(shape[0] / 2)].permute(dims=(1, 2, 0)).squeeze().numpy()
    axs[2].imshow(img2, cmap=cmap)
    axs[2].set_title(f'{img_name} central slice \n in axial view', fontsize=fontsize)
    if show_colorbar and fig is not None:
        fig.colorbar(img1, ax=axs[0])
        fig.colorbar(img1, ax=axs[1])
        fig.colorbar(img2, ax=axs[2])

def create_figure(fixed: torch.Tensor, moving: torch.Tensor, registered: torch.Tensor,
                  deformation: torch.Tensor, logSimga: None):
    if logSimga is None:
        nrow = 6
    else:
        nrow = 7
    ncol = 3
    axs, fig = init_figure(ncol, nrow)

    # fig, axs = plt.subplots(5, 3)
    set_axs_attribute(axs)
    fill_subplots(fixed, axs=axs[0, :], img_name='Fixed')
    fill_subplots(moving, axs=axs[1, :], img_name='Moving')
    fill_subplots(fixed - moving, axs=axs[2, :], img_name='Fix-Mov')
    fill_subplots(registered, axs=axs[3, :], img_name='Registered')
    fill_subplots(fixed - registered, axs=axs[4, :], img_name='Fix-Reg')
    deform_ = (deformation + 5) / 10
    fill_subplots(deform_, axs=axs[5, :], img_name='Def.', cmap=None)
    if logSimga:
        fill_subplots(logSimga, axs=axs[6: 0], img_name='LogSigma', cmap=None, fig=fig, show_colorbar=True)
    return fig


def init_figure(ncol, nrow) -> (List[plt.Axes], plt.Figure):
    fig = plt.figure(figsize=(2 * ncol + 1, 2 * nrow + 1))  # , constrained_layout=True)
    spec = gridspec.GridSpec(nrow, ncol, figure=fig,
                             wspace=0.2, hspace=0.2,
                             top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                             left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
    # spec = fig.add_gridspec(ncols=3, nrows=5, width_ratios=[0.5]*3, height_ratios=[1]*5)
    # spec.update(wspace=0.025, hspace=0.05)
    axs = []
    for i in range(nrow):
        tmp = []
        for j in range(ncol):
            ax = fig.add_subplot(spec[i, j])
            tmp.append(ax)
        axs.append(tmp)
    axs = np.asarray(axs)
    return axs, fig
