import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec


def set_axs_attribute(axs):
    for ax in list(axs.flatten()):
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)


def fill_subplots(img: torch.Tensor, axs, img_name='', fontsize=6, cmap='gray'):
    shape = img.shape[-3:]
    axs[0].imshow(img[0, :, int(shape[0] / 2), :, :].permute(dims=(1, 2, 0)).squeeze().numpy(), cmap=cmap)
    axs[0].set_title(f'{img_name} central slice \n in sagittal view', fontsize=fontsize)
    axs[1].imshow(img[0, :, :, int(shape[1] / 2), :].permute(dims=(1, 2, 0)).squeeze().numpy(), cmap=cmap)
    axs[1].set_title(f'{img_name} central slice \n in coronal view', fontsize=fontsize)
    axs[2].imshow(img[0, :, :, :, int(shape[0] / 2)].permute(dims=(1, 2, 0)).squeeze().numpy(), cmap=cmap)
    axs[2].set_title(f'{img_name} central slice \n in axial view', fontsize=fontsize)


def create_figure(fixed: torch.Tensor, moving: torch.Tensor, registered: torch.Tensor, deformation: torch.Tensor):
    nrow = 6
    ncol = 3
    fig = plt.figure(figsize=(ncol + 1, nrow + 1))  # , constrained_layout=True)
    spec = gridspec.GridSpec(nrow, ncol, figure=fig,
                             wspace=0.1, hspace=0.5,
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

    # fig, axs = plt.subplots(5, 3)
    set_axs_attribute(axs)
    fill_subplots(fixed, axs=axs[0, :], img_name='Fixed')
    fill_subplots(moving, axs=axs[1, :], img_name='Moving')
    fill_subplots(fixed - moving, axs=axs[2, :], img_name='Fix-Mov')
    fill_subplots(registered, axs=axs[3, :], img_name='Registered')
    fill_subplots(fixed - registered, axs=axs[4, :], img_name='Fix-Reg')
    fill_subplots((deformation + 10) / 20, axs=axs[5, :], img_name='Def.', cmap=None)
    return fig
