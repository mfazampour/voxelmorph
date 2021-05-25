from typing import Dict, List

import numpy as np
import torch
from monai.metrics import compute_meandice
from monai.metrics import compute_hausdorff_distance
from monai.metrics import compute_average_surface_distance
import torchio
from torchio.transforms import Resample
import SimpleITK as sitk


import matplotlib.pyplot as plt

from pylab import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes


def apply_model(model: torch.nn.Module, generator=None, inputs=None, y_true=None, device='cpu', losses=None,
                weights=None, is_test=False, has_seg=False):
    # generate inputs (and true outputs) and convert them to tensors
    assert generator is not None or (
            inputs is not None and y_true is not None), 'Either generator or input/y_true needed'
    if generator is not None:
        inputs, y_true = next(generator)
    if not isinstance(inputs[0], torch.Tensor):
        inputs = [torch.from_numpy(d).float().permute(0, 4, 1, 2, 3) for d in inputs]
        y_true = [torch.from_numpy(d).float().permute(0, 4, 1, 2, 3) for d in y_true]
    inputs = [t.to(device) for t in inputs]
    y_true = [t.to(device) for t in y_true]
    # run inputs through the model to produce a warped image and flow field
    if has_seg:
        y_pred = model(*inputs[:-1], registration=is_test)
    else:
        y_pred = model(*inputs, registration=is_test)

    if losses is None:
        return inputs, y_true, y_pred
    # calculate total loss
    loss = torch.tensor([0], dtype=torch.float).to(device)
    loss_list = []
    for n, loss_function in enumerate(losses):
        curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
        loss_list.append('%.6f' % curr_loss.item())
        loss += curr_loss
    return loss, loss_list, inputs, y_true, y_pred


def resize_to_1mm(img: torch.Tensor, affine):
    T = Resample(1.0)
    img_im = torchio.LabelMap(tensor=img.cpu().squeeze(0), affine=affine)
    return T(img_im).data.to(img.device).unsqueeze(dim=0)


def get_scores(device, mask_values, model: torch.nn.Module, transformer: torch.nn.Module,
               inputs, y_true, affine=None, resize_module: torch.nn.Module = None, spacing=None):
    inputs, y_true, y_pred = apply_model(model, inputs=inputs,
                                         y_true=y_true, device=device,
                                         is_test=True, has_seg=True)
    seg_fixed = y_true[-1].clone()
    seg_moving = inputs[-1].clone()
    dvf = y_pred[-1].detach()
    seg_morphed = transformer(seg_moving, dvf)
    if spacing is not None:
        asd_score_dg, dice_score_dg = calc_asd(seg_fixed, seg_moving, mask_values, spacing)
    if affine is not None:
        seg_morphed = resize_to_1mm(seg_morphed, affine)
        seg_fixed = resize_to_1mm(seg_fixed, affine)
        seg_moving = resize_to_1mm(seg_moving, affine)
    if spacing is None:  # calculate after resampling to 1mm isometric
        asd_score_dg, dice_score_dg = calc_asd(seg_fixed, seg_moving, mask_values, np.array([1, 1, 1]))

    if resize_module is not None:
        dvf = resize_module(dvf)
    seg_morphed = seg_morphed.round()
    shape = list(seg_fixed.shape)
    shape[1] = len(mask_values)
    one_hot_fixed = torch.zeros(shape, device=device)
    one_hot_moving = torch.zeros(shape, device=device)
    one_hot_morphed = torch.zeros(shape, device=device)
    for i, (val) in enumerate(mask_values):
        one_hot_fixed[:, i, seg_fixed[0, 0, ...] == val] = 1
        one_hot_moving[:, i, seg_moving[0, 0, ...] == val] = 1
        one_hot_morphed[:, i, seg_morphed[0, 0, ...] == val] = 1
        seg_fixed[:, 0, seg_fixed[0, 0, ...] == val] = i
        seg_morphed[:, 0, seg_morphed[0, 0, ...] == val] = i
        seg_moving[:, 0, seg_moving[0, 0, ...] == val] = i
    dice_score = compute_meandice(one_hot_fixed, one_hot_morphed, to_onehot_y=False)
    hd_score = torch.zeros_like(dice_score)
    asd_score = torch.zeros_like(dice_score)
    for i in range(len(mask_values)):
        hd_score[0, i] = compute_hausdorff_distance(seg_morphed, seg_fixed, i)
        asd_score[0, i] = compute_average_surface_distance(seg_morphed, seg_fixed, i)
    seg_maps = (seg_fixed.cpu(), seg_moving.cpu(), seg_morphed.cpu())
    return asd_score_dg, dice_score_dg, hd_score, seg_maps, dvf


def calc_scores(device, mask_values, model: torch.nn.Module, transformer: torch.nn.Module,
                test_generator=None, inputs=None, y_true=None,
                num_statistics_runs=10, calc_statistics=False, affine=None,
                resize_module: torch.nn.Module = None, keep_dvfs=True, spacing=None):
    reps = 1
    if calc_statistics:
        reps = num_statistics_runs
    dice_scores = []
    hd_scores = []
    asd_scores = []
    seg_maps = []
    dvfs = []

    with torch.no_grad():
        if test_generator is not None:
            inputs, y_true = next(test_generator)
        for n in range(reps):
            if reps > 50:
                print(f'---------- getting sample number: {n+1}/{reps} ------------')
            asd_score, dice_score, hd_score, seg_map, dvf = get_scores(device, mask_values, model, transformer,
                                                                       inputs=inputs, y_true=y_true,
                                                                       affine=affine, resize_module=resize_module,
                                                                       spacing=spacing)
            seg_maps.append(seg_map)
            if keep_dvfs:
                dvfs.append(dvf)
            dice_scores.append(dice_score)
            hd_scores.append(hd_score)
            asd_scores.append(asd_score)
        # calculate mean and return if no calc_statistics
        dice_scores = torch.cat(dice_scores)
        hd_scores = torch.cat(hd_scores)
        asd_scores = torch.cat(asd_scores)
        if calc_statistics:
            dice_std = dice_scores.std(dim=0, keepdim=True)
            hd_std = hd_scores.std(dim=0, keepdim=True)
            asd_std = asd_scores.std(dim=0, keepdim=True)
        else:
            dice_std, hd_std, asd_std = (torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]))
        return dice_scores, hd_scores, asd_scores, dice_std, hd_std, asd_std, seg_maps, dvfs


def create_toy_sample(img: torch.Tensor, mask: torch.Tensor, method: str = 'noise', num_changes=1, fill=0, sigma=1):
    toy = img.clone()
    num_regions = len(mask.unique())
    #  exclude background from changing
    for i in range(num_changes):
        region = np.random.randint(low=1, high=num_regions, size=1)
        im_indices = mask == mask.unique()[region]
        if method == 'swap':
            swap_indices(toy, indices=im_indices)
        elif method == 'constant':
            toy[im_indices] = fill
        elif method == 'noise':
            toy[im_indices] = torch.rand(img.shape)[im_indices] * sigma + fill
        else:
            NotImplementedError()
    return toy


def swap_indices(img: torch.Tensor, indices):
    return img


def calc_asd(seg_fixed, seg_moving, mask_values, spacing):
    """
    calculate average surface distances and Dice scores
    """

    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

    ASD = torch.zeros(len(mask_values), device=seg_fixed.device)
    DSC = torch.zeros(len(mask_values), device=seg_fixed.device)

    seg_fixed_arr = seg_fixed.squeeze().cpu().numpy()
    seg_moving_arr = seg_moving.squeeze().cpu().numpy()
    spacing = spacing.tolist()

    def calc_ASD(seg_fixed_im, seg_moving_im):
        seg_fixed_contour = sitk.LabelContour(seg_fixed_im)
        seg_moving_contour = sitk.LabelContour(seg_moving_im)

        hausdorff_distance_filter.Execute(seg_fixed_contour, seg_moving_contour)
        return hausdorff_distance_filter.GetAverageHausdorffDistance()

    def calc_DSC(seg_fixed_im, seg_moving_im):
        overlap_measures_filter.Execute(seg_fixed_im, seg_moving_im)
        return overlap_measures_filter.GetDiceCoefficient()

    for idx, val in enumerate(mask_values):

        seg_fixed_structure = np.where(seg_fixed_arr == val, 1, 0)
        seg_moving_structure = np.where(seg_moving_arr == val, 1, 0)

        seg_fixed_im = sitk.GetImageFromArray(seg_fixed_structure)
        seg_moving_im = sitk.GetImageFromArray(seg_moving_structure)

        seg_fixed_im.SetSpacing(spacing)
        seg_moving_im.SetSpacing(spacing)

        try:
            ASD[idx] = calc_ASD(seg_fixed_im, seg_moving_im)
        except:
            ASD[idx] = np.inf

        DSC[idx] = calc_DSC(seg_fixed_im, seg_moving_im)

    return ASD, DSC
