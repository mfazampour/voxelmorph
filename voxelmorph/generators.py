import os
import sys
import glob
import numpy as np
import random
from pathlib import Path
import torchio
from torchio.transforms import (
    RescaleIntensity,
    RandomAffine,
    RandomElasticDeformation,
    Compose,
    OneOf,
    CropOrPad,
    Pad,
    RandomFlip,
    Resample,
    Crop
)
import torch

from . import py


def biobank_transform(target_shape=None, min_value=0, max_value=1, target_spacing=None):
    if min_value is None:
        transforms = []
    else:
        rescale = RescaleIntensity((min_value, max_value))
        transforms = [rescale]
    if target_spacing is not None:
        transforms.append(Resample(target_spacing))
    if target_shape is not None:
        transforms.append(CropOrPad(target_shape=target_shape))
    return Compose(transforms)


def load_vol_pathes(patient_list_src: str, source_folder: str, img_pattern, seg_pattern, is_train: True):
    vol_names = []
    seg_names = []
    if is_train:
        patient_list_file = os.path.join(patient_list_src, 'train.txt')
    else:
        patient_list_file = os.path.join(patient_list_src, 'test.txt')
    if not os.path.isfile(patient_list_file):
        create_patient_list(img_pattern, patient_list_src, source_folder)

    # read folder names from file
    file = open(patient_list_file, 'r')
    lines = file.read().splitlines()
    for line in lines:
        vol_names.append(os.path.join(line, img_pattern))
        seg_names.append(os.path.join(line, seg_pattern))
    return vol_names, seg_names


def create_patient_list(img_pattern, patient_list_src, source_folder):
    patient_list = []
    for path in Path(source_folder).rglob(f'*{img_pattern}*'):
        patient_list.append(str(path.parent) + "\n")
    random.shuffle(patient_list)
    index_train = int(len(patient_list) * 0.8)
    file = open(patient_list_src + 'train.txt', 'w')
    file.writelines(patient_list[:index_train])
    file.close()
    file = open(patient_list_src + 'test.txt', 'w')
    file.writelines(patient_list[index_train:])
    file.close()


def volgen_biobank(patient_list_src: str, source_folder: str, is_train=True,
                   img_pattern='T2_FLAIR_unbiased_brain_affine_to_mni.nii.gz',
                   seg_pattern='T1_first_all_fast_firstseg_affine_to_mni.nii.gz',
                   batch_size=1, return_segs=False, np_var='vol',
                   target_shape=None, target_spacing=None, resize_factor=1, add_feat_axis=False):
    # convert glob path to filenames

    assert os.path.isdir(source_folder), f'{source_folder} is not a folder '

    vol_names, seg_names = load_vol_pathes(patient_list_src, source_folder, img_pattern=img_pattern, seg_pattern=seg_pattern, is_train=is_train)
    transform = biobank_transform(target_shape, target_spacing=target_spacing)
    transform_seg = biobank_transform(target_shape, target_spacing=target_spacing, min_value=None)

    while True:
        # generate [batchsize] random image indices
        indices = np.random.randint(len(vol_names), size=batch_size)

        # load volumes and concatenate
        load_params = dict(np_var=np_var, add_batch_axis=True, add_feat_axis=False, pad_shape=None,
                           resize_factor=resize_factor)
        imgs = [np.expand_dims(transform(py.utils.load_volfile(vol_names[i], **load_params)), axis=-1) for i in indices]
        vols = [np.concatenate(imgs, axis=0)]

        # optionally load segmentations and concatenate
        if return_segs:
            segs = [transform_seg(torchio.LabelMap(tensor=py.utils.load_volfile(seg_names[i], **load_params))).data.unsqueeze(dim=-1) for i in indices]
            vols.append(np.concatenate(segs, axis=0))

        yield tuple(vols)


def prostate_transforms(min_size, input_size):
    # Let's use one preprocessing transform and one augmentation transform
    # This transform will be applied only to scalar images:
    rescale = RescaleIntensity((0, 1))
    transforms = [rescale]
    # As RandomAffine is faster then RandomElasticDeformation, we choose to
    # apply RandomAffine 80% of the times and RandomElasticDeformation the rest
    # Also, there is a 25% chance that none of them will be applied
    # if self.opt.isTrain:
    #     spatial = OneOf(
    #         {RandomAffine(translation=5): 0.8, RandomElasticDeformation(): 0.2},
    #         p=0.75,
    #     )
    #     transforms += [RandomFlip(axes=(0, 2), p=0.8), spatial]

    ratio = min_size / np.max(input_size)
    transforms.append(Resample(ratio))
    transforms.append(CropOrPad(input_size))
    transform = Compose(transforms)
    return transform


def read_list_of_patients(root):
    patients = []
    for root, dirs, files in os.walk(root):
        if ('nonrigid' in root) or ('cropped' not in root):
            continue
        patients.append(root)
    return patients


def load_subject_(index, patients, subjects, load_mask=False):
    sample = patients[index % len(patients)]

    # load mr and turs file if it hasn't already been loaded
    if sample not in subjects:
        # print(f'loading patient {sample}')
        if load_mask:
            subject = torchio.Subject(mr=torchio.ScalarImage(sample + "/mr.mhd"),
                                      trus=torchio.ScalarImage(sample + "/trus.mhd"),
                                      mr_tree=torchio.LabelMap(sample + "/mr_tree.mhd"))
        else:
            subject = torchio.Subject(mr=torchio.ScalarImage(sample + "/mr.mhd"),
                                      trus=torchio.Image(sample + "/trus.mhd"))
        subjects[sample] = subject
    subject = subjects[sample]
    return sample, subject


def volgen_prostate(source_folder: str, img_pattern='T1_brain_affine_to_mni', seg_pattern='T1_brain_seg_affine_to_mni',
                    batch_size=1, return_segs=False, np_var='vol', target_shape=None):
    # convert glob path to filenames
    if target_shape is None:
        target_shape = [128, 128, 128]
    assert os.path.isdir(source_folder), f'{source_folder} is not a folder '

    patients = read_list_of_patients(source_folder)
    subjects = {}

    min_size = 64
    transform = prostate_transforms(min_size, target_shape)

    while True:
        # generate [batchsize] random image indices
        indices = np.random.randint(len(patients), size=batch_size)
        imgs = []
        for index in indices:
            sample, subject = load_subject_(index, patients, subjects, load_mask=return_segs)
            transformed_ = transform(subject)
            a = transformed_['mr'].data[:, :target_shape[0], :target_shape[1], :target_shape[2]]
            b = transformed_['trus'].data[:, :target_shape[0], :target_shape[1], :target_shape[2]]

            imgs.append(torch.cat([a, b], dim=0).unsqueeze(dim=0))
        vols = [torch.cat(imgs, dim=0)]

        # # optionally load segmentations and concatenate
        # if return_segs:
        #     segs = [np.expand_dims(transform(py.utils.load_volfile(seg_names[i], **load_params)), axis=-1) for i in
        #             indices]
        #     vols.append(np.concatenate(segs, axis=0))

        yield tuple(vols)


def volgen(
        vol_names,
        batch_size=1,
        return_segs=False,
        np_var='vol',
        pad_shape=None,
        resize_factor=1,
        add_feat_axis=True
):
    """
    Base generator for random volume loading. Volumes can be passed as a path to
    the parent directory, a glob pattern or a list of file paths. Corresponding
    segmentations are additionally loaded if return_segs is set to True. If
    loading segmentations, npz files with variable names 'vol' and 'seg' are
    expected.

    Parameters:
        vol_names: Path, glob pattern or list of volume files to load.
        batch_size: Batch size. Default is 1.
        return_segs: Loads corresponding segmentations. Default is False.
        np_var: Name of the volume variable if loading npz files. Default is 'vol'.
        pad_shape: Zero-pads loaded volumes to a given shape. Default is None.
        resize_factor: Volume resize factor. Default is 1.
        add_feat_axis: Load volume arrays with added feature axis. Default is True.
    """

    # convert glob path to filenames
    if isinstance(vol_names, str):
        if os.path.isdir(vol_names):
            vol_names = os.path.join(vol_names, '*')
        vol_names = glob.glob(vol_names)

    while True:
        # generate [batchsize] random image indices
        indices = np.random.randint(len(vol_names), size=batch_size)

        # load volumes and concatenate
        load_params = dict(np_var=np_var, add_batch_axis=True, add_feat_axis=add_feat_axis, pad_shape=pad_shape,
                           resize_factor=resize_factor)
        imgs = [py.utils.load_volfile(vol_names[i], **load_params) for i in indices]
        vols = [np.concatenate(imgs, axis=0)]

        # optionally load segmentations and concatenate
        if return_segs:
            load_params['np_var'] = 'seg'  # be sure to load seg
            segs = [py.utils.load_volfile(vol_names[i], **load_params) for i in indices]
            vols.append(np.concatenate(segs, axis=0))

        yield tuple(vols)


def scan_to_scan(vol_names, bidir=False, batch_size=1, prob_same=0, no_warp=False,
                 return_segs=False, loader_name='default', **kwargs):
    """
    Generator for scan-to-scan registration.

    Parameters:
        loader_name: Modified generator for using specific volume generator instead of the default one
        vol_names: List of volume files to load.
        bidir: Yield input image as output for bidirectional models. Default is False.
        batch_size: Batch size. Default is 1. TODO: as of now it seems only batch_size=1 is supported.
        prob_same: Induced probability that source and target inputs are the same. Default is 0.
        no_warp: Excludes null warp in output list if set to True (for affine training). Default if False.
        kwargs: Forwarded to the internal volgen generator.
    """
    zeros = None

    if loader_name == 'biobank':
        gen = volgen_biobank(source_folder=vol_names, batch_size=batch_size, return_segs=return_segs, **kwargs)
    elif loader_name == 'prostate':
        gen = volgen_prostate(source_folder=vol_names)
    else:
        gen = volgen(vol_names, batch_size=batch_size, **kwargs)

    while True:
        seg1 = None
        seg2 = None

        if loader_name == 'prostate':
            scan = next(gen)[0]
            scan1 = scan[:, 0:1, ...]
            scan2 = scan[:, 1:2, ...]

            # cache zeros
            if not no_warp and zeros is None:
                shape = scan1.shape[1:-1]
                zeros = torch.zeros((batch_size, *shape, len(shape)), dtype=scan1.dtype)
        else:
            data1 = next(gen)
            data2 = next(gen)

            scan1 = data1[0]
            scan2 = data2[0]

            if return_segs:
                seg1 = data1[1]
                seg2 = data2[1]

            # some induced chance of making source and target equal
            if prob_same > 0 and np.random.rand() < prob_same:
                if np.random.rand() > 0.5:
                    scan1 = scan2
                else:
                    scan2 = scan1

            # cache zeros
            if not no_warp and zeros is None:
                shape = scan1.shape[1:-1]
                zeros = np.zeros((batch_size, *shape, len(shape)))

        invols = [scan1, scan2]
        outvols = [scan2, scan1] if bidir else [scan2]
        if not no_warp:
            outvols.append(zeros)

        if seg1 is not None:
            invols.append(seg1)
            outvols.append(seg2)

        yield (invols, outvols)


def scan_to_atlas(vol_names, atlas, bidir=False, batch_size=1, no_warp=False, **kwargs):
    """
    Generator for scan-to-atlas registration.

    TODO: This could be merged into scan_to_scan() by adding an optional atlas
    argument like in semisupervised().

    Parameters:
        vol_names: List of volume files to load.
        atlas: Atlas volume data.
        bidir: Yield input image as output for bidirectional models. Default is False.
        batch_size: Batch size. Default is 1.
        no_warp: Excludes null warp in output list if set to True (for affine training). Default if False.
        kwargs: Forwarded to the internal volgen generator.
    """
    shape = atlas.shape[1:-1]
    zeros = np.zeros((batch_size, *shape, len(shape)))
    atlas = np.repeat(atlas, batch_size, axis=0)
    gen = volgen(vol_names, batch_size=batch_size, **kwargs)
    while True:
        scan = next(gen)[0]
        invols = [scan, atlas]
        outvols = [atlas, scan] if bidir else [atlas]
        if not no_warp:
            outvols.append(zeros)
        yield (invols, outvols)


def semisupervised(vol_names, labels, atlas_file=None, downsize=2):
    """
    Generator for semi-supervised registration training using ground truth segmentations.
    Scan-to-atlas training can be enabled by providing the atlas_file argument. It's
    expected that vol_names and atlas_file are npz files with both 'vol' and 'seg' arrays.

    Parameters:
        vol_names: List of volume npz files to load.
        labels: Array of discrete label values to use in training.
        atlas_file: Atlas npz file for scan-to-atlas training. Default is None.
        downsize: Downsize factor for segmentations. Default is 2.
    """
    # configure base generator
    gen = volgen(vol_names, return_segs=True, np_var='vol')
    zeros = None

    # internal utility to generate downsampled prob seg from discrete seg
    def split_seg(seg):
        prob_seg = np.zeros((*seg.shape[:4], len(labels)))
        for i, label in enumerate(labels):
            prob_seg[0, ..., i] = seg[0, ..., 0] == label
        return prob_seg[:, ::downsize, ::downsize, ::downsize, :]

    # cache target vols and segs if atlas is supplied
    if atlas_file:
        trg_vol = py.utils.load_volfile(atlas_file, np_var='vol', add_batch_axis=True, add_feat_axis=True)
        trg_seg = py.utils.load_volfile(atlas_file, np_var='seg', add_batch_axis=True, add_feat_axis=True)
        trg_seg = split_seg(trg_seg)

    while True:
        # load source vol and seg
        src_vol, src_seg = next(gen)
        src_seg = split_seg(src_seg)

        # load target vol and seg (if not provided by atlas)
        if not atlas_file:
            trg_vol, trg_seg = next(gen)
            trg_seg = split_seg(trg_seg)

        # cache zeros
        if zeros is None:
            shape = src_vol.shape[1:-1]
            zeros = np.zeros((1, *shape, len(shape)))

        invols = [src_vol, trg_vol, src_seg]
        outvols = [trg_vol, zeros, trg_seg]
        yield (invols, outvols)


def template_creation(vol_names, atlas, bidir=False, batch_size=1, **kwargs):
    """
    Generator for unconditional template creation.

    Parameters:
        vol_names: List of volume files to load.
        atlas: Atlas input volume data.
        bidir: Yield input image as output for bidirectional models. Default is False.
        batch_size: Batch size. Default is 1.
        kwargs: Forwarded to the internal volgen generator.
    """
    shape = atlas.shape[1:-1]
    zeros = np.zeros((batch_size, *shape, len(shape)))
    atlas = np.repeat(atlas, batch_size, axis=0)
    gen = volgen(vol_names, batch_size=batch_size, **kwargs)
    while True:
        scan = next(gen)[0]
        invols = [atlas, scan]  # TODO: this is opposite of the normal ordering and might be confusing
        outvols = [scan, atlas, zeros, zeros] if bidir else [scan, zeros, zeros]
        yield (invols, outvols)


def conditional_template_creation(vol_names, atlas, attributes, batch_size=1, np_var='vol', pad_shape=None,
                                  add_feat_axis=True):
    """
    Generator for conditional template creation.

    Parameters:
        vol_names: List of volume files to load.
        atlas: Atlas input volume data.
        attributes: Dictionary of phenotype data for each vol name.
        batch_size: Batch size. Default is 1.
        np_var: Name of the volume variable if loading npz files. Default is 'vol'.
        pad_shape: Zero-pads loaded volumes to a given shape. Default is None.
        add_feat_axis: Load volume arrays with added feature axis. Default is True.
    """
    shape = atlas.shape[1:-1]
    zeros = np.zeros((batch_size, *shape, len(shape)))
    atlas = np.repeat(atlas, batch_size, axis=0)
    while True:
        indices = np.random.randint(len(vol_names), size=batch_size)

        # load pheno from attributes dictionary
        pheno = np.stack([attributes[vol_names[i]] for i in indices], axis=0)

        # load volumes and concatenate
        load_params = dict(np_var=np_var, add_batch_axis=True, add_feat_axis=add_feat_axis, pad_shape=pad_shape)
        vols = [py.utils.load_volfile(vol_names[i], **load_params) for i in indices]
        vols = np.concatenate(vols, axis=0)

        invols = [pheno, atlas, vols]
        outvols = [vols, zeros, zeros, zeros]
        yield (invols, outvols)


def surf_semisupervised(
        vol_names,
        atlas_vol,
        atlas_seg,
        nb_surface_pts,
        labels=None,
        batch_size=1,
        surf_bidir=True,
        surface_pts_upsample_factor=2,
        smooth_seg_std=1,
        nb_labels_sample=None,
        sdt_vol_resize=1,
        align_segs=False,
        add_feat_axis=True
):
    """
    Scan-to-atlas generator for semi-supervised learning using surface point clouds from segmentations.

    Parameters:
        vol_names: List of volume files to load.
        atlas_vol: Atlas volume array.
        atlas_seg: Atlas segmentation array.
        nb_surface_pts: Total number surface points for all structures.
        labels: Label list to include. If None, all labels in atlas_seg are used. Default is None.
        batch_size: Batch size. NOTE some features only implemented for 1. Default is 1.
        surf_bidir: Train with bidirectional surface distance. Default is True.
        surface_pts_upsample_factor: Upsample factor for surface pointcloud. Default is 2.
        smooth_seg_std: Segmentation smoothness sigma. Default is 1.
        nb_labels_sample: Number of labels to sample. Default is None.
        sdt_vol_resize: Resize factor for signed distance transform volumes. Default is 1.
        align_segs: Whether to pass in segmentation image instead. Default is False.
        add_feat_axis: Load volume arrays with added feature axis. Default is True.
    """

    # some input checks
    assert nb_surface_pts > 0, 'number of surface point should be greater than 0'

    # prepare some shapes
    vol_shape = atlas_seg.shape
    sdt_shape = [int(f * sdt_vol_resize) for f in vol_shape]

    # compute labels from atlas, and the number of labels to sample.
    if labels is not None:
        atlas_seg = py.utils.filter_labels(atlas_seg, labels)
    else:
        labels = np.sort(np.unique(atlas_seg))[1:]

    # use all labels by default
    if nb_labels_sample is None:
        nb_labels_sample = len(labels)

    # prepare keras format atlases
    atlas_vol_bs = np.repeat(atlas_vol[np.newaxis, ..., np.newaxis], batch_size, axis=0)
    atlas_seg_bs = np.repeat(atlas_seg[np.newaxis, ..., np.newaxis], batch_size, axis=0)

    # prepare surface extraction function
    std_to_surf = lambda x, y: py.utils.sdt_to_surface_pts(x, y,
                                                           surface_pts_upsample_factor=surface_pts_upsample_factor,
                                                           thr=(1 / surface_pts_upsample_factor + 1e-5))

    # prepare zeros, which will be used for outputs unused in cost functions
    zero_flow = np.zeros((batch_size, *vol_shape, len(vol_shape)))
    zero_surface_values = np.zeros((batch_size, nb_surface_pts, 1))

    # precompute label edge volumes
    atlas_sdt = [None] * len(labels)
    atlas_label_vols = [None] * len(labels)
    nb_edges = np.zeros(len(labels))
    for li, label in enumerate(labels):  # if only one label, get surface points here
        atlas_label_vols[li] = atlas_seg == label
        atlas_label_vols[li] = py.utils.clean_seg(atlas_label_vols[li], smooth_seg_std)
        atlas_sdt[li] = py.utils.vol_to_sdt(atlas_label_vols[li], sdt=True, sdt_vol_resize=sdt_vol_resize)
        nb_edges[li] = np.sum(np.abs(atlas_sdt[li]) < 1.01)
    layer_edge_ratios = nb_edges / np.sum(nb_edges)

    # if working with all the labels passed in (i.e. no label sampling per batch), 
    # pre-compute the atlas surface points
    atlas_surface_pts = np.zeros((batch_size, nb_surface_pts, len(vol_shape) + 1))
    if nb_labels_sample == len(labels):
        nb_surface_pts_sel = py.utils.get_surface_pts_per_label(nb_surface_pts, layer_edge_ratios)
        for li, label in enumerate(labels):  # if only one label, get surface points here
            atlas_surface_pts_ = std_to_surf(atlas_sdt[li], nb_surface_pts_sel[li])[np.newaxis, ...]
            # get the surface point stack indexes for this element
            srf_idx = slice(int(np.sum(nb_surface_pts_sel[:li])), int(np.sum(nb_surface_pts_sel[:li + 1])))
            atlas_surface_pts[:, srf_idx, :-1] = np.repeat(atlas_surface_pts_, batch_size, 0)
            atlas_surface_pts[:, srf_idx, -1] = li

    # generator
    gen = volgen(vol_names, return_segs=True, batch_size=batch_size, add_feat_axis=add_feat_axis)

    assert batch_size == 1, 'only batch size 1 supported for now'

    while True:

        # prepare data
        X = next(gen)
        X_img = X[0]
        X_seg = py.utils.filter_labels(X[1], labels)

        # get random labels
        sel_label_idxs = range(len(labels))  # all labels
        if nb_labels_sample != len(labels):
            sel_label_idxs = np.sort(np.random.choice(range(len(labels)), size=nb_labels_sample, replace=False))
            sel_layer_edge_ratios = [layer_edge_ratios[li] for li in sel_label_idxs]
            nb_surface_pts_sel = py.utils.get_surface_pts_per_label(nb_surface_pts, sel_layer_edge_ratios)

        # prepare signed distance transforms and surface point arrays
        X_sdt_k = np.zeros((batch_size, *sdt_shape, nb_labels_sample))
        atl_dt_k = np.zeros((batch_size, *sdt_shape, nb_labels_sample))
        subj_surface_pts = np.zeros((batch_size, nb_surface_pts, len(vol_shape) + 1))
        if nb_labels_sample != len(labels):
            atlas_surface_pts = np.zeros((batch_size, nb_surface_pts, len(vol_shape) + 1))

        for li, sli in enumerate(sel_label_idxs):
            # get the surface point stack indexes for this element
            srf_idx = slice(int(np.sum(nb_surface_pts_sel[:li])), int(np.sum(nb_surface_pts_sel[:li + 1])))

            # get atlas surface points for this label
            if nb_labels_sample != len(labels):
                atlas_surface_pts_ = std_to_surf(atlas_sdt[sli], nb_surface_pts_sel[li])[np.newaxis, ...]
                atlas_surface_pts[:, srf_idx, :-1] = np.repeat(atlas_surface_pts_, batch_size, 0)
                atlas_surface_pts[:, srf_idx, -1] = sli

            # compute X distance from surface
            X_label = X_seg == labels[sli]
            X_label = py.utils.clean_seg_batch(X_label, smooth_seg_std)
            X_sdt_k[..., li] = py.utils.vol_to_sdt_batch(X_label, sdt=True, sdt_vol_resize=sdt_vol_resize)[..., 0]

            if surf_bidir:
                atl_dt = atlas_sdt[li][np.newaxis, ...]
                atl_dt_k[..., li] = np.repeat(atl_dt, batch_size, 0)
                ssp_lst = [std_to_surf(f[...], nb_surface_pts_sel[li]) for f in X_sdt_k[..., li]]
                subj_surface_pts[:, srf_idx, :-1] = np.stack(ssp_lst, 0)
                subj_surface_pts[:, srf_idx, -1] = li

        # check if returning segmentations instead of images
        # this is a bit hacky for basically building a segmentation-only network (no images)
        X_ret = X_img
        atlas_ret = atlas_vol_bs

        if align_segs:
            assert len(labels) == 1, 'align_seg generator is only implemented for single label'
            X_ret = X_seg == labels[0]
            atlas_ret = atlas_seg_bs == labels[0]

        # finally, output
        if surf_bidir:
            inputs = [X_ret, atlas_ret, X_sdt_k, atl_dt_k, subj_surface_pts, atlas_surface_pts]
            outputs = [atlas_ret, X_ret, zero_flow, zero_surface_values, zero_surface_values]
        else:
            inputs = [X_ret, atlas_ret, X_sdt_k, atlas_surface_pts]
            outputs = [atlas_ret, X_ret, zero_flow, zero_surface_values]

        yield (inputs, outputs)
