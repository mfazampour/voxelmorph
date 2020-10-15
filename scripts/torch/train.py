#!/usr/bin/env python

"""
Example script to train a VoxelMorph model.

For the CVPR and MICCAI papers, we have data arranged in train, validate, and test folders. Inside each folder
are normalized T1 volumes and segmentations in npz (numpy) format. You will have to customize this script slightly
to accommodate your own data. All images should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed. Otherwise,
registration will be scan-to-scan.
"""

import os
import random
import argparse
import glob
import time
from datetime import datetime

import numpy as np
import torch
# import napari
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm


def fill_subplots(img: torch.Tensor, axs, img_name=''):
    shape = img.shape[-3:]
    axs[0].imshow(img[0, 0, int(shape[0] / 2), :, :].squeeze().numpy(), cmap='gray')
    axs[0].set_title = f'{img_name} central slice in sagittal view'
    axs[1].imshow(img[0, 0, :, int(shape[1] / 2), :].squeeze().numpy(), cmap='gray')
    axs[1].set_title = f'{img_name} central slice in coronal view'
    axs[2].imshow(img[0, 0, :, :, int(shape[0] / 2)].squeeze().numpy(), cmap='gray')
    axs[2].set_title = f'{img_name} central slice in axial view'


def create_figure(fixed: torch.Tensor, moving: torch.Tensor, registered: torch.Tensor):
    fig, axs = plt.subplots(5, 3)
    fill_subplots(fixed, axs=axs[0, :], img_name='fixed')
    fill_subplots(moving, axs=axs[1, :], img_name='moving')
    fill_subplots(fixed - moving, axs=axs[2, :], img_name='diff of moving and fixed')
    fill_subplots(registered, axs=axs[3, :], img_name='registered')
    fill_subplots(fixed - registered, axs=axs[4, :], img_name='diff of registered and fixed ')
    return fig


# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('datadir', help='base data directory')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--model-dir', default='models', help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true', help='specify that data has multiple channels')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1500, help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=100, help='frequency of model saves (default: 100)')
parser.add_argument('--load-model', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0, help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true', help='disable cudnn determinism - might slow down training')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+', help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7, help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2, help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

# loss hyperparameters
parser.add_argument('--image-loss', default='mse', help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                    help='weight of deformation loss (default: 0.01)')
parser.add_argument('--use-biobank', action='store_true', help='set to True if using biobank data')
parser.add_argument('--inshape', type=int, nargs='+',
                    help='after cropping shape of input. '
                         'default is equal to image size. specify if the input can\'t path through UNet')
parser.add_argument('--log-dir', type=str, default=None, help='folder for tensorboard logs')

args = parser.parse_args()

bidir = args.bidir

if args.log_dir is None:
    args.log_dir = args.model_dir

writer = SummaryWriter(log_dir=f'{args.log_dir}/{datetime.now().strftime("%m.%d.%Y_%H.%M")}')

if args.use_biobank:
    train_vol_names = args.datadir
else:
    # load and prepare training data
    train_vol_names = glob.glob(os.path.join(args.datadir, '*.npz'))
    random.shuffle(train_vol_names)  # shuffle volume list
    assert len(train_vol_names) > 0, 'Could not find any training data'

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel

if 'inshape' not in args:
    args.inshape = None

if args.atlas:
    # scan-to-atlas generator
    atlas = vxm.py.utils.load_volfile(args.atlas, np_var='vol', add_batch_axis=True, add_feat_axis=add_feat_axis)
    generator = vxm.generators.scan_to_atlas(train_vol_names, atlas, batch_size=args.batch_size, bidir=args.bidir,
                                             add_feat_axis=add_feat_axis)
else:
    # scan-to-scan generator
    generator = vxm.generators.scan_to_scan(train_vol_names, batch_size=args.batch_size,
                                            bidir=args.bidir, add_feat_axis=add_feat_axis,
                                            use_biobank=args.use_biobank, target_shape=args.inshape)

# extract shape from sampled input
inshape = next(generator)[0][0].shape[1:-1]

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
assert args.batch_size >= nb_gpus, 'Batch size (%d) should be no less than the number of gpus (%d)' % (
args.batch_size, nb_gpus)

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

if args.load_model:
    # load initial model (if specified)
    model = vxm.networks.VxmDense.load(args.load_model, device)
else:
    # otherwise configure new model
    model = vxm.networks.VxmDense(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=bidir,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize
    )

if nb_gpus > 1:
    # use multiple GPUs via DataParallel
    model = torch.nn.DataParallel(model)
    model.save = model.module.save

# prepare the model for training and send to device
model.to(device)
model.train()

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = vxm.losses.NCC().loss
elif args.image_loss == 'mse':
    image_loss_func = vxm.losses.MSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

# need two image loss functions if bidirectional
if bidir:
    losses = [image_loss_func, image_loss_func]
    weights = [0.5, 0.5]
else:
    losses = [image_loss_func]
    weights = [1]

# prepare deformation loss
losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
weights += [args.weight]

# training loops
for epoch in range(args.initial_epoch, args.epochs):

    # save model checkpoint
    model.save(os.path.join(model_dir, '%04d.pt' % epoch))

    for step in range(args.steps_per_epoch):

        step_start_time = time.time()

        # generate inputs (and true outputs) and convert them to tensors
        inputs, y_true = next(generator)
        inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
        y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]
        # if should_crop:
        #     inputs = [t[:, :, :inshape[0], :inshape[1], :inshape[2]] for t in inputs]
        #     y_true = [t[:, :, :inshape[0], :inshape[1], :inshape[2]] for t in y_true]

        # run inputs through the model to produce a warped image and flow field
        y_pred = model(*inputs)

        # calculate total loss
        loss = 0
        loss_list = []
        for n, loss_function in enumerate(losses):
            curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
            loss_list.append('%.6f' % curr_loss.item())
            loss += curr_loss

        loss_info = 'loss: %.6f  (%s)' % (loss.item(), ', '.join(loss_list))

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print step info
        epoch_info = 'epoch: %04d' % (epoch + 1)
        step_info = ('step: %d/%d' % (step + 1, args.steps_per_epoch)).ljust(14)
        time_info = 'time: %.2f sec' % (time.time() - step_start_time)
        print('  '.join((epoch_info, step_info, time_info, loss_info)), flush=True)

        # tensorboard logging
        writer.add_figure(tag='volumes',
                          figure=create_figure(y_true[0].cpu(), inputs[0].cpu(), y_pred[0].detach().cpu()),
                          global_step=(epoch + 1) * args.steps_per_epoch + step + 1)

# final model save
model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))
