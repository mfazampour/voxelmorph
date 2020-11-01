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
from matplotlib import gridspec

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm


def main():
    parser = parse_args()
    args = parser.parse_args()

    bidir = args.bidir

    # tensorboard
    if args.log_dir is None:
        args.log_dir = args.model_dir
    writer = SummaryWriter(log_dir=f'{args.log_dir}/{datetime.now().strftime("%m.%d.%Y_%H.%M")}')

    generator = create_data_generator(args)

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

    model = create_model(args, bidir, device, inshape, nb_gpus)

    losses, optimizer, weights, loss_names = optimizers(args, bidir, model)
    # training loops
    train(args, device, generator, losses, model, model_dir, optimizer, weights, writer, loss_names)


def optimizers(args, bidir, model):
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # prepare image loss
    loss_names = []
    available_loss_images = ['ncc', 'mse', 'ssim', 'mind']
    if args.image_loss == 'ncc':
        image_loss_func = vxm.losses.NCC().loss
    elif args.image_loss == 'mse':
        image_loss_func = vxm.losses.MSE().loss
    elif args.image_loss == 'ssim':
        image_loss_func = vxm.losses.SSIM().loss
    elif args.image_loss == 'mind':
        image_loss_func = vxm.losses.MIND().loss
    else:
        raise ValueError(f'Image loss should be among {available_loss_images}, but found {args.image_loss}')
    # need two image loss functions if bidirectional
    if bidir:
        losses = [image_loss_func, image_loss_func]
        loss_names += [args.image_loss, args.image_loss]
        weights = [0.5, 0.5]
    else:
        losses = [image_loss_func]
        loss_names += [args.image_loss]
        weights = [1]
    # prepare deformation loss
    losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
    loss_names += 'Regularization'
    weights += [args.weight]
    return losses, optimizer, weights, loss_names


def create_model(args, bidir, device, inshape, nb_gpus):
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
    return model


def parse_args():
    global parser
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
    parser.add_argument('--cudnn-nondet', action='store_true',
                        help='disable cudnn determinism - might slow down training')
    # network architecture parameters
    parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters (default: 16 32 32 32)')
    parser.add_argument('--dec', type=int, nargs='+',
                        help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
    parser.add_argument('--int-steps', type=int, default=7, help='number of integration steps (default: 7)')
    parser.add_argument('--int-downsize', type=int, default=2,
                        help='flow downsample factor for integration (default: 2)')
    parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')
    # loss hyperparameters
    parser.add_argument('--image-loss', default='mse',
                        help='image reconstruction loss - can be mse or ncc (default: mse)')
    parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                        help='weight of deformation loss (default: 0.01)')
    parser.add_argument('--use-biobank', action='store_true', help='set to True if using biobank data')
    parser.add_argument('--inshape', type=int, nargs='+',
                        help='after cropping shape of input. '
                             'default is equal to image size. specify if the input can\'t path through UNet')
    parser.add_argument('--log-dir', type=str, default=None, help='folder for tensorboard logs')
    parser.add_argument('--display_freq', type=int, default=20, help='frequency of plotting results in tensorboard')
    parser.add_argument('--patient_list_src', type=str, default='/tmp/',
                        help='directory to store patient list for training and testing')
    return parser


def create_data_generator(args):
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
                                                use_biobank=args.use_biobank, target_shape=args.inshape,
                                                patient_list_src=args.patient_list_src)
    return generator


def train(args, device, generator, losses, model, model_dir, optimizer, weights, writer, loss_names):
    ssim = vxm.losses.SSIM()

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
            loss = torch.tensor([0], dtype=torch.float).to(device)
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
            tensorboard_log(args, epoch, inputs, loss_list, loss_names, step, writer,
                            y_pred[0].detach(), y_true, y_pred[-1].detach(), ssim=ssim)
    # final model save
    model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))


def tensorboard_log(args, epoch, inputs, loss_list, loss_names, step,
                    writer: SummaryWriter, y_pred, y_true, ddf, ssim: vxm.losses.SSIM = None):
    global_step = (epoch) * args.steps_per_epoch + step + 1
    if global_step % args.display_freq == 1:
        figure = vxm.torch.utils.create_figure(y_true[0].cpu(), inputs[0].cpu(), y_pred.cpu(),
                                               ddf.cpu())
        writer.add_figure(tag='volumes',
                          figure=figure,
                          global_step=epoch * args.steps_per_epoch + step + 1)
        loss_dict = {}
        for name, value in zip(loss_names, list(map(float, loss_list))):
            loss_dict[name] = value
        writer.add_scalars(main_tag='loss', tag_scalar_dict=loss_dict, global_step=global_step)

        fix_to_mov = torch.mean((y_true[0][y_true[0] != 0] - inputs[0][y_true[0] != 0]) ** 2).cpu()
        fix_to_reg = torch.mean((y_true[0][y_true[0] != 0] - y_pred[y_true[0] != 0]) ** 2).cpu()
        ssim_mov = ssim.loss(y_true[0], inputs[0]).item()
        ssim_reg = ssim.loss(y_true[0], y_pred).item()
        ssim_increment = ssim_reg/(ssim_mov + 0.001) - 1
        diff_dict = {'Fix. to Mov.': fix_to_mov.item(),
                     'Fix. to Reg.': fix_to_reg.item(),
                     'SSD improvement': 1 - (fix_to_reg.item()/(fix_to_mov.item() + 1e-9)),
                     'SSIM Mov.': ssim_mov,
                     'SSIM Reg.': ssim_reg,
                     'SSIM increment': ssim_increment}
        writer.add_scalars(main_tag='diffs', tag_scalar_dict=diff_dict, global_step=global_step)


if __name__ == "__main__":
    main()
