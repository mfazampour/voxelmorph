import torch
from torch.autograd import Variable
from torch import optim
import cv2
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from voxelmorph.torch.losses import SSIM

def ssim_test():
    # npImg1 = cv2.imread("/tmp/einstein.png")
    # img1 = torch.from_numpy(np.transpose(npImg1, (2, 0, 1))).float().unsqueeze(0) / 255.0

    img = sitk.ReadImage('/mnt/data/biobank_sample/imgs/2515394/T1_brain_affine_to_mni.nii.gz')
    npImg1 = sitk.GetArrayFromImage(img)
    img1 = torch.tensor(npImg1).unsqueeze(0).unsqueeze(0)
    img1 = (img1 - img1.min()) / (img1.max() - img1.min())

    img2 = torch.rand(img1.size())

    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()

    img2.requires_grad = True

    # Functional: pytorch_ssim.ssim(img1, img2, window_size = 11, size_average = True)
    ssim = SSIM()
    ssim_value = ssim.loss(img1, img2).item()
    print("Initial ssim:", ssim_value)

    # Module: pytorch_ssim.SSIM(window_size = 11, size_average = True)
    # ssim_loss = SSIM2()

    optimizer = optim.Adam([img2], lr=0.01)

    while ssim_value < 0.95:
        optimizer.zero_grad()
        ssim_out = -ssim.loss(img1, img2)
        ssim_value = - ssim_out.item()
        if len(img1.shape) - 2 == 3:  # 3D
            if ssim_value > 0.5:
                plt.imshow(img2.detach().cpu().squeeze()[int(img2.shape[2] / 2), :, :].numpy())
                plt.show()
                plt.imshow(img2.detach().cpu().squeeze()[:, int(img2.shape[3] / 2), :].numpy())
                plt.show()
                plt.imshow(img2.detach().cpu().squeeze()[:, :, int(img2.shape[4] / 2)].numpy())
                plt.show()
        else:
            if ssim_value > 0.9:
                plt.imshow(img2.detach().cpu().squeeze().permute((1, 2, 0)).numpy())
                plt.show()
        print(ssim_value)
        ssim_out.backward()
        optimizer.step()