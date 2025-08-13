# coding=utf-8
import os
from model.TriFuse import TriFuse
# from model.TriFusemobilenet import TriFuse
# from model.TriFuseVGG import TriFuse
# from model.TriFuseResNet import TriFuse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
from lib.dataset import Data
from lib.data_prefetcher import DataPrefetcher
import pytorch_iou
import pytorch_ssim

IOU = pytorch_iou.IOU(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
bce_loss = nn.BCELoss(reduction='mean')

L1 = nn.L1Loss()

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def muti_loss(pred, target):
    bce_out = bce_loss(pred, target)
    iou_out = IOU(pred, target)

    loss = (bce_out + iou_out).mean()

    return loss

# BBA
def tesnor_bound(img, ksize):
    '''
    :param img: tensor, B*C*H*W
    :param ksize: tensor, ksize * ksize
    :param 2patches: tensor, B * C * H * W * ksize * ksize
    :return: tensor, (inflation - corrosion), B * C * H * W
    '''

    B, C, H, W = img.shape
    pad = int((ksize - 1) // 2)
    img_pad = F.pad(img, pad=[pad, pad, pad, pad], mode='constant', value=0)
    # unfold in the second and third dimensions
    patches = img_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    corrosion, _ = torch.min(patches.contiguous().view(B, C, H, W, -1), dim=-1)
    inflation, _ = torch.max(patches.contiguous().view(B, C, H, W, -1), dim=-1)
    return inflation - corrosion


if __name__ == '__main__':

    save_path = './model_weight'
    if not os.path.exists(save_path): os.mkdir(save_path)
    lr = 0.0001
    batch_size = 4
    epoch = 400
    num_params = 0

    img_root = './dataset/VDT2048/Train/'
    # img_root = './dataset/VDTRW/Train/' # train VDT-RW
    data = Data(img_root)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=8)

    net = TriFuse().cuda()
    net.load_pretrained_model()
    params = net.parameters()
    optimizer = torch.optim.Adam(params, 0.0001, betas=(0.9, 0.999))

    for p in net.parameters():
        num_params += p.numel()
    print("The number of parameters: {}".format(num_params))
    iter_num = len(loader)
    net.train()

    for epochi in tqdm(range(0, epoch + 1)):
        prefetcher = DataPrefetcher(loader)
        rgb, t, d, eg, label = prefetcher.next()
        r_sal_loss = 0
        epoch_ave_loss = 0
        i = 0
        while rgb is not None:
            i += 1

            bound = tesnor_bound(label, 3).cuda()

            x1e_pred, x1e_pred_t, x1e_pred_d, x_pred, x_refine = net(rgb, t, d)
            predict_bound0 = tesnor_bound(torch.sigmoid(x_refine), 3)
            loss_x0eg = muti_loss(predict_bound0, bound)
            loss0 = structure_loss(x_refine, label)


            predict_bound1 = tesnor_bound(torch.sigmoid(x_pred[0]), 3)
            predict_bound2 = tesnor_bound(torch.sigmoid(x_pred[1]), 3)
            predict_bound3 = tesnor_bound(torch.sigmoid(x_pred[2]), 3)
            predict_bound4 = tesnor_bound(torch.sigmoid(x_pred[3]), 3)

            loss1 = structure_loss(x1e_pred, label)

            loss2 = structure_loss(x1e_pred_t, label)

            loss3 = structure_loss(x1e_pred_d, label)

            loss4 = structure_loss(x_pred[0], label)
            loss5 = structure_loss(x_pred[1], label)
            loss6 = structure_loss(x_pred[2], label)
            loss7 = structure_loss(x_pred[3], label)

            loss_x4eg = muti_loss(predict_bound4, bound)
            loss_x3eg = muti_loss(predict_bound3, bound)
            loss_x2eg = muti_loss(predict_bound2, bound)
            loss_x1eg = muti_loss(predict_bound1, bound)

            sal_loss = (loss0 * 2 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss_x4eg + loss_x3eg + loss_x2eg + loss_x1eg + loss_x0eg * 2)

            r_sal_loss += sal_loss.data
            sal_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 100 == 0:
                loss_value = r_sal_loss / 100
                log_msg = 'epoch: [%2d/%2d], iter: [%5d/%5d]  ||  loss: %5.4f, lr: %7.6f' % (
                    epochi, epoch, i, iter_num, loss_value, lr)
                print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  loss : %5.4f, lr: %7.6f' % (
                    epochi, epoch, i, iter_num, r_sal_loss / 100, lr,))
                epoch_ave_loss += (r_sal_loss / 100)
                r_sal_loss = 0
            rgb, t, d, eg, label = prefetcher.next()
        print('epoch-%2d_ave_loss: %7.6f' % (epochi, (epoch_ave_loss / (10.5 / batch_size))))
        epoch_avg_loss = epoch_ave_loss / (10.5 / batch_size)
        if epochi % 10 == 0:
            model_path = '%s/epoch_%d.pth' % (save_path, epochi)
            torch.save(net.state_dict(), '%s/epoch_%d.pth' % (save_path, epochi))
    torch.save(net.state_dict(), '%s/final.pth' % (save_path))