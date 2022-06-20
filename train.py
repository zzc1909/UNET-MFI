import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms as T
from Model import no_share_unet
from transforms import *
from Ddataset import BraTSDataset
from glob import glob
from losses import *
import torchvision
#from evaluation import *
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from metrics import dice_coef
from tqdm import tqdm
import argparse


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=1200,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight decay')

    # datapath and dataset
    parser.add_argument('--train_list', type=str, default='train1.txt')
    parser.add_argument('--val_list', type=str, default='val1.txt')
    parser.add_argument('--data_path', type=str, default='/media/zzc/zzc12/pycharmproject/3D_miss_mod/2018/data/')
    parser.add_argument('--train_transforms', type=str, default='Compose([ RandCrop3D((128,128,128)), RandomRotion(10),RandomFlip(0), NumpyType((np.float32, np.int64)), ])')
    parser.add_argument('--val_transforms', type=str, default='Compose([Pad((0, 0, 0, 5, 0)),NumpyType((np.float32, np.int64)),])')
    opt = parser.parse_args()
    return opt


def adjust_lr(init_lr,optimizer, epoch,total_epo):
    cur_lr = init_lr * (1-epoch/total_epo)**0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

def train_model(model, criterion, optimizer, dataload, val_loader,sche=None,num_epochs=1200,deepSupvision=False):
    best_acc = 0.80
    for epoch in range(num_epochs):
        model.train()
        adjust_lr(init_lr=5e-5,optimizer=optimizer,epoch=epoch,total_epo=num_epochs)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x1,x2,x3,x4,y,mask_code in dataload:
            step += 1
            input1,input2,input3,input4 = x1.cuda(),x2.cuda(),x3.cuda(),x4.cuda()
            labels = y.cuda()
            optimizer.zero_grad()


            mask_code = mask_code.cuda()
            allOut =  model(input1,input2,input3,input4,mask_code)
            loss= 0
            step_loss = []
            for out in allOut:
                cur_loss = criterion(out, labels)
                loss += cur_loss
                step_loss.append(cur_loss.item())
            loss.backward()
            optimizer.step()
            epoch_loss += step_loss[-1]
            print("%d/%d,lr:%0.6f" %(step, (dt_size - 1) // dataload.batch_size + 1,optimizer.state_dict()['param_groups'][0]['lr']),'loss:',step_loss)

        ###evaluate model every 5 epoch
        if epoch % 5 ==0:
            print('evaling..........')
            H, W, T = 240, 240, 155
            WT_dice, TC_dice, ET_dice = [], [], []

            model.eval()
            with torch.no_grad():
                for i, (x1, x2, x3, x4, target, mask) in tqdm(enumerate(val_loader), total=len(val_loader)):
                    mask = mask.cuda()

                    input_x11 = x1[:, :, :120, :120, :]
                    input_x12 = x1[:, :, :120, 120:, :]
                    input_x13 = x1[:, :, 120:, :120, :]
                    input_x14 = x1[:, :, 120:, 120:, :]
                    input_x11, input_x12, input_x13, input_x14 = input_x11.cuda(), input_x12.cuda(), input_x13.cuda(), input_x14.cuda()

                    input_x21 = x2[:, :, :120, :120, :]
                    input_x22 = x2[:, :, :120, 120:, :]
                    input_x23 = x2[:, :, 120:, :120, :]
                    input_x24 = x2[:, :, 120:, 120:, :]
                    input_x21, input_x22, input_x23, input_x24 = input_x21.cuda(), input_x22.cuda(), input_x23.cuda(), input_x24.cuda()

                    input_x31 = x3[:, :, :120, :120, :]
                    input_x32 = x3[:, :, :120, 120:, :]
                    input_x33 = x3[:, :, 120:, :120, :]
                    input_x34 = x3[:, :, 120:, 120:, :]
                    input_x31, input_x32, input_x33, input_x34 = input_x31.cuda(), input_x32.cuda(), input_x33.cuda(), input_x34.cuda()

                    input_x41 = x4[:, :, :120, :120, :]
                    input_x42 = x4[:, :, :120, 120:, :]
                    input_x43 = x4[:, :, 120:, :120, :]
                    input_x44 = x4[:, :, 120:, 120:, :]
                    input_x41, input_x42, input_x43, input_x44 = input_x41.cuda(), input_x42.cuda(), input_x43.cuda(), input_x44.cuda()

                    output1 = model(input_x11, input_x21, input_x31, input_x41, mask)[-5:]
                    output2 = model(input_x12, input_x22, input_x32, input_x42, mask)[-5:]
                    output3 = model(input_x13, input_x23, input_x33, input_x43, mask)[-5:]
                    output4 = model(input_x14, input_x24, input_x34, input_x44, mask)[-5:]
                    output1 = (output1[0] + output1[1] + output1[2] + output1[3] + output1[4]) / 5
                    output2 = (output2[0] + output2[1] + output2[2] + output2[3] + output2[4]) / 5
                    output3 = (output3[0] + output3[1] + output3[2] + output3[3] + output3[4]) / 5
                    output4 = (output4[0] + output4[1] + output4[2] + output4[3] + output4[4]) / 5
                    outputs_half1 = torch.cat((output1, output2), dim=3)
                    outputs_half2 = torch.cat((output3, output4), dim=3)
                    outputs = torch.cat((outputs_half1, outputs_half2), dim=2)
                    outputs = torch.sigmoid(outputs)

                    output = outputs[0, :, :H, :W, :T].cpu().numpy()
                    target = target[0, :, :H, :W, :T].numpy()
                    WT_out = output[0, ...]
                    WT_out[WT_out > 0.5] = 1
                    WT_out[WT_out < 0.5] = 0

                    TC_out = output[1, ...]
                    TC_out[TC_out > 0.5] = 1
                    TC_out[TC_out < 0.5] = 0

                    ET_out = output[2, ...]  # 240,240,155
                    ET_out[ET_out > 0.5] = 1
                    ET_out[ET_out < 0.5] = 0

                    WT_label = target[0, ...]
                    TC_label = target[1, ...]
                    ET_label = target[2, ...]  # 240,240,155
                    wt_dice = dice_coef(WT_out, WT_label)
                    et_dice = dice_coef(ET_out, ET_label)
                    tc_dice = dice_coef(TC_out, TC_label)
                    WT_dice.append(wt_dice)
                    ET_dice.append(et_dice)
                    TC_dice.append(tc_dice)

                mean_wt,mean_tc,mean_et = np.mean(WT_dice),np.mean(TC_dice),np.mean(ET_dice)
                print('WT Dice: %.4f' % mean_wt)
                print('TC Dice: %.4f' % mean_tc)
                print('ET Dice: %.4f' % mean_et)
            acc = (mean_wt+mean_tc+mean_et)/3
            if acc>best_acc:
                state = {
                    'model.state': model.state_dict(),
                    'WT_Dice': mean_wt,
                    'TC_Dice': mean_tc,
                    'ET_Dice': mean_et,
                    'epoch': epoch,
                }
                torch.save(state, 'best_model/val_fold1_18_%.4f.pth' % (acc))
                best_acc = acc
        ###save model parameters every 50 epoch when the epoch>800
        if epoch % 50 ==0 and epoch >= 899:
            torch.save(model.state_dict(), 'best_model/val_fold1_18_%d.pth' % (epoch))
        if sche:
            sche.step()
        print("epoch %d mean_loss:%0.6f bset_acc%0.6f" % (epoch, epoch_loss,best_acc))

    return model


# train the model
def train():
    opt = parse_option()
    model = no_share_unet(in_channel=1, out_channel=3, diff=True,deepSupvision=True).cuda()
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(),lr=opt.learning_rate,weight_decay=opt.weight_decay)

    train_set = BraTSDataset(opt.train_list, root=opt.data_path, mode='train',
                             transforms=opt.train_transforms)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=opt.batch_size,
        shuffle=True,
        pin_memory=True, )

    val_set = BraTSDataset(opt.val_list, root=opt.data_path, mode='val',
                            transforms=opt.val_transforms)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=opt.batch_size,
        pin_memory=True, )

    train_model(model, criterion, optimizer, train_loader,val_loader,sche=None,num_epochs=opt.epochs,deepSupvision=True)



if __name__ == '__main__':
    train()

