import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
# from models import DnCNN
from networks import DnCNN
from myLoss import MaxLikelyloss
from dataset import prepare_data, Dataset
from utils import *
import misc
from networks import new_unet
from networks import unet


def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    # net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    if opt.uncertainty:
        net = unet.UNet(in_channels=1, out_channels=2)
    else:
        # net = DnCNN.DnCNN(num_layers=opt.num_of_layers)
        net = unet.UNet(in_channels=1,out_channels=1)

    # net.apply(weights_init_kaiming)
    if opt.uncertainty:
        criterion = MaxLikelyloss()
    else:    
        criterion = nn.MSELoss(size_average=False)
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000000)
    # training
    # writer = SummaryWriter(opt.save_dir)
    step = 0
    noiseL_B=[0,55] # ingnored when opt.mode=='S'
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            scheduler.step()
            current_lr = scheduler.optimizer.param_groups[0]['lr']
        # set learning rate
        # for param_group in optimizer.param_groups:
        #     param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data
            if opt.mode == 'S':
                noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL/255.)
            if opt.mode == 'B':
                noise = torch.zeros(img_train.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0,:,:,:].size()
                    noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
            imgn_train = img_train + noise
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())
            out_train = model(imgn_train)
            # loss = criterion(out_train, noise) / (imgn_train.size()[0]*2)
            loss = criterion(out_train, img_train) / (imgn_train.size()[0]*2)
            loss.backward()
            optimizer.step()
            # results
            model.eval()
            # out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
            out_train = torch.clamp(model(imgn_train), 0., 1.)
            psnr_train = batch_PSNR(out_train[:,:1], img_train, 1.)
            # print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
            #     (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
            if i % opt.print_freq == 0:
                misc.message_log(opt, "Train: epoch: {}, iters: {}/{} loss: {:.6f} PSNR_train: {:.6f}".format
                    (epoch+1, i+1, len(loader_train), loss.item(), psnr_train) )

            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            # if step % 10 == 0:
            #     # Log the scalar values
            #     writer.add_scalar('loss', loss.item(), step)
            #     writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
        ## the end of each epoch

        model.eval()
        # validate
        psnr_val = 0
        loss_val = 0
        for k in range(len(dataset_val)):
            img_val = torch.unsqueeze(dataset_val[k], 0)
            noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
            imgn_val = img_val + noise
            img_val, imgn_val = Variable(img_val.cuda(), volatile=True), Variable(imgn_val.cuda(), volatile=True)
            # out_val = torch.clamp(imgn_val-model(imgn_val), 0., 1.)
            out_val = model(imgn_val)
            loss_val += criterion(out_val, img_val)
            out_val = torch.clamp(out_val, 0., 1.)
            psnr_val += batch_PSNR(out_val[:,:1], img_val, 1.)

            misc.save_result(imgn_val, out_val, opt, k)
        loss_val /= len(dataset_val)
        psnr_val /= len(dataset_val)
        # print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
        print("Average loss on validation dataset: {}".format(loss_val))
        misc.message_log(opt, "Valid: epoch {}, loss_val: {:.6f}, PSNR_val: {:.4f}, current lr: {:.6f}".format(epoch+1, loss_val, psnr_val, current_lr) )

        # save model
        torch.save(model.state_dict(), os.path.join(opt.save_dir, 'net.pth'))

if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description="Unet for denoising...")
    parser.add_argument("--preprocess", default=False, action='store_true', help='run prepare_data or not')
    parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
    parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--milestone", type=int, default=70, help="When to decay learning rate; should be less than epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--save_dir", type=str, default="logs", help='path of log files')
    parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
    parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
    parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
    parser.add_argument("--uncertainty", default=False, action='store_true', help='if estimate the uncertainty.')
    parser.add_argument("--gpu_ids", type=str, default="0", help='specify which gpu to use.')
    parser.add_argument("--datapath_train", type=str, default="./data", help='datapath of training set.')
    parser.add_argument("--datapath_valid", type=str, default="./data", help='datapath of validation set.')
    parser.add_argument("--max_num_iter", type=int, default=2000000, help='the max number of iterations')
    parser.add_argument("--print_freq", type=int, default=50, help='frequency of showing training results on console')


    opt = parser.parse_args()

    print(opt.preprocess)
    print(opt.uncertainty)
    print(type(opt.preprocess) ) 

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids


    if opt.preprocess:
        print("to process the  data")
        if opt.mode == 'S':
            prepare_data(data_path=opt.datapath_train, patch_size=256, stride=70, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path=opt.datapath_train, patch_size=50, stride=10, aug_times=2)
    main()
