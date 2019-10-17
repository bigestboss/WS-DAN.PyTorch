"""TRAINING
Created: May 04,2019 - Yuchong Gu
Revised: May 07,2019 - Yuchong Gu
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import time
import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from optparse import OptionParser

from utils import accuracy
from models import *
from dataset import *

import cv2

def generate_attention_image(image, attention_map):
    h, w, _ = image.shape
    mask = np.mean(attention_map, axis=-1, keepdims=True)
    mask = (mask / np.max(mask) * 255.0).astype(np.uint8)
    mask = cv2.resize(mask, (w, h))

    image = (image / 2.0 + 0.5) * 255.0
    image = image.astype(np.uint8)

    color_map = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
    attention_image = cv2.addWeighted(image, 0.5, color_map.astype(np.uint8), 0.5, 0)
    attention_image = cv2.cvtColor(attention_image, cv2.COLOR_BGR2RGB)
    return attention_image

def main():
    parser = OptionParser()
    parser.add_option('-j', '--workers', dest='workers', default=16, type='int',
                      help='number of data loading workers (default: 16)')
    parser.add_option('-e', '--epochs', dest='epochs', default=80, type='int',
                      help='number of epochs (default: 80)')
    parser.add_option('-b', '--batch-size', dest='batch_size', default=16, type='int',
                      help='batch size (default: 16)')
    parser.add_option('-c', '--ckpt', dest='ckpt', default=False,
                      help='load checkpoint model (default: False)')
    parser.add_option('-v', '--verbose', dest='verbose', default=100, type='int',
                      help='show information for each <verbose> iterations (default: 100)')

    parser.add_option('--lr', '--learning-rate', dest='lr', default=1e-3, type='float',
                      help='learning rate (default: 1e-3)')
    parser.add_option('--sf', '--save-freq', dest='save_freq', default=10, type='int',
                      help='saving frequency of .ckpt models (default: 1)')
    parser.add_option('--sd', '--save-dir', dest='save_dir', default='./models',
                      help='saving directory of .ckpt models (default: ./models)')
    parser.add_option('--init', '--initial-training', dest='initial_training', default=1, type='int',
                      help='train from 1-beginning or 0-resume training (default: 1)')

    (options, args) = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s', level=logging.INFO)
    warnings.filterwarnings("ignore")

    ##################################
    # Initialize model
    ##################################
    image_size = (448, 448)
    num_classes = 200
    num_attentions = 32
    start_epoch = 0

    feature_net = inception_v3(pretrained=True)
    net = WSDAN(num_classes=num_classes, M=num_attentions, net=feature_net)

    # feature_center: size of (#classes, #attention_maps, #channel_features)
    feature_center = torch.zeros(num_classes, num_attentions, net.num_features * net.expansion).to(torch.device("cuda"))

    if options.ckpt:
        ckpt = options.ckpt

        if options.initial_training == 0:
            # Get Name (epoch)
            epoch_name = (ckpt.split('/')[-1]).split('.')[0]
            start_epoch = int(epoch_name)

        # Load ckpt and get state_dict
        checkpoint = torch.load(ckpt)
        state_dict = checkpoint['state_dict']

        # Load weights
        net.load_state_dict(state_dict)
        logging.info('Network loaded from {}'.format(options.ckpt))

        # load feature center
        if 'feature_center' in checkpoint:
            feature_center = checkpoint['feature_center'].to(torch.device("cuda"))
            logging.info('feature_center loaded from {}'.format(options.ckpt))

    ##################################
    # Initialize saving directory
    ##################################
    save_dir = options.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ##################################
    # Use cuda
    ##################################
    cudnn.benchmark = True
    net.to(torch.device("cuda"))
    net = nn.DataParallel(net)

    ##################################
    # Load dataset
    ##################################
    train_dataset, validate_dataset = CustomDataset(phase='train', shape=image_size), \
                                      CustomDataset(phase='val'  , shape=image_size)

    train_loader, validate_loader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True,
                                               num_workers=options.workers, pin_memory=True), \
                                    DataLoader(validate_dataset, batch_size=options.batch_size * 4, shuffle=False,
                                               num_workers=options.workers, pin_memory=True)

    optimizer = torch.optim.SGD(net.parameters(), lr=options.lr, momentum=0.9, weight_decay=0.00001)
    loss = nn.CrossEntropyLoss()

    ##################################
    # Learning rate scheduling
    ##################################
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    ##################################
    # TRAINING
    ##################################
    logging.info('')
    logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
                 format(options.epochs, options.batch_size, len(train_dataset), len(validate_dataset)))
    writer = SummaryWriter(log_dir='./log', comment='WS-DAN')
    for epoch in range(start_epoch, options.epochs):
        train(epoch=epoch,
              data_loader=train_loader,
              net=net,
              feature_center=feature_center,
              loss=loss,
              optimizer=optimizer,
              save_freq=options.save_freq,
              save_dir=options.save_dir,
              verbose=options.verbose,
              writer=writer
              )
        val_loss = validate(data_loader=validate_loader,
                            net=net,
                            loss=loss,
                            verbose=options.verbose)
        scheduler.step()
    writer.close()

def train(**kwargs):
    # Retrieve training configuration
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    loss = kwargs['loss']
    optimizer = kwargs['optimizer']
    feature_center = kwargs['feature_center']
    epoch = kwargs['epoch']
    save_freq = kwargs['save_freq']
    save_dir = kwargs['save_dir']
    verbose = kwargs['verbose']
    writer = kwargs['writer']
    # Attention Regularization: LA Loss
    l2_loss = nn.MSELoss()

    # Default Parameters
    beta = 0.05
    theta_c = 0.5
    theta_d = 0.5
    crop_size = (256, 256)  # size of cropped images for 'See Better'

    # metrics initialization
    batches = 0
    epoch_loss = np.array([0, 0, 0], dtype='float')  # Loss on Raw/Crop/Drop Images
    epoch_acc = np.array([[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]], dtype='float')  # Top-1/3/5 Accuracy for Raw/Crop/Drop Images

    # begin training
    start_time = time.time()
    logging.info('Epoch %03d, Learning Rate %g' % (epoch + 1, optimizer.param_groups[0]['lr']))
    net.train()
    for i, (X, y) in enumerate(data_loader):
        batch_start = time.time()

        # obtain data for training
        X = X.to(torch.device("cuda"))
        y = y.to(torch.device("cuda"))

        ##################################
        # Raw Image
        ##################################
        # print(X.size(2), X.size(3))
        y_pred, feature_matrix, attention_map = net(X)
        # reshape center and bap
        feature_center=feature_center.reshape((feature_center.shape[0],-1))
        feature_matrix=feature_matrix.reshape((feature_matrix.shape[0],-1))
        #get this batch's batch_center
        batch_center = feature_center[y]
        #Normalize centermatrix batch_center
        batch_center=nn.functional.normalize(batch_center,2,-1)
        # Update Feature Center
        feature_center[y] += beta * (feature_matrix.detach() - batch_center)
        loss_center = l2_loss(feature_matrix, batch_center)

        # loss
        batch_loss_1 = loss(y_pred, y)
        epoch_loss[0] += batch_loss_1.item()
        # backward
        # optimizer.zero_grad()
        # batch_loss.backward()
        # optimizer.step()



        # metrics: top-1, top-3, top-5 error
        with torch.no_grad():
            epoch_acc[0] += accuracy(y_pred, y, topk=(1, 3, 5))

        # ##################################
        # # Attention Cropping
        # ##################################
        # with torch.no_grad():
        #     crop_mask = F.upsample_bilinear(attention_map, size=(X.size(2), X.size(3))) > theta_c
        #     crop_images = []
        #     for batch_index in range(crop_mask.size(0)):
        #         nonzero_indices = torch.nonzero(crop_mask[batch_index, 0, ...])
        #         height_min = nonzero_indices[:, 0].min()
        #         height_max = nonzero_indices[:, 0].max()
        #         width_min = nonzero_indices[:, 1].min()
        #         width_max = nonzero_indices[:, 1].max()
        #         # print(height_min,height_max,width_min,width_max)
        #         crop_images.append(F.upsample_bilinear(X[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max], size=crop_size))
        #     crop_images = torch.cat(crop_images, dim=0)
        # # crop images forward
        # # print(crop_images.size(2), crop_images.size(3))
        # y_pred, _, _ = net(crop_images)
        #
        # # loss
        # batch_loss_2 = loss(y_pred, y)
        # epoch_loss[1] += batch_loss_2.item()
        #
        # # backward
        # # optimizer.zero_grad()
        # # batch_loss.backward()
        # # optimizer.step()
        #
        # # metrics: top-1, top-3, top-5 error
        # with torch.no_grad():
        #     epoch_acc[1] += accuracy(y_pred, y, topk=(1, 3, 5))
        #
        # ##################################
        # # Attention Dropping
        # ##################################
        # with torch.no_grad():
        #     drop_mask = F.upsample_bilinear(attention_map, size=(X.size(2), X.size(3))) <= theta_d
        #     drop_images = X * drop_mask.float()
        #
        # # drop images forward
        # y_pred, _, _ = net(drop_images)
        #
        # # loss
        # batch_loss_3 = loss(y_pred, y)
        # epoch_loss[2] += batch_loss_3.item()

        # # metrics: top-1, top-3, top-5 error
        # with torch.no_grad():
        #     epoch_acc[2] += accuracy(y_pred, y, topk=(1, 3, 5))
        # backward
        # totol_loss = batch_loss_1+batch_loss_2+batch_loss_3+loss_center
        totol_loss = batch_loss_1+loss_center
        optimizer.zero_grad()
        totol_loss.backward()
        optimizer.step()



        # end of this batch
        batches += 1
        batch_end = time.time()
        if (i + 1) % verbose == 0:
            logging.info('\tBatch %d: (Raw) Loss %.4f, Accuracy: (%.2f, %.2f, %.2f), (Crop) Loss %.4f, Accuracy: (%.2f, %.2f, %.2f), (Drop) Loss %.4f, Accuracy: (%.2f, %.2f, %.2f), Time %3.2f' %
                         (i + 1,
                          epoch_loss[0] / batches, epoch_acc[0, 0] / batches, epoch_acc[0, 1] / batches, epoch_acc[0, 2] / batches,
                          epoch_loss[1] / batches, epoch_acc[1, 0] / batches, epoch_acc[1, 1] / batches, epoch_acc[1, 2] / batches,
                          epoch_loss[2] / batches, epoch_acc[2, 0] / batches, epoch_acc[2, 1] / batches, epoch_acc[2, 2] / batches,
                          batch_end - batch_start))
            writer.add_image('raw_img', X[0], (epoch+1) * 100+(i + 1) / verbose)
            # writer.add_image('crop_mask', crop_mask[0], (epoch+1) * 100+(i + 1) / verbose)
            # writer.add_image('crop_img', crop_images[0], (epoch+1) * 100+(i + 1) / verbose)
            # writer.add_image('drop_mask', drop_mask[0], (epoch+1) * 100+(i + 1) / verbose)
            # writer.add_image('drop_img', drop_images[0], (epoch+1) * 100+(i + 1) / verbose)
            crop_mask = F.upsample_bilinear(attention_map, size=(X.size(2), X.size(3))) > theta_c
            writer.add_image('attention_img', (X * crop_mask.float())[0], (epoch+1) * 100+(i + 1) / verbose)
            # print(type(attention_map[0]))
            # writer.add_image('attention_img',generate_attention_image(X[0],attention_map[0].cpu().numpy()) , (epoch + 1) * 100 + (i + 1) / verbose)

    # save checkpoint model
    if epoch % save_freq == 0:
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict,
            'feature_center': feature_center.cpu()},
            os.path.join(save_dir, '%03d.ckpt' % (epoch + 1)))

    # end of this epoch
    end_time = time.time()

    # metrics for average
    epoch_loss /= batches
    epoch_acc /= batches

    # show information for this epoch
    logging.info('Train: (Raw) Loss %.4f, Accuracy: (%.2f, %.2f, %.2f), (Crop) Loss %.4f, Accuracy: (%.2f, %.2f, %.2f), (Drop) Loss %.4f, Accuracy: (%.2f, %.2f, %.2f), Time %3.2f'%
                 (epoch_loss[0], epoch_acc[0, 0], epoch_acc[0, 1], epoch_acc[0, 2],
                  epoch_loss[1], epoch_acc[1, 0], epoch_acc[1, 1], epoch_acc[1, 2],
                  epoch_loss[2], epoch_acc[2, 0], epoch_acc[2, 1], epoch_acc[2, 2],
                  end_time - start_time))
    writer.add_scalars('scalar/train',{'acc_raw':epoch_acc[0, 0],'acc_crop':epoch_acc[1, 0],'acc_drop':epoch_acc[2, 0]},epoch)
    writer.add_scalars('scalar/train',{'loss_raw':epoch_loss[0],'loss_crop':epoch_loss[1],'loss_drop':epoch_loss[2]},epoch)

def validate(**kwargs):
    # Retrieve training configuration
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    loss = kwargs['loss']
    verbose = kwargs['verbose']

    # Default Parameters
    theta_c = 0.5
    crop_size = (256, 256)  # size of cropped images for 'See Better'

    # metrics initialization
    batches = 0
    epoch_loss = 0
    epoch_acc = np.array([0, 0, 0], dtype='float') # top - 1, 3, 5

    # begin validation
    start_time = time.time()
    net.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            batch_start = time.time()

            # obtain data
            X = X.to(torch.device("cuda"))
            y = y.to(torch.device("cuda"))

            ##################################
            # Raw Image
            ##################################
            y_pred_raw, feature_matrix, attention_map = net(X)

            # ##################################
            # # Object Localization and Refinement
            # ##################################
            # crop_mask = F.upsample_bilinear(attention_map, size=(X.size(2), X.size(3))) > theta_c
            # crop_images = []
            # for batch_index in range(crop_mask.size(0)):
            #     nonzero_indices = torch.nonzero(crop_mask[batch_index, 0, ...])
            #     height_min = nonzero_indices[:, 0].min()
            #     height_max = nonzero_indices[:, 0].max()
            #     width_min = nonzero_indices[:, 1].min()
            #     width_max = nonzero_indices[:, 1].max()
            #     crop_images.append(F.upsample_bilinear(X[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max], size=crop_size))
            # crop_images = torch.cat(crop_images, dim=0)
            #
            # y_pred_crop, _, _ = net(crop_images)

            # final prediction
            # y_pred = (y_pred_raw + y_pred_crop) / 2
            y_pred = y_pred_raw
            # loss
            batch_loss = loss(y_pred, y)
            epoch_loss += batch_loss.item()

            # metrics: top-1, top-3, top-5 error
            epoch_acc += accuracy(y_pred, y, topk=(1, 3, 5))

            # end of this batch
            batches += 1
            batch_end = time.time()
            if (i + 1) % verbose == 0:
                logging.info('\tBatch %d: Loss %.5f, Accuracy: Top-1 %.2f, Top-3 %.2f, Top-5 %.2f, Time %3.2f' %
                         (i + 1, epoch_loss / batches, epoch_acc[0] / batches, epoch_acc[1] / batches, epoch_acc[2] / batches, batch_end - batch_start))


    # end of validation
    end_time = time.time()

    # metrics for average
    epoch_loss /= batches
    epoch_acc /= batches

    # show information for this epoch
    logging.info('Valid: Loss %.5f,  Accuracy: Top-1 %.2f, Top-3 %.2f, Top-5 %.2f, Time %3.2f'%
                 (epoch_loss, epoch_acc[0], epoch_acc[1], epoch_acc[2], end_time - start_time))
    logging.info('')

    return epoch_loss


if __name__ == '__main__':
    main()
