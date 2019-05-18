import torch
import torch.nn as nn
import numpy as np
import os

from models import *
from utils import *
from data_loading import *
from patch_gan_model import Discriminator

## TODOS:
## 1. Dump SH in file
## 
## 
## Notes:
## 1. SH is not normalized
## 2. Face is normalized and denormalized - shall we not normalize in the first place?


# Enable WANDB Logging
WANDB_ENABLE = True

def predict_celeba(sfs_net_model, dl, train_epoch_num = 0,
                    use_cuda = False, out_folder = None, wandb = None, suffix = 'CelebA_Val', dump_all_images = False):
 
    # debugging flag to dump image
    fix_bix_dump = 0
    recon_loss  = nn.L1Loss() 

    if use_cuda:
        recon_loss  = recon_loss.cuda()

    tloss = 0 # Total loss
    rloss = 0 # Reconstruction loss

    for bix, data in enumerate(dl):
        face = data
        if use_cuda:
            face   = face.cuda()
        
        # predicted_face == reconstruction
        predicted_normal, predicted_albedo, predicted_sh, predicted_shading, predicted_face = sfs_net_model(face)
        
        if bix == fix_bix_dump or dump_all_images:
            # save predictions in log folder
            file_name = out_folder + suffix + '_' + str(train_epoch_num) + '_' + str(bix)
            # log images
            wandb_log_images(wandb, predicted_normal, None, suffix+' Predicted Normal', train_epoch_num, suffix+' Predicted Normal', path=file_name + '_predicted_normal.png')
            wandb_log_images(wandb, predicted_albedo, None, suffix +' Predicted Albedo', train_epoch_num, suffix+' Predicted Albedo', path=file_name + '_predicted_albedo.png')
            wandb_log_images(wandb, predicted_shading, None, suffix+' Predicted Shading', train_epoch_num, suffix+' Predicted Shading', path=file_name + '_predicted_shading.png', denormalize=False)
            wandb_log_images(wandb, predicted_face, None, suffix+' Predicted face', train_epoch_num, suffix+' Predicted face', path=file_name + '_predicted_face.png', denormalize=False)
            wandb_log_images(wandb, face, None, suffix+' Ground Truth', train_epoch_num, suffix+' Ground Truth', path=file_name + '_gt_face.png')

            # TODO:
            # Dump SH as CSV or TXT file
        
        # Loss computation
        # Reconstruction loss
        total_loss  = recon_loss(predicted_face, face)

        # Logging for display and debugging purposes
        tloss += total_loss.item()
    
    len_dl = len(dl)
    wandb.log({suffix+' Total loss': tloss/len_dl}, step=train_epoch_num)
            

    # return average loss over dataset
    return tloss / len_dl

def predict_sfsnet_gan(sfs_net_model, albedo_gen_model, albedo_dis_model, dl, gan_real_dl, train_epoch_num = 0,
                    use_cuda = False, out_folder = None, wandb = None, suffix = 'GAN Val'):
 
    # debugging flag to dump image
    fix_bix_dump = 0

    albedo_loss = nn.SmoothL1Loss() #nn.L1Loss()
    recon_loss  = nn.SmoothL1Loss() #nn.L1Loss() 
    gan_loss    = torch.nn.MSELoss()

    lamda_recon  = 0.5
    lamda_albedo = 0.5

    if use_cuda:
        albedo_loss = albedo_loss.cuda()
        recon_loss  = recon_loss.cuda()
        gan_loss    = gan_loss.cuda()

    tloss = 0 # Total loss
    aloss = 0 # Albedo loss
    rloss = 0 # Reconstruction loss
    ganloss = 0 # Gan loss
    disloss = 0 # Dis Loss

    real_gan_iter = iter(gan_real_dl)
    for bix, data in enumerate(dl):
        albedo, normal, mask, sh, face, label = data
        if use_cuda:
            albedo = albedo.cuda()
            normal = normal.cuda()
            mask   = mask.cuda()
            sh     = sh.cuda()
            face   = face.cuda()
            label  = label.cuda()

        # Apply Mask on input image
        # face = applyMask(face, mask)
        # predicted_face == reconstruction
        # predicted_normal, predicted_albedo, predicted_sh, predicted_shading, shading_residual, updated_shading, predicted_face = sfs_net_model(face)

        # Apply Mask on input image
        # face = applyMask(face, mask)
        predicted_normal, albedo_features, predicted_sh, shading_residual = sfs_net_model(face)
        # GAN Training
        valid = torch.ones(albedo.shape[0], requires_grad = False)
        fake = torch.zeros(albedo.shape[0], requires_grad = False)
        
        if use_cuda:
            valid = valid.cuda()
            fake = fake.cuda()
        # Get real sample
        real_data = next(real_gan_iter, None)
        if real_data is None:
            train_real_gan_iter = iter(gan_real_dl)
            real_data = next(train_real_gan_iter, None)
        
        # GAN loss
        fake_albedo = albedo_gen_model(albedo_features)
        pred_fake   = albedo_dis_model(fake_albedo)
        loss_GAN    = gan_loss(pred_fake, valid)
        # loss_pixel  = gan_loss_pixelwise(pred_fake, real_B)

        out_shading = get_shading(predicted_normal, predicted_sh)
        updated_shading = out_shading + shading_residual
        out_recon = reconstruct_image(updated_shading, fake_albedo)

        # albedo recon loss
        current_albedo_loss = albedo_loss(fake_albedo, albedo)
        current_recon_loss  = recon_loss(out_recon, face)

        total_loss = lamda_albedo * current_albedo_loss + lamda_recon * current_recon_loss + loss_GAN 

        # Real loss
        pred_real = albedo_dis_model(albedo)
        loss_real = gan_loss(pred_real, label)
        # Fake loss
        pred_fake = albedo_dis_model(fake_albedo.detach())
        loss_fake = gan_loss(pred_fake, fake)
        # Total loss
        loss_d = (loss_real + loss_fake) / 2

        # Logging for display and debugging purposes
        tloss += total_loss.item()
        # nloss += current_normal_loss.item()
        aloss += current_albedo_loss.item()
        # shloss += current_sh_loss.item()
        rloss += current_recon_loss.item()
        ganloss += loss_GAN.item()
        disloss += loss_d.item()


        if bix == fix_bix_dump:
            # save predictions in log folder
            file_name = out_folder + suffix + '_' + str(train_epoch_num) + '_' + str(fix_bix_dump)
            # log images
            # save_p_normal = get_normal_in_range(predicted_normal)
            save_gt_normal = get_normal_in_range(normal)
            save_p_normal = predicted_normal

            wandb_log_images(wandb, save_p_normal, mask, suffix+' Predicted Normal', train_epoch_num, suffix+' Predicted Normal', path=file_name + '_predicted_normal.png')
            wandb_log_images(wandb, fake_albedo, mask, suffix +' Predicted Albedo', train_epoch_num, suffix+' Predicted Albedo', path=file_name + '_predicted_albedo.png')
            wandb_log_images(wandb, out_shading, mask, suffix+' Predicted Shading', train_epoch_num, suffix+' Predicted Shading', path=file_name + '_predicted_shading.png', denormalize=False)
            wandb_log_images(wandb, shading_residual, mask, suffix+' Predicted Shading Residual', train_epoch_num, suffix+' Predicted Shading Residual', path=file_name + '_predicted_residual_shading.png', denormalize=False)
            wandb_log_images(wandb, updated_shading, mask, suffix+' Predicted Updated Shading', train_epoch_num, suffix+' Predicted Updated Shading', path=file_name + '_predicted_updated_shading.png', denormalize=False)
            wandb_log_images(wandb, out_recon, mask, suffix+' Predicted face', train_epoch_num, suffix+' Predicted face', path=file_name + '_predicted_face.png', denormalize=False)
            wandb_log_images(wandb, face, mask, suffix+' Ground Truth', train_epoch_num, suffix+' Ground Truth', path=file_name + '_gt_face.png')
            wandb_log_images(wandb, save_gt_normal, mask, suffix+' Ground Truth Normal', train_epoch_num, suffix+' Ground Normal', path=file_name + '_gt_normal.png')
            wandb_log_images(wandb, albedo, mask, suffix+' Ground Truth Albedo', train_epoch_num, suffix+' Ground Albedo', path=file_name + '_gt_albedo.png')
            # Get face with real SH
            real_sh_face = sfs_net_model.get_face(sh, predicted_normal, fake_albedo)
            wandb_log_images(wandb, real_sh_face, mask, 'Val Real SH Predicted Face', train_epoch_num, 'Val Real SH Predicted Face', path=file_name + '_real_sh_face.png')
            syn_face     = sfs_net_model.get_face(sh, normal, albedo)
            wandb_log_images(wandb, syn_face, mask, 'Val Real SH GT Face', train_epoch_num, 'Val Real SH GT Face', path=file_name + '_syn_gt_face.png')

            # TODO:
            # Dump SH as CSV or TXT file
    
    len_dl = len(dl)
    wandb.log({suffix+' Total loss': tloss/len_dl, suffix+'Albedo loss': aloss/len_dl, suffix + 'Recon loss': rloss/len_dl, suffix + 'Gen Loss': ganloss/len_dl, suffix + 'Dis Loss': disloss/len_dl}, step=train_epoch_num)
            
    # return average loss over dataset
    return tloss / len_dl, aloss / len_dl, rloss / len_dl, ganloss/len_dl, disloss / len_dl

def predict_sfsnet(sfs_net_model, albedo_gen_model, dl, train_epoch_num = 0,
                    use_cuda = False, out_folder = None, wandb = None, suffix = 'Val'):
 
    # debugging flag to dump image
    fix_bix_dump = 0

    albedo_loss = nn.SmoothL1Loss() #nn.L1Loss()
    recon_loss  = nn.SmoothL1Loss() #nn.L1Loss() 

    lamda_recon  = 1
    lamda_albedo = 1

    if use_cuda:
        albedo_loss = albedo_loss.cuda()
        recon_loss  = recon_loss.cuda()

    tloss = 0 # Total loss
    aloss = 0 # Albedo loss
    rloss = 0 # Reconstruction loss

    for bix, data in enumerate(dl):
        albedo, normal, mask, sh, face, label = data
        if use_cuda:
            albedo = albedo.cuda()
            normal = normal.cuda()
            mask   = mask.cuda()
            sh     = sh.cuda()
            face   = face.cuda()
            label  = label.cuda()

        # Apply Mask on input image
        # face = applyMask(face, mask)
        # predicted_face == reconstruction
        # predicted_normal, predicted_albedo, predicted_sh, predicted_shading, shading_residual, updated_shading, predicted_face = sfs_net_model(face)

        # Apply Mask on input image
        # face = applyMask(face, mask)
        predicted_normal, albedo_features, predicted_sh, shading_residual = sfs_net_model(face)

        fake_albedo = albedo_gen_model(albedo_features)

        out_shading = get_shading(predicted_normal, predicted_sh)
        updated_shading = out_shading + shading_residual
        out_recon = reconstruct_image(updated_shading, fake_albedo)

        # albedo recon loss
        current_albedo_loss = albedo_loss(fake_albedo, albedo)
        current_recon_loss  = recon_loss(out_recon, face)

        total_loss = lamda_albedo * current_albedo_loss + lamda_recon * current_recon_loss 

        # Logging for display and debugging purposes
        tloss += total_loss.item()
        # nloss += current_normal_loss.item()
        aloss += current_albedo_loss.item()
        # shloss += current_sh_loss.item()
        rloss += current_recon_loss.item()

        if bix == fix_bix_dump:
            # save predictions in log folder
            file_name = out_folder + suffix + '_' + str(train_epoch_num) + '_' + str(fix_bix_dump)
            # log images
            # save_p_normal = get_normal_in_range(predicted_normal)
            save_gt_normal = get_normal_in_range(normal)
            save_p_normal = predicted_normal

            wandb_log_images(wandb, save_p_normal, mask, suffix+' Predicted Normal', train_epoch_num, suffix+' Predicted Normal', path=file_name + '_predicted_normal.png')
            wandb_log_images(wandb, fake_albedo, mask, suffix +' Predicted Albedo', train_epoch_num, suffix+' Predicted Albedo', path=file_name + '_predicted_albedo.png')
            wandb_log_images(wandb, out_shading, mask, suffix+' Predicted Shading', train_epoch_num, suffix+' Predicted Shading', path=file_name + '_predicted_shading.png', denormalize=False)
            wandb_log_images(wandb, shading_residual, mask, suffix+' Predicted Shading Residual', train_epoch_num, suffix+' Predicted Shading Residual', path=file_name + '_predicted_residual_shading.png', denormalize=False)
            wandb_log_images(wandb, updated_shading, mask, suffix+' Predicted Updated Shading', train_epoch_num, suffix+' Predicted Updated Shading', path=file_name + '_predicted_updated_shading.png', denormalize=False)
            wandb_log_images(wandb, out_recon, mask, suffix+' Predicted face', train_epoch_num, suffix+' Predicted face', path=file_name + '_predicted_face.png', denormalize=False)
            wandb_log_images(wandb, face, mask, suffix+' Ground Truth', train_epoch_num, suffix+' Ground Truth', path=file_name + '_gt_face.png')
            wandb_log_images(wandb, save_gt_normal, mask, suffix+' Ground Truth Normal', train_epoch_num, suffix+' Ground Normal', path=file_name + '_gt_normal.png')
            wandb_log_images(wandb, albedo, mask, suffix+' Ground Truth Albedo', train_epoch_num, suffix+' Ground Albedo', path=file_name + '_gt_albedo.png')
            # Get face with real SH
            real_sh_face = sfs_net_model.get_face(sh, predicted_normal, fake_albedo)
            wandb_log_images(wandb, real_sh_face, mask, 'Val Real SH Predicted Face', train_epoch_num, 'Val Real SH Predicted Face', path=file_name + '_real_sh_face.png')
            syn_face     = sfs_net_model.get_face(sh, normal, albedo)
            wandb_log_images(wandb, syn_face, mask, 'Val Real SH GT Face', train_epoch_num, 'Val Real SH GT Face', path=file_name + '_syn_gt_face.png')

            # TODO:
            # Dump SH as CSV or TXT file
    
    len_dl = len(dl)
    wandb.log({suffix+' Total loss': tloss/len_dl, suffix+'Albedo loss': aloss/len_dl, suffix + 'Recon loss': rloss/len_dl}, step=train_epoch_num)
            
    # return average loss over dataset
    return tloss / len_dl, aloss / len_dl, rloss / len_dl

def gan_based_train(sfs_net_model, albedo_gen_model, albedo_dis_model, syn_data, celeba_data=None, read_first=None,
          batch_size = 10, num_epochs = 10, log_path = './results/metadata/', use_cuda=False, wandb=None,
          lr = 0.01, wt_decay=0.005):

    # data processing
    syn_train_csv = syn_data + '/train.csv'
    syn_test_csv  = syn_data + '/test.csv'
    
    celeba_train_csv = None
    celeba_test_csv = None
    if celeba_data is not None:
        celeba_train_csv = celeba_data + '/train.csv'
        celeba_test_csv = celeba_data + '/test.csv'

    # Load Synthetic dataset
    train_dataset, val_dataset = get_sfsnet_dataset(syn_dir=syn_data+'train/', read_from_csv=syn_train_csv, read_celeba_csv=celeba_train_csv, read_first=read_first, validation_split=2)
    test_dataset, _ = get_sfsnet_dataset(syn_dir=syn_data+'test/', read_from_csv=None, read_celeba_csv=celeba_test_csv, read_first=100, validation_split=0)

    syn_train_dl  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    syn_val_dl    = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    syn_test_dl   = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    gan_real_train_dataset, gan_real_val_dataset = get_sfsnet_dataset(syn_dir=syn_data+'train/', read_from_csv=syn_train_csv, read_celeba_csv=None, read_first=read_first, validation_split=2)
    gan_real_test_dataset, _ = get_sfsnet_dataset(syn_dir=syn_data+'test/', read_from_csv=syn_test_csv, read_celeba_csv=None, read_first=read_first, validation_split=0)
    
    gan_real_train_dl  = DataLoader(gan_real_train_dataset, batch_size=batch_size, shuffle=True)
    train_real_gan_iter = iter(gan_real_train_dl)

    gan_real_val_dl     = DataLoader(gan_real_val_dataset, batch_size=batch_size, shuffle=True)
    gan_real_test_dl  = DataLoader(gan_real_test_dataset, batch_size=batch_size, shuffle=True)

    print('Synthetic dataset: Train data: ', len(syn_train_dl), ' Val data: ', len(syn_val_dl), ' Test data: ', len(syn_test_dl))

    model_checkpoint_dir = log_path + 'checkpoints/'
    out_images_dir       = log_path + 'out_images/'
    out_syn_images_dir   = out_images_dir

    os.system('mkdir -p {}'.format(model_checkpoint_dir))
    os.system('mkdir -p {}'.format(out_syn_images_dir + 'train/'))
    os.system('mkdir -p {}'.format(out_syn_images_dir + 'val/'))
    os.system('mkdir -p {}'.format(out_syn_images_dir + 'test/'))

    # Collect model parameters
    model_parameters = sfs_net_model.parameters()
    optimizer = torch.optim.Adam(model_parameters, lr=lr) #, weight_decay=wt_decay)
    albedo_loss = nn.SmoothL1Loss() #nn.L1Loss()
    recon_loss  = nn.SmoothL1Loss() #nn.L1Loss() 
    gan_loss = torch.nn.MSELoss()

    # Collect and initialize gen-dis optimizers
    g_optimizer = torch.optim.Adam(albedo_gen_model.parameters(), lr=lr)
    d_optimizer = torch.optim.Adam(albedo_dis_model.parameters(), lr=lr)
    
    lamda_recon  = 1
    lamda_albedo = 10

    if use_cuda:
        albedo_loss = albedo_loss.cuda()
        recon_loss  = recon_loss.cuda()
        gan_loss    = gan_loss.cuda()

    syn_train_len    = len(syn_train_dl)

    for epoch in range(1, num_epochs+1):
        tloss = 0 # Total loss
        aloss = 0 # Albedo loss
        rloss = 0 # Reconstruction loss
        ganloss = 0 # Gan Loss
        disloss = 0 # Dis Loss

        for bix, data in enumerate(syn_train_dl):
            albedo, normal, mask, sh, face, label = data
            if use_cuda:
                albedo = albedo.cuda()
                normal = normal.cuda()
                mask   = mask.cuda()
                sh     = sh.cuda()
                face   = face.cuda()
                label  = label.cuda()
           
            # Apply Mask on input image
            # GAN Training
            valid = torch.ones(albedo.shape[0], requires_grad = False)
            fake = torch.zeros(albedo.shape[0], requires_grad = False)
            
            if use_cuda:
                valid = valid.cuda()
                fake = fake.cuda()
            # Train Albedo Generator
            g_optimizer.zero_grad()
            
            # GAN loss
            predicted_normal, albedo_features, predicted_sh, shading_residual = sfs_net_model(face)
            fake_albedo = albedo_gen_model(albedo_features)
            pred_fake   = albedo_dis_model(fake_albedo)
            # print(pred_fake.shape, valid.shape)
            loss_GAN    = gan_loss(pred_fake, valid)
            # loss_pixel  = gan_loss_pixelwise(pred_fake, real_B)

            out_shading = get_shading(predicted_normal, predicted_sh)
            updated_shading = out_shading + shading_residual
            out_recon = reconstruct_image(updated_shading, fake_albedo)

            # albedo recon loss
            current_albedo_loss = albedo_loss(fake_albedo, albedo)
            current_recon_loss  = recon_loss(out_recon, face)

            total_loss = lamda_albedo * current_albedo_loss + loss_GAN 
            total_loss.backward()
            g_optimizer.step()

            # Training Albedo Discriminator
            d_optimizer.zero_grad()
            # Real loss
            pred_real = albedo_dis_model(albedo)
            loss_real = gan_loss(pred_real, label)
            # Fake loss
            pred_fake = albedo_dis_model(fake_albedo.detach())
            loss_fake = gan_loss(pred_fake, fake)
            # Total loss
            loss_d = (loss_real + loss_fake) / 2

            loss_d.backward()
            d_optimizer.step()

            # Logging for display and debugging purposes
            tloss += total_loss.item()
            # nloss += current_normal_loss.item()
            aloss += current_albedo_loss.item()
            # shloss += current_sh_loss.item()
            rloss += current_recon_loss.item()
            ganloss += loss_GAN.item()
            disloss += loss_d.item()

        print('Epoch: {} - Total Loss: {}, Albedo Loss: {}, Generator Loss: {}, Discriminator Loss: {}'.format(epoch, tloss, aloss, ganloss, disloss))
        log_prefix = 'GAN Training '

        if epoch % 1 == 0:
            print('Training set results: Total Loss: {}, Albedo Loss: {}, Generator Loss: {}, Discriminator Loss: {}, '.format(tloss / syn_train_len, \
                   aloss / syn_train_len, ganloss / syn_train_len, disloss / syn_train_len))
            # Log training info
            wandb.log({log_prefix + 'Train Total loss': tloss/syn_train_len, log_prefix + 'Train Albedo loss': aloss/syn_train_len, log_prefix + 'Train Gen loss': ganloss/syn_train_len, log_prefix + 'Train Dis loss': disloss/syn_train_len})
            
            wandb.log({log_prefix + 'Acc Train Total loss': tloss, log_prefix + 'Acc Train Albedo loss': aloss, log_prefix + 'Acc Train Gen loss': ganloss, log_prefix + 'Acc Train Dis loss': disloss})
            
            # Log images in wandb
            file_name = out_syn_images_dir + 'train/' +  'train_' + str(epoch)
            # save_p_normal = get_normal_in_range(predicted_normal)
            save_gt_normal = get_normal_in_range(normal)
            save_p_normal = predicted_normal
            wandb_log_images(wandb, save_p_normal, mask, 'Train Predicted Normal', epoch, 'Train Predicted Normal', path=file_name + '_predicted_normal.png')
            wandb_log_images(wandb, fake_albedo, mask, 'Train Predicted Albedo', epoch, 'Train Predicted Albedo', path=file_name + '_predicted_albedo.png')
            wandb_log_images(wandb, out_shading, mask, 'Train Predicted Shading', epoch, 'Train Predicted Shading', path=file_name + '_predicted_shading.png', denormalize=False)
            wandb_log_images(wandb, shading_residual, mask, 'Train Predicted Shading Residual', epoch, 'Train Predicted Shading Residual', path=file_name + '_predicted_residual_shading.png', denormalize=False)
            wandb_log_images(wandb, updated_shading, mask, 'Train Predicted Updated Shading', epoch, 'Train Predicted Updated Shading', path=file_name + '_predicted_updated_shading.png', denormalize=False)
            wandb_log_images(wandb, out_recon, mask, 'Train Recon', epoch, 'Train Recon', path=file_name + '_predicted_face.png')
            wandb_log_images(wandb, face, mask, 'Train Ground Truth', epoch, 'Train Ground Truth', path=file_name + '_gt_face.png')
            wandb_log_images(wandb, save_gt_normal, mask, 'Train Ground Truth Normal', epoch, 'Train Ground Truth Normal', path=file_name + '_gt_normal.png')
            wandb_log_images(wandb, albedo, mask, 'Train Ground Truth Albedo', epoch, 'Train Ground Truth Albedo', path=file_name + '_gt_albedo.png')
            # Get face with real_sh, predicted normal and albedo for debugging
            real_sh_face = sfs_net_model.get_face(sh, predicted_normal, fake_albedo)
            syn_face     = sfs_net_model.get_face(sh, normal, albedo)
            wandb_log_images(wandb, real_sh_face, mask, 'Train Real SH Predicted Face', epoch, 'Train Real SH Predicted Face', path=file_name + '_real_sh_face.png')
            wandb_log_images(wandb, syn_face, mask, 'Train Real SH GT Face', epoch, 'Train Real SH GT Face', path=file_name + '_syn_gt_face.png')

            v_total, v_albedo, v_recon, v_gloss, v_dloss = predict_sfsnet_gan(sfs_net_model, albedo_gen_model, albedo_dis_model, syn_val_dl, gan_real_val_dl, train_epoch_num=epoch, use_cuda=use_cuda,
                                                                         out_folder=out_syn_images_dir+'/val/', wandb=wandb, suffix='GAN Val')
            # wandb.log({log_prefix + 'Val Total loss': v_total, log_prefix + 'Val Albedo loss': v_albedo, log_prefix + 'Val Recon loss': v_recon})
            

            print('Val set results: Total Loss: {}, Albedo Loss: {},  Recon Loss: {}, Gan Loss: {}, Dis Loss: {}'.format(v_total, v_albedo, v_recon, v_gloss, v_dloss))
            
            # Model saving
            torch.save(sfs_net_model.state_dict(), model_checkpoint_dir + 'sfs_net_model.pkl')
            torch.save(albedo_gen_model.state_dict(), model_checkpoint_dir + 'albedo_gen_model.pkl')
            torch.save(albedo_dis_model.state_dict(), model_checkpoint_dir + 'albedo_dis_model.pkl')
        if epoch % 5 == 0:
            t_total, t_albedo, t_recon, t_gloss, t_dloss = predict_sfsnet_gan(sfs_net_model, albedo_gen_model, albedo_dis_model, syn_test_dl, gan_real_test_dl, train_epoch_num=epoch, use_cuda=use_cuda, 
                                                                        out_folder=out_syn_images_dir + '/test/', wandb=wandb, suffix='GAN Test')

            print('Test-set results: Total Loss: {}, Albedo Loss: {}, Gan Loss: {}, Dis Loss: {} \n'.format(t_total, t_albedo, t_gloss, t_dloss))

def train(sfs_net_model, albedo_gen_model, albedo_dis_model, syn_data, celeba_data=None, read_first=None,
          batch_size = 10, num_epochs = 10, log_path = './results/metadata/', use_cuda=False, wandb=None,
          lr = 0.01, wt_decay=0.005):

    # data processing
    syn_train_csv = syn_data + '/train.csv'
    syn_test_csv  = syn_data + '/test.csv'
    
    celeba_train_csv = None
    celeba_test_csv = None
    if celeba_data is not None:
        celeba_train_csv = celeba_data + '/train.csv'
        celeba_test_csv = celeba_data + '/test.csv'

    # Load Synthetic dataset
    train_dataset, val_dataset = get_sfsnet_dataset(syn_dir=syn_data+'train/', read_from_csv=syn_train_csv, read_celeba_csv=celeba_train_csv, read_first=read_first, validation_split=2)
    test_dataset, _ = get_sfsnet_dataset(syn_dir=syn_data+'test/', read_from_csv=syn_test_csv, read_celeba_csv=celeba_test_csv, read_first=100, validation_split=0)

    syn_train_dl  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    syn_val_dl    = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    syn_test_dl   = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print('Synthetic dataset: Train data: ', len(syn_train_dl), ' Val data: ', len(syn_val_dl), ' Test data: ', len(syn_test_dl))

    model_checkpoint_dir = log_path + 'checkpoints/'
    out_images_dir       = log_path + 'out_images/'
    out_syn_images_dir   = out_images_dir

    os.system('mkdir -p {}'.format(model_checkpoint_dir))
    os.system('mkdir -p {}'.format(out_syn_images_dir + 'train/'))
    os.system('mkdir -p {}'.format(out_syn_images_dir + 'val/'))
    os.system('mkdir -p {}'.format(out_syn_images_dir + 'test/'))

    # Collect model parameters
    model_parameters = sfs_net_model.parameters()
    optimizer = torch.optim.Adam(model_parameters, lr=lr) #, weight_decay=wt_decay)
    albedo_loss = nn.SmoothL1Loss() #nn.L1Loss()
    recon_loss  = nn.SmoothL1Loss() #nn.L1Loss() 
    
    if use_cuda:
        albedo_loss = albedo_loss.cuda()
        recon_loss  = recon_loss.cuda()

    lamda_recon  = 1
    lamda_albedo = 1

    if use_cuda:
        albedo_loss = albedo_loss.cuda()
        recon_loss  = recon_loss.cuda()

    syn_train_len    = len(syn_train_dl)

    for epoch in range(1, num_epochs+1):
        tloss = 0 # Total loss
        aloss = 0 # Albedo loss
        rloss = 0 # Reconstruction loss

        for bix, data in enumerate(syn_train_dl):
            albedo, normal, mask, sh, face, _ = data
            if use_cuda:
                albedo = albedo.cuda()
                normal = normal.cuda()
                mask   = mask.cuda()
                sh     = sh.cuda()
                face   = face.cuda()
           
            # Apply Mask on input image
            # face = applyMask(face, mask)
            predicted_normal, albedo_features, predicted_sh, shading_residual = sfs_net_model(face)

            # GAN loss
            fake_albedo = albedo_gen_model(albedo_features)

            out_shading = get_shading(predicted_normal, predicted_sh)
            updated_shading = out_shading + shading_residual
            out_recon = reconstruct_image(updated_shading, fake_albedo)

            # albedo recon loss
            current_albedo_loss = albedo_loss(fake_albedo, albedo)
            current_recon_loss  = recon_loss(out_recon, face)

            total_loss = lamda_albedo * current_albedo_loss + lamda_recon * current_recon_loss
            total_loss.backward()
            optimizer.step()

            # Logging for display and debugging purposes
            tloss += total_loss.item()
            # nloss += current_normal_loss.item()
            aloss += current_albedo_loss.item()
            # shloss += current_sh_loss.item()
            rloss += current_recon_loss.item()

        print('Epoch: {} - Total Loss: {}, Albedo Loss: {}, Recon Loss: {}'.format(epoch, tloss, aloss, rloss))
        log_prefix = 'Syn Data'
        if celeba_data is not None:
            log_prefix = 'Mix Data '

        if epoch % 1 == 0:
            print('Training set results: Total Loss: {}, Albedo Loss: {}, Recon Loss: {}'.format(tloss / syn_train_len, \
                   aloss / syn_train_len, rloss / syn_train_len))
            # Log training info
            wandb.log({log_prefix + 'Train Total loss': tloss/syn_train_len, log_prefix + 'Train Albedo loss': aloss/syn_train_len, log_prefix + 'Train Recon loss': rloss/syn_train_len})
            
            # Log images in wandb
            file_name = out_syn_images_dir + 'train/' +  'train_' + str(epoch)
            # save_p_normal = get_normal_in_range(predicted_normal)
            save_gt_normal = get_normal_in_range(normal)
            save_p_normal = predicted_normal
            wandb_log_images(wandb, save_p_normal, mask, 'Train Predicted Normal', epoch, 'Train Predicted Normal', path=file_name + '_predicted_normal.png')
            wandb_log_images(wandb, fake_albedo, mask, 'Train Predicted Albedo', epoch, 'Train Predicted Albedo', path=file_name + '_predicted_albedo.png')
            wandb_log_images(wandb, out_shading, mask, 'Train Predicted Shading', epoch, 'Train Predicted Shading', path=file_name + '_predicted_shading.png', denormalize=False)
            wandb_log_images(wandb, shading_residual, mask, 'Train Predicted Shading Residual', epoch, 'Train Predicted Shading Residual', path=file_name + '_predicted_residual_shading.png', denormalize=False)
            wandb_log_images(wandb, updated_shading, mask, 'Train Predicted Updated Shading', epoch, 'Train Predicted Updated Shading', path=file_name + '_predicted_updated_shading.png', denormalize=False)
            wandb_log_images(wandb, out_recon, mask, 'Train Recon', epoch, 'Train Recon', path=file_name + '_predicted_face.png')
            wandb_log_images(wandb, face, mask, 'Train Ground Truth', epoch, 'Train Ground Truth', path=file_name + '_gt_face.png')
            wandb_log_images(wandb, save_gt_normal, mask, 'Train Ground Truth Normal', epoch, 'Train Ground Truth Normal', path=file_name + '_gt_normal.png')
            wandb_log_images(wandb, albedo, mask, 'Train Ground Truth Albedo', epoch, 'Train Ground Truth Albedo', path=file_name + '_gt_albedo.png')
            # Get face with real_sh, predicted normal and albedo for debugging
            real_sh_face = sfs_net_model.get_face(sh, predicted_normal, fake_albedo)
            syn_face     = sfs_net_model.get_face(sh, normal, albedo)
            wandb_log_images(wandb, real_sh_face, mask, 'Train Real SH Predicted Face', epoch, 'Train Real SH Predicted Face', path=file_name + '_real_sh_face.png')
            wandb_log_images(wandb, syn_face, mask, 'Train Real SH GT Face', epoch, 'Train Real SH GT Face', path=file_name + '_syn_gt_face.png')

            v_total, v_albedo, v_recon = predict_sfsnet(sfs_net_model, albedo_gen_model, syn_val_dl, train_epoch_num=epoch, use_cuda=use_cuda,
                                                                         out_folder=out_syn_images_dir+'/val/', wandb=wandb)
            # wandb.log({log_prefix + 'Val Total loss': v_total, log_prefix + 'Val Albedo loss': v_albedo, log_prefix + 'Val Recon loss': v_recon})
            print('Val set results: Total Loss: {}, Albedo Loss: {},  Recon Loss: {}'.format(v_total, v_albedo, v_recon))
            
            # Model saving
            torch.save(sfs_net_model.state_dict(), model_checkpoint_dir + 'sfs_net_model.pkl')
        if epoch % 5 == 0:
            t_total, t_albedo, t_recon = predict_sfsnet(sfs_net_model, albedo_gen_model, syn_test_dl, train_epoch_num=epoch, use_cuda=use_cuda, 
                                                                        out_folder=out_syn_images_dir + '/test/', wandb=wandb, suffix='Test')

            # wandb.log({log_prefix+'Test Total loss': t_total, log_prefix+'Test Albedo loss': t_albedo, log_prefix+'Test Recon loss': t_recon})

            print('Test-set results: Total Loss: {}, Albedo Loss: {}, Recon Loss: {}\n'.format(t_total, t_albedo, t_recon))


def train_with_shading_loss(sfs_net_model, syn_data, celeba_data=None, read_first=None,
          batch_size = 10, num_epochs = 10, log_path = './results/metadata/', use_cuda=False, wandb=None,
          lr = 0.01, wt_decay=0.005):

    # data processing
    syn_train_csv = syn_data + '/train.csv'
    syn_test_csv  = syn_data + '/test.csv'
    
    celeba_train_csv = None
    celeba_test_csv = None
    if celeba_data is not None:
        celeba_train_csv = celeba_data + '/train.csv'
        celeba_test_csv = celeba_data + '/test.csv'

    # Load Synthetic dataset
    train_dataset, val_dataset = get_sfsnet_dataset(syn_dir=syn_data+'train/', read_from_csv=syn_train_csv, read_celeba_csv=celeba_train_csv, read_first=read_first, validation_split=2)
    test_dataset, _ = get_sfsnet_dataset(syn_dir=syn_data+'test/', read_from_csv=syn_test_csv, read_celeba_csv=celeba_test_csv, read_first=100, validation_split=0)

    syn_train_dl  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    syn_val_dl    = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    syn_test_dl   = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print('Synthetic dataset: Train data: ', len(syn_train_dl), ' Val data: ', len(syn_val_dl), ' Test data: ', len(syn_test_dl))

    model_checkpoint_dir = log_path + 'checkpoints/'
    out_images_dir       = log_path + 'out_images/'
    out_syn_images_dir   = out_images_dir

    os.system('mkdir -p {}'.format(model_checkpoint_dir))
    os.system('mkdir -p {}'.format(out_syn_images_dir + 'train/'))
    os.system('mkdir -p {}'.format(out_syn_images_dir + 'val/'))
    os.system('mkdir -p {}'.format(out_syn_images_dir + 'test/'))

    # Collect model parameters
    model_parameters = sfs_net_model.parameters()
    optimizer = torch.optim.Adam(model_parameters, lr=lr, weight_decay=wt_decay)
    albedo_loss = nn.SmoothL1Loss() #nn.L1Loss()
    recon_loss  = nn.SmoothL1Loss() #nn.L1Loss() 
    shading_loss = nn.SmoothL1Loss()

    if use_cuda:
        albedo_loss = albedo_loss.cuda()
        recon_loss  = recon_loss.cuda()
        shading_loss = shading_loss.cuda()

    lamda_recon  = 1 #0.3
    lamda_albedo = 1 #0.5
    lamda_shading = 1 #0.7

    syn_train_len    = len(syn_train_dl)

    for epoch in range(1, num_epochs+1):
        tloss = 0 # Total loss
        aloss = 0 # Albedo loss
        rloss = 0 # Reconstruction loss
        shloss = 0 # Shading loss

        for bix, data in enumerate(syn_train_dl):
            albedo, normal, mask, sh, face = data
            if use_cuda:
                albedo = albedo.cuda()
                normal = normal.cuda()
                mask   = mask.cuda()
                sh     = sh.cuda()
                face   = face.cuda()
           
            # Apply Mask on input image
            # face = applyMask(face, mask)
            predicted_normal, predicted_albedo, predicted_sh, out_shading, shading_residual, updated_shading, out_recon = sfs_net_model(face)
            
            # Loss computation
            # Normal loss
            # current_normal_loss = normal_loss(predicted_normal, normal)
            # Albedo loss
            current_albedo_loss = albedo_loss(predicted_albedo, albedo)
            # SH loss
            # current_sh_loss     = sh_loss(predicted_sh, sh)
    
            # corrected shading should be close to predicted shading
            gt_shading = get_shading(normal, sh)
            # current_shading_loss = shading_loss(updated_shading, gt_shading)

            # Reconstruction loss
            # Edge case: Shading generation requires denormalized normal and sh
            # Hence, denormalizing face here
            current_recon_loss  = recon_loss(out_recon, face)

            total_loss = lamda_albedo * current_albedo_loss + lamda_recon * current_recon_loss # + \
            #                lamda_shading * current_shading_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Logging for display and debugging purposes
            tloss += total_loss.item()
            # nloss += current_normal_loss.item()
            aloss += current_albedo_loss.item()
            # shloss += current_sh_loss.item()
            rloss += current_recon_loss.item()
            shloss += current_shading_loss.item()

        print('Epoch: {} - Total Loss: {}, Albedo Loss: {}, Recon Loss: {}'.format(epoch, tloss, aloss, rloss))
        log_prefix = 'Syn Data'
        if celeba_data is not None:
            log_prefix = 'Mix Data '

        if epoch % 1 == 0:
            print('Training set results: Total Loss: {}, Albedo Loss: {}, Recon Loss: {}'.format(tloss / syn_train_len, \
                   aloss / syn_train_len, rloss / syn_train_len))
            # Log training info
            wandb.log({log_prefix + 'Train Total loss': tloss/syn_train_len, log_prefix + 'Train Albedo loss': aloss/syn_train_len, log_prefix + 'Train Recon loss': rloss/syn_train_len, log_prefix+'Train Shading loss': shloss/syn_train_len})
            
            # Log images in wandb
            file_name = out_syn_images_dir + 'train/' +  'train_' + str(epoch)
            # save_p_normal = get_normal_in_range(predicted_normal)
            save_gt_normal = get_normal_in_range(normal)
            save_p_normal = predicted_normal
            wandb_log_images(wandb, save_p_normal, mask, 'Train Predicted Normal', epoch, 'Train Predicted Normal', path=file_name + '_predicted_normal.png')
            wandb_log_images(wandb, predicted_albedo, mask, 'Train Predicted Albedo', epoch, 'Train Predicted Albedo', path=file_name + '_predicted_albedo.png')
            wandb_log_images(wandb, out_shading, mask, 'Train Predicted Shading', epoch, 'Train Predicted Shading', path=file_name + '_predicted_shading.png', denormalize=False)
            wandb_log_images(wandb, shading_residual, mask, 'Train Predicted Shading Residual', epoch, 'Train Predicted Shading Residual', path=file_name + '_predicted_residual_shading.png', denormalize=False)
            wandb_log_images(wandb, updated_shading, mask, 'Train Predicted Updated Shading', epoch, 'Train Predicted Updated Shading', path=file_name + '_predicted_updated_shading.png', denormalize=False)
            wandb_log_images(wandb, out_recon, mask, 'Train Recon', epoch, 'Train Recon', path=file_name + '_predicted_face.png')
            wandb_log_images(wandb, face, mask, 'Train Ground Truth', epoch, 'Train Ground Truth', path=file_name + '_gt_face.png')
            wandb_log_images(wandb, save_gt_normal, mask, 'Train Ground Truth Normal', epoch, 'Train Ground Truth Normal', path=file_name + '_gt_normal.png')
            wandb_log_images(wandb, albedo, mask, 'Train Ground Truth Albedo', epoch, 'Train Ground Truth Albedo', path=file_name + '_gt_albedo.png')
            # Get face with real_sh, predicted normal and albedo for debugging
            real_sh_face = sfs_net_model.get_face(sh, predicted_normal, predicted_albedo)
            syn_face     = sfs_net_model.get_face(sh, normal, albedo)
            wandb_log_images(wandb, real_sh_face, mask, 'Train Real SH Predicted Face', epoch, 'Train Real SH Predicted Face', path=file_name + '_real_sh_face.png')
            wandb_log_images(wandb, syn_face, mask, 'Train Real SH GT Face', epoch, 'Train Real SH GT Face', path=file_name + '_syn_gt_face.png')

            v_total, v_albedo, v_recon = predict_sfsnet(sfs_net_model, albedo_gen_model, syn_val_dl, train_epoch_num=epoch, use_cuda=use_cuda,
                                                                         out_folder=out_syn_images_dir+'/val/', wandb=wandb)
            wandb.log({log_prefix + 'Val Total loss': v_total, log_prefix + 'Val Albedo loss': v_albedo, log_prefix + 'Val Recon loss': v_recon})
            

            print('Val set results: Total Loss: {}, Albedo Loss: {},  Recon Loss: {}'.format(v_total, v_albedo, v_recon))
            
            # Model saving
            torch.save(sfs_net_model.state_dict(), model_checkpoint_dir + 'sfs_net_model.pkl')
        if epoch % 5 == 0:
            t_total, t_albedo, t_recon = predict_sfsnet(sfs_net_model, albedo_gen_model, syn_test_dl, train_epoch_num=epoch, use_cuda=use_cuda, 
                                                                        out_folder=out_syn_images_dir + '/test/', wandb=wandb, suffix='Test')

            wandb.log({log_prefix+'Test Total loss': t_total, log_prefix+'Test Albedo loss': t_albedo, log_prefix+'Test Recon loss': t_recon})

            print('Test-set results: Total Loss: {}, Albedo Loss: {}, Recon Loss: {}\n'.format(t_total, t_albedo, t_recon))
