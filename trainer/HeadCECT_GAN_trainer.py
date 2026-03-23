#!/usr/bin/env python3
"""Full-feature CTA2 trainer with parity to CycTrainer.

Supports optional registration (`regist`), bidirectional cycle branch (`bidirect`),
replay buffers, LR schedulers, metrics (MAE/PSNR/SSIM), deformation saving, and
consistent logging/checkpointing.
"""

import os
import itertools
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import RandomAffine, ToPILImage
import inspect

import warnings
from .utils import Resize, ToTensor, Logger, ReplayBuffer, LambdaLR, smooothing_loss
from .datasets import ImageDataset, ValDataset
from Model.HeadCECT_GAN import HeadCECTGANGenerator, HeadCECTGANDiscriminator
from .reg import Reg
from .transformer import Transformer_2D
from skimage.metrics import structural_similarity as ssim_metric
from PIL import Image


def weighted_l1_loss(pred, target, weight=5.0, threshold=0.0):
    """
    L1 loss with higher weight for pixels > threshold (bright regions).
    Assumes input range [-1, 1].
    """
    abs_diff = torch.abs(pred - target)
    mask = (target > threshold).float()
    w = 1.0 + mask * weight
    return torch.mean(abs_diff * w)


class HeadCECTGAN_Trainer:
    def __init__(self, config):
        super().__init__()
        self.config = config
        device = torch.device('cuda' if config.get('cuda', True) and torch.cuda.is_available() else 'cpu')
        self.device = device

        # --- Networks ---
        # Primary generator A->B
        self.netG_A2B = HeadCECTGANGenerator(config['input_nc'], config['output_nc']).to(device)
        # discriminator for domain B
        self.netD_B = HeadCECTGANDiscriminator(config['output_nc']).to(device)

        # Optional bidirectional branch
        self.bidirect = config.get('bidirect', False)
        if self.bidirect:
            self.netG_B2A = HeadCECTGANGenerator(config['input_nc'], config['output_nc']).to(device)
            self.netD_A = HeadCECTGANDiscriminator(config['output_nc']).to(device)

        # Optional registration module
        self.regist = config.get('regist', False)
        if self.regist:
            self.R_A = Reg(config['size'], config['size'], config['input_nc'], config['input_nc']).to(device)
            self.spatial_transform = Transformer_2D().to(device)

        # Initialize weights if requested
        if config.get('init_weights', True):
            # Use a safe local initializer to avoid touching other modules/files.
            # This prevents AttributeError when external modules expose non-standard
            # wrappers (no `weight` attribute) and avoids deprecated API calls in
            # other files. We intentionally keep this change local to this trainer.
            def _safe_weights_init(m):
                try:
                    if hasattr(m, 'weight') and m.weight is not None:
                        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
                except Exception:
                    pass
                try:
                    if hasattr(m, 'bias') and m.bias is not None:
                        torch.nn.init.constant_(m.bias.data, 0.0)
                except Exception:
                    pass

            nets_to_init = [self.netG_A2B, self.netD_B]
            if self.bidirect:
                nets_to_init += [self.netG_B2A, self.netD_A]
            if self.regist:
                nets_to_init += [self.R_A]

            for net in nets_to_init:
                net.apply(_safe_weights_init)

        # Globally suppress a noisy torch.meshgrid warning (keeps change local to trainer runtime)
        warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

        # --- Optimizers ---
        if self.bidirect:
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()), lr=config['lr'], betas=(0.5, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        else:
            self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        if self.regist:
            self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        # --- Losses ---
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()

        # --- Inputs & targets ---
        # create buffers on the correct device without using deprecated constructors
        self.input_A = torch.empty((config['batchSize'], config['input_nc'], config['size'], config['size']), device=self.device)
        self.input_B = torch.empty((config['batchSize'], config['output_nc'], config['size'], config['size']), device=self.device)
        self.target_real = None
        self.target_fake = None

        # --- Replay buffers ---
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        # --- Dataset loader ---
        level = config.get('noise_level', 1)
        # Local safe ToTensor that correctly handles grayscale numpy arrays and PIL Images
        # Helper to create RandomAffine compatible with different torchvision versions
        def _make_random_affine(degrees, translate, scale, fill):
            # RandomAffine signature changed between torchvision versions (fill vs fillcolor)
            sig = inspect.signature(RandomAffine)
            if 'fill' in sig.parameters:
                return RandomAffine(degrees=degrees, translate=translate, scale=scale, fill=fill)
            elif 'fillcolor' in sig.parameters:
                return RandomAffine(degrees=degrees, translate=translate, scale=scale, fillcolor=fill)
            else:
                # Fallback without fill (older versions)
                return RandomAffine(degrees=degrees, translate=translate, scale=scale)

        # Use CycleTrainer-style PIL-based transforms by default

        transforms_1 = [ToPILImage(), _make_random_affine(degrees=level, translate=[0.02 * level, 0.02 * level], scale=[1 - 0.02 * level, 1 + 0.02 * level], fill=-1), ToTensor(), Resize(size_tuple=(config['size'], config['size']))]
        transforms_2 = [ToPILImage(), _make_random_affine(degrees=1, translate=[0.02, 0.02], scale=[0.98, 1.02], fill=-1), ToTensor(), Resize(size_tuple=(config['size'], config['size']))]

        self.dataloader = DataLoader(ImageDataset(config['dataroot'], level, transforms_1=transforms_1, transforms_2=transforms_2, unaligned=False), batch_size=config['batchSize'], shuffle=True, num_workers=config.get('n_cpu', 0))

        val_transforms = [ToTensor(), Resize(size_tuple=(config['size'], config['size']))]
        self.val_data = DataLoader(ValDataset(config['val_dataroot'], transforms_=val_transforms, unaligned=False), batch_size=config['batchSize'], shuffle=False, num_workers=config.get('n_cpu', 0))

        self.test_data = DataLoader(ImageDataset(config.get('test_dataroot', config['dataroot']), level, transforms_1=transforms_1, transforms_2=transforms_2, unaligned=False), batch_size=1, shuffle=False, num_workers=config.get('n_cpu', 0))

        # --- Logger ---
        self.logger = Logger(config.get('name', 'cta2'), config.get('port', 8097), config.get('n_epochs', 100), len(self.dataloader))

        # --- LR schedulers ---
        n_epochs = config.get('n_epochs', 100)
        decay_epoch = config.get('decay_epoch', int(n_epochs * 0.5))
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=LambdaLR(n_epochs, 0, decay_epoch).step)
        self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_B, lr_lambda=LambdaLR(n_epochs, 0, decay_epoch).step)
        if self.bidirect:
            self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_A, lr_lambda=LambdaLR(n_epochs, 0, decay_epoch).step)
        if self.regist:
            self.lr_scheduler_R_A = torch.optim.lr_scheduler.LambdaLR(self.optimizer_R_A, lr_lambda=LambdaLR(n_epochs, 0, decay_epoch).step)

        # --- Save root ---
        self.save_root = config.get('save_root', './')
        os.makedirs(self.save_root, exist_ok=True)

    def train(self):
        cfg = self.config
        for epoch in range(cfg.get('epoch', 0), cfg['n_epochs']):
            for i, batch in enumerate(self.dataloader):
                # Prepare inputs
                real_A = Variable(self.input_A.copy_(batch['A'])).to(self.device)
                real_B = Variable(self.input_B.copy_(batch['B'])).to(self.device)

                # ---- Branching similar to CycTrainer ----
                if self.bidirect:
                    # Both directions trained
                    if self.regist:
                        # Train registration + both generators
                        self.optimizer_R_A.zero_grad()
                        self.optimizer_G.zero_grad()

                        fake_B, flow_B = self.netG_A2B(real_A)
                        pred_fake = self.netD_B(fake_B)
                        loss_GAN_A2B = cfg['Adv_lamda'] * self.MSE_loss(pred_fake, torch.ones_like(pred_fake))

                        fake_A, flow_A = self.netG_B2A(real_B)
                        pred_fake = self.netD_A(fake_A)
                        loss_GAN_B2A = cfg['Adv_lamda'] * self.MSE_loss(pred_fake, torch.ones_like(pred_fake))

                        Trans = self.R_A(fake_B, real_B)
                        SysRegist_A2B = self.spatial_transform(fake_B, Trans)
                        # Use weighted L1 for registration to focus on vessels
                        SR_loss = cfg['Corr_lamda'] * weighted_l1_loss(SysRegist_A2B, real_B)
                        SM_loss = cfg['Smooth_lamda'] * smooothing_loss(Trans)

                        # Add smoothing loss for generator flow
                        flow_loss = cfg['Smooth_lamda'] * (smooothing_loss(flow_B) + smooothing_loss(flow_A))
                        
                        # Add TV loss for image continuity (reduce scattered dots)
                        tv_loss_val = cfg.get('TV_lamda', 1.0) * (smooothing_loss(fake_B) + smooothing_loss(fake_A))

                        recovered_A, _ = self.netG_B2A(fake_B)
                        loss_cycle_ABA = cfg['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)

                        recovered_B, _ = self.netG_A2B(fake_A)
                        # Use weighted L1 for B cycle consistency (CTA domain)
                        loss_cycle_BAB = cfg['Cyc_lamda'] * weighted_l1_loss(recovered_B, real_B)

                        loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + SR_loss + SM_loss + flow_loss + tv_loss_val
                        loss_Total.backward()
                        self.optimizer_G.step()
                        self.optimizer_R_A.step()

                        # Discriminator A
                        self.optimizer_D_A.zero_grad()
                        pred_real = self.netD_A(real_A)
                        loss_D_real = cfg['Adv_lamda'] * self.MSE_loss(pred_real, torch.ones_like(pred_real))
                        fake_A_buf = self.fake_A_buffer.push_and_pop(fake_A)
                        pred_fake = self.netD_A(fake_A_buf.detach())
                        loss_D_fake = cfg['Adv_lamda'] * self.MSE_loss(pred_fake, torch.zeros_like(pred_fake))
                        loss_D_A = (loss_D_real + loss_D_fake)
                        loss_D_A.backward()
                        self.optimizer_D_A.step()

                        # Discriminator B
                        self.optimizer_D_B.zero_grad()
                        pred_real = self.netD_B(real_B)
                        loss_D_real = cfg['Adv_lamda'] * self.MSE_loss(pred_real, torch.ones_like(pred_real))
                        fake_B_buf = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake = self.netD_B(fake_B_buf.detach())
                        loss_D_fake = cfg['Adv_lamda'] * self.MSE_loss(pred_fake, torch.zeros_like(pred_fake))
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()
                        self.optimizer_D_B.step()

                        # Logging
                        try:
                            self.logger.log({'loss_D_B': loss_D_B, 'SR_loss': SR_loss}, images={'real_A': real_A, 'real_B': real_B, 'fake_B': fake_B})
                        except Exception:
                            pass
                    else:
                        # Bidirect without registration
                        self.optimizer_G.zero_grad()
                        fake_B, _ = self.netG_A2B(real_A)
                        pred_fake = self.netD_B(fake_B)
                        loss_GAN_A2B = cfg['Adv_lamda'] * self.MSE_loss(pred_fake, torch.ones_like(pred_fake))

                        fake_A, _ = self.netG_B2A(real_B)
                        pred_fake = self.netD_A(fake_A)
                        loss_GAN_B2A = cfg['Adv_lamda'] * self.MSE_loss(pred_fake, torch.ones_like(pred_fake))

                        recovered_A, _ = self.netG_B2A(fake_B)
                        loss_cycle_ABA = cfg['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)

                        recovered_B, _ = self.netG_A2B(fake_A)
                        # Use weighted L1 for B cycle consistency
                        loss_cycle_BAB = cfg['Cyc_lamda'] * weighted_l1_loss(recovered_B, real_B)

                        # Add smoothing loss for generator flow
                        flow_loss = cfg['Smooth_lamda'] * (smooothing_loss(flow_B) + smooothing_loss(flow_A))
                        
                        # Add TV loss for image continuity
                        tv_loss_val = cfg.get('TV_lamda', 1.0) * (smooothing_loss(fake_B) + smooothing_loss(fake_A))

                        loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + flow_loss + tv_loss_val
                        loss_Total.backward()
                        self.optimizer_G.step()

                        # Discriminator A
                        self.optimizer_D_A.zero_grad()
                        pred_real = self.netD_A(real_A)
                        loss_D_real = cfg['Adv_lamda'] * self.MSE_loss(pred_real, torch.ones_like(pred_real))
                        fake_A_buf = self.fake_A_buffer.push_and_pop(fake_A)
                        pred_fake = self.netD_A(fake_A_buf.detach())
                        loss_D_fake = cfg['Adv_lamda'] * self.MSE_loss(pred_fake, torch.zeros_like(pred_fake))
                        loss_D_A = (loss_D_real + loss_D_fake)
                        loss_D_A.backward()
                        self.optimizer_D_A.step()

                        # Discriminator B
                        self.optimizer_D_B.zero_grad()
                        pred_real = self.netD_B(real_B)
                        loss_D_real = cfg['Adv_lamda'] * self.MSE_loss(pred_real, torch.ones_like(pred_real))
                        fake_B_buf = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake = self.netD_B(fake_B_buf.detach())
                        loss_D_fake = cfg['Adv_lamda'] * self.MSE_loss(pred_fake, torch.zeros_like(pred_fake))
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()
                        self.optimizer_D_B.step()

                        try:
                            self.logger.log({'loss_D_B': loss_D_B}, images={'real_A': real_A, 'real_B': real_B, 'fake_B': fake_B})
                        except Exception:
                            pass
                else:
                    # Single direction (A->B)
                    if self.regist:
                        self.optimizer_R_A.zero_grad()
                        self.optimizer_G.zero_grad()
                        fake_B, flow_B = self.netG_A2B(real_A)
                        Trans = self.R_A(fake_B, real_B)
                        SysRegist_A2B = self.spatial_transform(fake_B, Trans)
                        # Use weighted L1 for registration
                        SR_loss = cfg['Corr_lamda'] * weighted_l1_loss(SysRegist_A2B, real_B)
                        pred_fake0 = self.netD_B(fake_B)
                        adv_loss = cfg['Adv_lamda'] * self.MSE_loss(pred_fake0, torch.ones_like(pred_fake0))
                        SM_loss = cfg['Smooth_lamda'] * smooothing_loss(Trans)
                        
                        # Add smoothing loss for generator flow
                        flow_loss = cfg['Smooth_lamda'] * smooothing_loss(flow_B)

                        # Add TV loss for image continuity
                        tv_loss_val = cfg.get('TV_lamda', 1.0) * smooothing_loss(fake_B)
                        
                        total_loss = SM_loss + adv_loss + SR_loss + flow_loss + tv_loss_val
                        total_loss.backward()
                        self.optimizer_R_A.step()
                        self.optimizer_G.step()

                        # Discriminator B update
                        self.optimizer_D_B.zero_grad()
                        with torch.no_grad():
                            fake_B, _ = self.netG_A2B(real_A)
                        pred_fake0 = self.netD_B(fake_B)
                        pred_real = self.netD_B(real_B)
                        loss_D_B = cfg['Adv_lamda'] * self.MSE_loss(pred_fake0, torch.zeros_like(pred_fake0)) + cfg['Adv_lamda'] * self.MSE_loss(pred_real, torch.ones_like(pred_real))
                        loss_D_B.backward()
                        self.optimizer_D_B.step()

                        try:
                            self.logger.log({'loss_D_B': loss_D_B, 'SR_loss': SR_loss}, images={'real_A': real_A, 'real_B': real_B, 'fake_B': fake_B})
                        except Exception:
                            pass
                    else:
                        # Only A->B GAN
                        self.optimizer_G.zero_grad()
                        fake_B, flow_B = self.netG_A2B(real_A)
                        pred_fake = self.netD_B(fake_B)
                        adv_loss = cfg['Adv_lamda'] * self.MSE_loss(pred_fake, torch.ones_like(pred_fake))
                        
                        # Add smoothing loss for generator flow
                        flow_loss = cfg['Smooth_lamda'] * smooothing_loss(flow_B)

                        # Add TV loss for image continuity
                        tv_loss_val = cfg.get('TV_lamda', 1.0) * smooothing_loss(fake_B)
                        
                        total_loss = adv_loss + flow_loss + tv_loss_val
                        total_loss.backward()
                        self.optimizer_G.step()

                        # Discriminator B
                        self.optimizer_D_B.zero_grad()
                        pred_real = self.netD_B(real_B)
                        loss_D_real = cfg['Adv_lamda'] * self.MSE_loss(pred_real, torch.ones_like(pred_real))
                        fake_B_buf = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake = self.netD_B(fake_B_buf.detach())
                        loss_D_fake = cfg['Adv_lamda'] * self.MSE_loss(pred_fake, torch.zeros_like(pred_fake))
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()
                        self.optimizer_D_B.step()

                        try:
                            self.logger.log({'loss_D_B': loss_D_B}, images={'real_A': real_A, 'real_B': real_B, 'fake_B': fake_B})
                        except Exception:
                            pass

            # End epoch: save checkpoints
            torch.save(self.netG_A2B.state_dict(), os.path.join(self.save_root, 'netG_A2B.pth'))
            if self.bidirect:
                torch.save(self.netG_B2A.state_dict(), os.path.join(self.save_root, 'netG_B2A.pth'))
                torch.save(self.netD_A.state_dict(), os.path.join(self.save_root, 'netD_A.pth'))
            torch.save(self.netD_B.state_dict(), os.path.join(self.save_root, 'netD_B.pth'))
            if self.regist:
                torch.save(self.R_A.state_dict(), os.path.join(self.save_root, 'Regist.pth'))

            # Validation metrics
            with torch.no_grad():
                MAE_total = 0.0
                PSNR_total = 0.0
                SSIM_total = 0.0
                num = 0
                for j, batch in enumerate(self.val_data):
                    real_A = Variable(self.input_A.copy_(batch['A'])).to(self.device)
                    real_B = Variable(self.input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()
                    fake_B = self.netG_A2B(real_A)[0].detach().cpu().numpy().squeeze()
                    mae = self.MAE(fake_B, real_B)
                    psnr = self.PSNR(fake_B, real_B)
                    ssim_val = ssim_metric((fake_B + 1) / 2, (real_B + 1) / 2, data_range=1.0)
                    MAE_total += mae
                    PSNR_total += psnr
                    SSIM_total += ssim_val
                    num += 1
                if num > 0:
                    print('Val MAE:', MAE_total / num)
                    print('Val PSNR:', PSNR_total / num)
                    print('Val SSIM:', SSIM_total / num)

            # Step LR schedulers
            self.lr_scheduler_G.step()
            self.lr_scheduler_D_B.step()
            if self.bidirect:
                self.lr_scheduler_D_A.step()
            if self.regist:
                self.lr_scheduler_R_A.step()

    def test(self):
        # Load checkpoints
        g_ck = os.path.join(self.save_root, 'netG_A2B.pth')
        if os.path.exists(g_ck):
            self.netG_A2B.load_state_dict(torch.load(g_ck, map_location=self.device))
        if self.regist:
            r_ck = os.path.join(self.save_root, 'Regist.pth')
            if os.path.exists(r_ck):
                self.R_A.load_state_dict(torch.load(r_ck, map_location=self.device))

        out_dir = os.path.join(self.save_root, 'test_results')
        os.makedirs(out_dir, exist_ok=True)

        MAE = 0.0
        PSNR = 0.0
        SSIM = 0.0
        num = 0
        with torch.no_grad():
            for i, batch in enumerate(self.test_data):
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()
                fake_B = self.netG_A2B(real_A.to(self.device))[0].detach().cpu().numpy().squeeze()

                # save image
                img = ((fake_B + 1) * 127.5).astype('uint8')
                Image.fromarray(img).save(os.path.join(out_dir, f'fake_{i}.png'))

                mae = self.MAE(fake_B, real_B)
                psnr = self.PSNR(fake_B, real_B)
                ssim_val = ssim_metric((fake_B + 1) / 2, (real_B + 1) / 2, data_range=1.0)
                MAE += mae
                PSNR += psnr
                SSIM += ssim_val
                num += 1

        if num > 0:
            print('MAE:', MAE / num)
            print('PSNR:', PSNR / num)
            print('SSIM:', SSIM / num)

    def PSNR(self, fake, real):
        x, y = np.where(real != -1)
        mse = np.mean(((fake[x, y] + 1) / 2. - (real[x, y] + 1) / 2.) ** 2)
        if mse < 1.0e-10:
            return 100
        else:
            PIXEL_MAX = 1
            return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

    def MAE(self, fake, real):
        x, y = np.where(real != -1)
        mae = np.abs(fake[x, y] - real[x, y]).mean()
        return mae / 2

    def save_deformation(self, defms, root):
        heatmapshow = None
        defms_ = defms.data.cpu().float().numpy()
        dir_x = defms_[0]
        dir_y = defms_[1]
        x_max, x_min = dir_x.max(), dir_x.min()
        y_max, y_min = dir_y.max(), dir_y.min()
        dir_x = ((dir_x - x_min) / (x_max - x_min)) * 255
        dir_y = ((dir_y - y_min) / (y_max - y_min)) * 255
        tans_x = cv2.normalize(dir_x, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        tans_x = cv2.applyColorMap(tans_x, cv2.COLORMAP_JET)
        tans_y = cv2.normalize(dir_y, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        tans_y = cv2.applyColorMap(tans_y, cv2.COLORMAP_JET)
        gradxy = cv2.addWeighted(tans_x, 0.5, tans_y, 0.5, 0)
        cv2.imwrite(root, gradxy)

