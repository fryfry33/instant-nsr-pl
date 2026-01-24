import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss
import json
import numpy as np
import os

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug

import models
from models.utils import cleanup
from models.ray_utils import get_rays
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, binary_cross_entropy

import trimesh

def debug_prior_alignment(system, output_name="debug_alignment_check.obj"):
    """
    Crée un nuage de points pour voir comment le Prior est orienté.
    """
    if system.trainer.global_rank != 0: return # Seulement sur le GPU principal

    print(f"--- [DEBUG] GÉNÉRATION DU VISUEL PRIOR : {output_name} ---")
    
    # 1. On scanne l'espace (boîte de -1 à 1)
    N = 64
    x = np.linspace(-1.0, 1.0, N)
    y = np.linspace(-1.0, 1.0, N)
    z = np.linspace(-1.0, 1.0, N)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z)
    pts = np.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], axis=1)
    
    # On envoie sur le GPU pour le calcul
    pts_tensor = torch.from_numpy(pts).float().to(system.device)
    
    # 2. On demande au système : "Quelle est la distance SDF ici ?"
    # (C'est ici que votre rotation dans get_prior_sdf_at va être testée)
    with torch.no_grad():
        sdf_values = system.get_prior_sdf_at(pts_tensor)
        
    # 3. On garde les points proches de 0 (la surface)
    mask = sdf_values.abs().squeeze() < 0.05
    surface_points = pts[mask.cpu().numpy()]
    
    if len(surface_points) > 0:
        # 4. Sauvegarde
        pcd = trimesh.points.PointCloud(surface_points, colors=[255, 0, 0, 255]) # Rouge
        pcd.export(output_name)
        print(f"✅ [DEBUG] Fichier généré : {output_name}")
    else:
        print("❌ [DEBUG] Aucun point trouvé (Prior hors champ ?)")
# --------------------------------------------------------------------------
@systems.register('neus-system')
class NeuSSystem(BaseSystem):
    
    def on_train_start(self):
        """
        Cette méthode est appelée automatiquement par Pytorch Lightning
        juste avant de commencer la boucle d'entraînement.
        """
        # On lance le debug immédiatement
        if self.use_prior:
            debug_prior_alignment(self, "debug_prior_orientation.obj")
    
    
    def prepare(self):
        self.criterions = {
            'psnr': PSNR()
        }
        self.train_num_samples = self.config.model.train_num_rays * (self.config.model.num_samples_per_ray + self.config.model.get('num_samples_per_ray_bg', 0))
        self.train_num_rays = self.config.model.train_num_rays

        # --- [MODIFICATION START] LOAD PRIOR VIA ENV VAR ---
        # 1. On cherche d'abord la variable d'environnement
        prior_dir = os.environ.get("PRIOR_DIR")

        # 2. Si elle n'est pas définie, on tente un chemin par défaut (fallback)
        if prior_dir is None:
            # Fallback Kaggle standard
            prior_dir = "/kaggle/working/prior_data"
            rank_zero_info(f"⚠️ Variable PRIOR_DIR non définie. Tentative avec : {prior_dir}")

        rank_zero_info(f"📁 Recherche du Prior dans : {prior_dir}")
        
        try:
            # On vérifie si le fichier existe
            # Priorité 1 : Version lissée
            path_npy = os.path.join(prior_dir, "sdf_volume_smooth.npy")
            if not os.path.exists(path_npy):
                # Priorité 2 : Version standard
                path_npy = os.path.join(prior_dir, "sdf_volume.npy")
            
            if os.path.exists(path_npy):
                sdf_vol_np = np.load(path_npy, allow_pickle=True)
                
                self.register_buffer("sdf_prior_vol", torch.from_numpy(sdf_vol_np).float().unsqueeze(0).unsqueeze(0))
                
                # Chargement de la config JSON associée
                config_path = os.path.join(prior_dir, "sdf_config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        meta = json.load(f)
                    self.register_buffer("prior_min", torch.tensor(meta["min_bound"]).float())
                    self.register_buffer("prior_max", torch.tensor(meta["max_bound"]).float())
                    
                    self.use_prior = True
                    rank_zero_info(f"✅ PRIOR LOADED SUCCESSFULLY from {path_npy}")
                else:
                    rank_zero_info(f"❌ Erreur: sdf_config.json introuvable dans {prior_dir}")
                    self.use_prior = False
            else:
                rank_zero_info(f"⚠️ Fichier .npy introuvable dans {prior_dir}. Prior désactivé.")
                self.use_prior = False
                
        except Exception as e:
            rank_zero_info(f"⚠️ CRASH chargement Prior: {e}")
            self.use_prior = False
        # --- [MODIFICATION END] ---

    def forward(self, batch):
        return self.model(batch['rays'])
    
    def preprocess_data(self, batch, stage):
        if 'index' in batch: # validation / testing
            index = batch['index']
        else:
            if self.config.model.batch_image_sampling:
                index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,), device=self.dataset.all_images.device)
            else:
                index = torch.randint(0, len(self.dataset.all_images), size=(1,), device=self.dataset.all_images.device)
        if stage in ['train']:
            c2w = self.dataset.all_c2w[index]
            x = torch.randint(
                0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            y = torch.randint(
                0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions[y, x]
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index, y, x]
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1).to(self.rank)
        else:
            c2w = self.dataset.all_c2w[index][0]
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index][0] 
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index].view(-1).to(self.rank)

        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        if stage in ['train']:
            if self.config.model.background_color == 'white':
                self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
            elif self.config.model.background_color == 'random':
                self.model.background_color = torch.rand((3,), dtype=torch.float32, device=self.rank)
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
        
        if self.dataset.apply_mask:
            rgb = rgb * fg_mask[...,None] + self.model.background_color * (1 - fg_mask[...,None])
        
        batch.update({
            'rays': rays,
            'rgb': rgb,
            'fg_mask': fg_mask
        })      
    
    # --- [MODIFICATION START] HELPER FUNCTION ---
    # --- Remplacer cette fonction dans systems/neus.py ---
    def get_prior_sdf_at(self, points):
        """
        Version V2 calibrée (Cumul : Ancien 0.03 + Nouveau 0.17).
        Total Translation : X=0.20, Y=-0.01, Z=0.02
        """
        if not self.use_prior:
            return torch.zeros(points.shape[0], 1, device=points.device)
        
        # --- CALIBRAGE CUMULÉ ---
        # Nouveau Total : [0.20, -0.01, 0.02]
        # camobrage pour gaparini [0.20, -0.02, 0.02]
        correction_vector = torch.tensor([0.0, 0.0, 0.0], device=points.device)
        
        # On soustrait le vecteur total pour aligner
        points_corrected = points - correction_vector
        
        # -----------------------------------------------------

        denom = self.prior_max - self.prior_min
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        
        points_norm = 2 * (points_corrected - self.prior_min) / denom - 1
        
        grid_coords = points_norm.view(1, 1, 1, -1, 3)
        
        prior_values = F.grid_sample(
            self.sdf_prior_vol, 
            grid_coords, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )
        return prior_values.view(-1, 1)
    # --- [MODIFICATION END] ---

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = 0.

        # --- DYNAMIC RAY SAMPLING ---
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples_full'].sum().item()))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)

        # --- CALCUL DES LOSS CLASSIQUES ---
        loss_rgb_mse = F.mse_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
        self.log('train/loss_rgb_mse', loss_rgb_mse)
        loss += loss_rgb_mse * self.C(self.config.system.loss.lambda_rgb_mse)

        loss_rgb_l1 = F.l1_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
        self.log('train/loss_rgb', loss_rgb_l1)
        loss += loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)        

        loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()
        self.log('train/loss_eikonal', loss_eikonal)
        loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)
        
        opacity = torch.clamp(out['opacity'].squeeze(-1), 1.e-3, 1.-1.e-3)
        loss_mask = binary_cross_entropy(opacity, batch['fg_mask'].float())
        self.log('train/loss_mask', loss_mask)
        loss += loss_mask * (self.C(self.config.system.loss.lambda_mask) if self.dataset.has_mask else 0.0)

        loss_opaque = binary_cross_entropy(opacity, opacity)
        self.log('train/loss_opaque', loss_opaque)
        loss += loss_opaque * self.C(self.config.system.loss.lambda_opaque)

        loss_sparsity = torch.exp(-self.config.system.loss.sparsity_scale * out['sdf_samples'].abs()).mean()
        self.log('train/loss_sparsity', loss_sparsity)
        loss += loss_sparsity * self.C(self.config.system.loss.lambda_sparsity)

        if self.C(self.config.system.loss.lambda_curvature) > 0:
            loss_curvature = out['sdf_laplace_samples'].abs().mean()
            self.log('train/loss_curvature', loss_curvature)
            loss += loss_curvature * self.C(self.config.system.loss.lambda_curvature)

        if self.C(self.config.system.loss.lambda_distortion) > 0:
            loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
            self.log('train/loss_distortion', loss_distortion)
            loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion)    

        if self.config.model.learned_background and self.C(self.config.system.loss.lambda_distortion_bg) > 0:
            loss_distortion_bg = flatten_eff_distloss(out['weights_bg'], out['points_bg'], out['intervals_bg'], out['ray_indices_bg'])
            self.log('train/loss_distortion_bg', loss_distortion_bg)
            loss += loss_distortion_bg * self.C(self.config.system.loss.lambda_distortion_bg)        

        # --- [PRIOR AVEC DECAY CONFIGURABLE] ---
        if self.use_prior and 'sdf_samples' in out:
            # 1. Lecture des paramètres depuis le YAML (avec valeurs par défaut de sécurité)
            loss_cfg = self.config.system.loss
            decay_start = loss_cfg.get('prior_decay_start', 3000)
            decay_end = loss_cfg.get('prior_decay_end', 15000)
            min_factor = loss_cfg.get('prior_min_factor', 0.1)

            # 2. Calcul du facteur de Decay
            current_step = self.global_step
            if current_step < decay_start:
                decay_factor = 1.0
            elif current_step > decay_end:
                decay_factor = min_factor
            else:
                # Interpolation linéaire
                progress = (current_step - decay_start) / (decay_end - decay_start)
                decay_factor = 1.0 - (1.0 - min_factor) * progress
            
            # Log pour vérifier que ça marche
            if self.global_step % 100 == 0:
                self.log('train_params/prior_decay_factor', decay_factor)

            # 3. Récupération des points 3D (Gestion 1D vs 3D)
            points_3d = None
            if 'points' in out and out['points'].ndim == 1:
                if 'ray_indices' in out and 'rays' in batch:
                    rays = batch['rays']
                    rays_o, rays_d = rays[:, :3], rays[:, 3:6]
                    ray_idx = out['ray_indices'].long()
                    t_vals = out['points'] 
                    points_3d = rays_o[ray_idx] + rays_d[ray_idx] * t_vals.unsqueeze(-1)
            elif 'points' in out and out['points'].ndim == 2 and out['points'].shape[-1] == 3:
                points_3d = out['points']
            
            # 4. Calcul Loss et Application
            if points_3d is not None:
                sdf_pred = out['sdf_samples']
                sdf_target = self.get_prior_sdf_at(points_3d)
                
                # Fix du shape mismatch [N] vs [N, 1]
                if sdf_pred.ndim == 1 and sdf_target.ndim == 2:
                    sdf_target = sdf_target.squeeze(-1)
                
                loss_prior_val = F.l1_loss(sdf_pred, sdf_target)
                
                # Récupération du poids de base dans le YAML
                base_lambda = 1.0
                if hasattr(self.config.system.loss, 'lambda_prior'):
                    base_lambda = self.C(self.config.system.loss.lambda_prior)
                
                # Poids effectif = Base * Decay
                final_lambda = base_lambda * decay_factor
                
                self.log('train/loss_prior', loss_prior_val)
                self.log('train_params/lambda_prior_effective', final_lambda)
                
                loss += final_lambda * loss_prior_val
        # ----------------------------------------

        # --- REGULARIZATIONS DU MODÈLE ---
        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_
        
        self.log('train/inv_s', out['inv_s'], prog_bar=True)

        # Logging des paramètres
        for name, value in self.config.system.loss.items():
            # On log tous les lambdas sauf le prior statique pour éviter la confusion
            if name.startswith('lambda') and name != 'lambda_prior': 
                self.log(f'train_params/{name}', self.C(value))

        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)

        return {
            'loss': loss
        }
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ] + ([
            {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ])
        return {
            'psnr': psnr,
            'index': batch['index']
        }
          
    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)          

    def test_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ] + ([
            {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ])
        return {
            'psnr': psnr,
            'index': batch['index']
        }      
    
    def test_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True)    

            self.save_img_sequence(
                f"it{self.global_step}-test",
                f"it{self.global_step}-test",
                '(\d+)\.png',
                save_format='mp4',
                fps=30
            )
            self.export()
    
    def export(self):
        mesh = self.model.export(self.config.export)
        self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
            **mesh
        )
