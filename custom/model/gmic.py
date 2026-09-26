# Copyright (C) 2020 Yiqiu Shen, Nan Wu, Jason Phang, Jungkyu Park, Kangning Liu,
# Sudarshini Tyagi, Laura Heacock, S. Gene Kim, Linda Moy, Kyunghyun Cho, Krzysztof J. Geras
#
# This file is part of GMIC.
#
# GMIC is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# GMIC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with GMIC.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

"""
Module that define the core logic of GMIC
"""

# DataParallel-compatible GMIC implementation
# Save this as src/modeling/gmic_dataparallel.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utilities import tools
import model.modules as m

class GMIC(nn.Module):
    """
    DataParallel-compatible version of GMIC that properly registers all submodules.
    """
    def __init__(self, parameters):
        super(GMIC, self).__init__()
        
        # save parameters
        self.experiment_parameters = parameters
        self.cam_size = parameters["cam_size"]
        
        # === PROPERLY REGISTER ALL SUBMODULES ===
        
        # 1. Global Network Components
        if "use_v1_global" in parameters and parameters["use_v1_global"]:
            self.ds_net = m.DownsampleNetworkResNet18V1()
        else:
            self.ds_net = m.ResNetV2(
                input_channels=1, num_filters=16,
                first_layer_kernel_size=(7,7), first_layer_conv_stride=2,
                first_layer_padding=3,
                first_pool_size=3, first_pool_stride=2, first_pool_padding=0,
                blocks_per_layer_list=[2, 2, 2, 2, 2],
                block_strides_list=[1, 2, 2, 2, 2],
                block_fn=m.BasicBlockV2,
                growth_factor=2
            )
        
        self.left_postprocess_net = m.PostProcessingStandard(parameters)
        
        # 2. Local Network Components
        self.dn_resnet = m.ResNetV1(64, m.BasicBlockV1, [2,2,2,2], 3)
        
        # 3. Attention Module Components
        self.mil_attn_V = nn.Linear(512, 128, bias=False)
        self.mil_attn_U = nn.Linear(512, 128, bias=False)
        self.mil_attn_w = nn.Linear(128, 1, bias=False)
        self.classifier_linear = nn.Linear(512, parameters["num_classes"], bias=False)
        
        # 4. Fusion Components
        self.fusion_dnn = nn.Linear(parameters["post_processing_dim"]+512, parameters["num_classes"])
        
        # 5. Non-learnable components (these don't need to be registered)
        self.percent_t = parameters["percent_t"]
        self.K = parameters["K"]
        self.crop_shape = parameters["crop_shape"]
        
        # Buffer for device reference
        self.register_buffer("_device_ref", torch.empty(0))

    def _first_param_device(self):
        """Get device from first parameter, with fallback to buffer"""
        for p in self.parameters():
            return p.device
        return self._device_ref.device

    @property
    def global_backbone(self):
        return self.ds_net

    @property
    def local_backbone(self):
        return self.dn_resnet

    @property
    def global_1x1_and_post(self):
        return self.left_postprocess_net

    @property
    def mil_attention_block(self):
        return nn.ModuleDict({
            'mil_attn_V': self.mil_attn_V,
            'mil_attn_U': self.mil_attn_U,
            'mil_attn_w': self.mil_attn_w
        })

    @property
    def classifier_heads(self):
        return nn.ModuleDict({
            'classifier_linear': self.classifier_linear,
            'fusion_dnn': self.fusion_dnn
        })

    def _convert_crop_position(self, crops_x_small, cam_size, x_original):
        """Convert crop locations from cam_size to x_original"""
        h, w = cam_size
        _, _, H, W = x_original.size()
        
        top_k_prop_x = crops_x_small[:, :, 0] / h
        top_k_prop_y = crops_x_small[:, :, 1] / w
        
        assert np.max(top_k_prop_x) <= 1.0, "top_k_prop_x >= 1.0"
        assert np.min(top_k_prop_x) >= 0.0, "top_k_prop_x <= 0.0"
        assert np.max(top_k_prop_y) <= 1.0, "top_k_prop_y >= 1.0"
        assert np.min(top_k_prop_y) >= 0.0, "top_k_prop_y <= 0.0"
        
        top_k_interpolate_x = np.expand_dims(np.around(top_k_prop_x * H), -1)
        top_k_interpolate_y = np.expand_dims(np.around(top_k_prop_y * W), -1)
        top_k_interpolate_2d = np.concatenate([top_k_interpolate_x, top_k_interpolate_y], axis=-1)
        return top_k_interpolate_2d

    def _retrieve_crop(self, x_original_pytorch, crop_positions, crop_method):
        """Retrieve crops from original image"""
        batch_size, num_crops, _ = crop_positions.shape
        crop_h, crop_w = self.crop_shape
        
        device = x_original_pytorch.device
        output = torch.ones(
            (batch_size, num_crops, crop_h, crop_w), 
            device=device, dtype=x_original_pytorch.dtype
        )
        
        for i in range(batch_size):
            for j in range(num_crops):
                tools.crop_pytorch(
                    x_original_pytorch[i, 0, :, :],
                    self.crop_shape,
                    crop_positions[i,j,:],
                    output[i,j,:,:],
                    method=crop_method
                )
        return output

    def _global_network_forward(self, x):
        """Global network forward pass"""
        last_feature_map = self.ds_net(x)
        cam = self.left_postprocess_net(last_feature_map)
        return last_feature_map, cam

    def _aggregation_function_forward(self, cam):
        """Top-T percent aggregation"""
        batch_size, num_class, H, W = cam.size()
        cam_flatten = cam.view(batch_size, num_class, -1)
        top_t = int(round(W*H*self.percent_t))
        selected_area = cam_flatten.topk(top_t, dim=2)[0]
        return selected_area.mean(dim=2)

    def _retrieve_roi_forward(self, x_original, cam_size, h_small):
        """ROI retrieval forward pass"""
        _, _, H, W = x_original.size()
        (h, w) = cam_size
        N, C, h_h, w_h = h_small.size()
        
        assert h_h == h, "h_h!=h"
        assert w_h == w, "w_h!=w"
        
        crop_x_adjusted = int(np.round(self.crop_shape[0] * h / H))
        crop_y_adjusted = int(np.round(self.crop_shape[1] * w / W))
        crop_shape_adjusted = (crop_x_adjusted, crop_y_adjusted)
        
        current_images = h_small
        all_max_position = []
        
        max_vals = current_images.view(N, C, -1).max(dim=2, keepdim=True)[0].unsqueeze(-1)
        min_vals = current_images.view(N, C, -1).min(dim=2, keepdim=True)[0].unsqueeze(-1)
        range_vals = max_vals - min_vals
        normalize_images = current_images - min_vals
        normalize_images = normalize_images / range_vals
        current_images = normalize_images.sum(dim=1, keepdim=True)
        
        for _ in range(self.K):
            max_pos = tools.get_max_window(current_images, crop_shape_adjusted, "avg")
            all_max_position.append(max_pos)
            mask = tools.generate_mask_uplft(current_images, crop_shape_adjusted, max_pos, None)
            current_images = current_images * mask
            
        return torch.cat(all_max_position, dim=1).data.cpu().numpy()

    def _local_network_forward(self, x_crop):
        """Local network forward pass"""
        res = self.dn_resnet(x_crop.expand(-1, 3, -1, -1))
        res = res.mean(dim=2).mean(dim=2)
        return res

    def _attention_module_forward(self, h_crops):
        """Attention module forward pass"""
        batch_size, num_crops, h_dim = h_crops.size()
        h_crops_reshape = h_crops.view(batch_size * num_crops, h_dim)
        
        attn_projection = torch.sigmoid(self.mil_attn_U(h_crops_reshape)) * \
                         torch.tanh(self.mil_attn_V(h_crops_reshape))
        attn_score = self.mil_attn_w(attn_projection)
        
        attn_score_reshape = attn_score.view(batch_size, num_crops)
        attn = F.softmax(attn_score_reshape, dim=1)
        
        z_weighted_avg = torch.sum(attn.unsqueeze(-1) * h_crops, 1)
        y_crops = torch.sigmoid(self.classifier_linear(z_weighted_avg))
        
        return z_weighted_avg, attn, y_crops

    def forward(self, x_original):
        """Forward pass supporting DataParallel"""
        # Input normalization
        if isinstance(x_original, np.ndarray):
            x_original = torch.from_numpy(x_original).permute(0, 3, 1, 2).float()
        elif isinstance(x_original, torch.Tensor):
            if x_original.dim() == 3:
                x_original = x_original.unsqueeze(0)
            if x_original.size(1) not in (1, 3):
                x_original = x_original.permute(0, 3, 1, 2).contiguous()
        else:
            raise TypeError("x_original must be numpy array or torch.Tensor")

        # Global network
        h_g, saliency_map = self._global_network_forward(x_original)
        
        # Aggregation function
        y_global = self._aggregation_function_forward(saliency_map)
        
        # ROI retrieval
        small_x_locations = self._retrieve_roi_forward(x_original, self.cam_size, saliency_map)
        
        # Convert crop positions
        patch_locations = self._convert_crop_position(small_x_locations, self.cam_size, x_original)
        
        # Retrieve crops
        crops_variable = self._retrieve_crop(x_original, patch_locations, "upper_left")
        
        # Local network
        batch_size, num_crops, I, J = crops_variable.size()
        crops_variable = crops_variable.view(batch_size * num_crops, I, J).unsqueeze(1)
        h_crops = self._local_network_forward(crops_variable).view(batch_size, num_crops, -1)
        
        # Attention module
        z, patch_attns, y_local = self._attention_module_forward(h_crops)
        
        # Fusion (return the RAW logit; sigmoid is applied in the loss/eval)
        g1, _ = torch.max(h_g, dim=2)
        global_vec, _ = torch.max(g1, dim=2)
        concat_vec = torch.cat([global_vec, z], dim=1)
        y_fusion = self.fusion_dnn(concat_vec)

        # Native GMIC deep supervision: expose all three heads + the saliency map.
        #   y_fusion    : raw logits (B, num_classes)            -> BCEWithLogits in loss
        #   y_global    : top-t%% pooled sigmoid saliency (B, C)  -> probabilities
        #   y_local     : sigmoid(classifier_linear) (B, C)      -> probabilities
        #   saliency_map: sigmoid saliency (B, C, h, w)          -> probabilities
        # Output class order is [benign, malignant]; malignant = index 1.
        return y_fusion, y_global, y_local, saliency_map