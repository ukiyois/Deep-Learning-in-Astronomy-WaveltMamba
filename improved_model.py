import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict

def build_color_features(ugriz: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(ugriz.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    u, g, r, i, z = x.unbind(dim=1)
    color = torch.stack([u - g, g - r, r - i, i - z], dim=1)
    q1 = color.quantile(0.25, dim=0, keepdim=True)
    q3 = color.quantile(0.75, dim=0, keepdim=True)
    iqr = (q3 - q1).clamp_min(1e-3)
    med = color.median(dim=0, keepdim=True).values
    return (color - med) / iqr


class StandardConv2Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class SimplifiedAttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attention_weights = self.attention_conv(x)
        weighted_features = attention_weights * x
        output = weighted_features.sum(dim=[2, 3])
        weight_sum = attention_weights.sum(dim=[2, 3])
        return output / (weight_sum + 1e-8)

class FastFeatureFusion(nn.Module):
    def __init__(self, in_channels=128, conv_out=256, embed_dim=256, output_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.output_dim = output_dim
        self.conv_block = StandardConv2Layer(in_channels, conv_out, kernel_size=3)
        self.feature_proj = nn.Conv2d(conv_out, embed_dim, kernel_size=1)
        self.attention_pool = SimplifiedAttentionPooling(embed_dim)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim)
        )
        
    def forward(self, x):
        x = self.conv_block(x)
        x = self.feature_proj(x)
        x = self.attention_pool(x)
        return self.output_proj(x)


class BaseFeatureExtractor(nn.Module):
    def __init__(self, input_channels: int = 3, output_dim: int = 256, use_wavelet_mamba: bool = True):
        super().__init__()
        # WaveletMamba is the core feature extractor, always enabled
        self.use_wavelet_mamba = True  # Always True, WaveletMamba is the core
        self.conv_initial = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.wavelet_mamba_layer = WaveletMamba(
            in_channels=128,
            d_model=128,
            kernel_size=3
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.output_dim = output_dim
        self.fast_fusion = FastFeatureFusion(
            in_channels=128,
            conv_out=256,
            embed_dim=256,
            output_dim=output_dim
        )
    
    def forward(self, x: torch.Tensor, 
                return_intermediate: bool = False):
        x2 = self.conv_initial(x)
        if return_intermediate:
            x3, wavelet_intermediate = self.wavelet_mamba_layer(x2, return_intermediate=True)
        else:
            x3 = self.wavelet_mamba_layer(x2)
            wavelet_intermediate = None
        x3 = self.bn3(x3)
        # In-place operation: x2 is only saved when return_intermediate=True, otherwise can safely use +=
        if return_intermediate:
            x3 = x3 + x2  # Need to save x2, cannot use in-place
        else:
            x3 += x2  # In-place operation (x2 is no longer used)
        x3 = self.relu3(x3)
        x = self.fast_fusion(x3)
        
        if return_intermediate:
            intermediate_dict = {'x2': x2}
            if wavelet_intermediate is not None:
                intermediate_dict.update(wavelet_intermediate)
            return x, intermediate_dict
        return x


class SSMBlock(nn.Module):
    """Mamba state space model block with linear complexity sequence modeling"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        d_inner = int(self.expand * self.d_model)
        self.d_inner = d_inner
        self.input_norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv-1, groups=d_inner)
        self.gate_conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv-1, groups=d_inner)
        self.activation = nn.SiLU()
        
        self.A_log = nn.Parameter(torch.randn(d_inner, d_state))
        self.A_log.data.uniform_(-4, -1)
        
        self.B = nn.Parameter(torch.randn(d_inner, d_state))
        nn.init.xavier_uniform_(self.B)
        
        self.C = nn.Parameter(torch.randn(d_inner, d_state))
        nn.init.xavier_uniform_(self.C)
        
        self.D = nn.Parameter(torch.ones(d_inner))
        
        if dt_rank == "auto":
            dt_rank = max(16, d_model // 16)
        self.dt_rank = dt_rank
        
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank**-0.5 * dt_min
        with torch.no_grad():
            dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_proj.weight.copy_(torch.randn(d_inner, dt_rank) * dt_init_std)
            self.dt_proj.bias.copy_(inv_dt)
        
        self.A_proj = nn.Linear(d_inner, dt_rank, bias=False)
        nn.init.xavier_uniform_(self.A_proj.weight)
        self.out_proj = nn.Linear(d_inner, d_model)
    
    def _discretize(self, A, B, dt):
        d_inner, d_state = A.shape
        if dt.dim() == 1:
            dt_exp = dt.view(1, d_inner, 1)
        elif dt.dim() == 2:
            dt_exp = dt.unsqueeze(-1)
        else:
            dt_exp = dt
            if dt_exp.dim() == 3 and dt_exp.size(-1) == 1:
                pass
            else:
                raise RuntimeError(f"Unexpected dt shape: {dt.shape}")

        A_expanded = A.unsqueeze(0) * dt_exp
        A_bar = torch.exp(A_expanded)
        A_inv = (1.0 / (A + 1e-8)).unsqueeze(0)

        exp_term = A_bar - 1.0
        small_mask = torch.abs(A_expanded) < 1e-6
        exp_term = torch.where(small_mask, A_expanded, exp_term)

        dt_tiled = dt_exp.expand_as(A_expanded)
        A_inv_safe = torch.where(small_mask, dt_tiled, A_inv * exp_term)
        B_bar = A_inv_safe * B.unsqueeze(0)

        if dt.dim() == 1:
            while A_bar.dim() > 2:
                A_bar = A_bar.squeeze(0)
            while B_bar.dim() > 2:
                B_bar = B_bar.squeeze(0)
        
        return A_bar, B_bar
    
    def forward(self, x):
        x = self.input_norm(x)
        B, L, d = x.shape
        
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        x = x.transpose(1, 2).contiguous()
        x = self.conv1d(x)[..., :L]
        x = x.transpose(1, 2).contiguous()
        x = self.activation(x)
        
        z_aligned = z.transpose(1, 2).contiguous()
        z_aligned = self.gate_conv1d(z_aligned)[..., :L]
        z_aligned = z_aligned.transpose(1, 2).contiguous()
        z_aligned = self.activation(z_aligned)
        
        A = -torch.exp(self.A_log.float())
        A_input = self.A_proj(x)
        dt = self.dt_proj(A_input)
        dt = F.softplus(dt)
        
        dt_avg = dt.mean(dim=1).mean(dim=0)
        A_bar_single, B_bar_single = self._discretize(A, self.B, dt_avg)
        
        A_bar = A_bar_single.unsqueeze(0).expand(B, -1, -1)
        B_bar = B_bar_single.unsqueeze(0).expand(B, -1, -1)
        
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        y_all = torch.zeros(B, L, self.d_inner, device=x.device, dtype=x.dtype)
        
        chunk_size = 16
        for chunk_start in range(0, L, chunk_size):
            chunk_end = min(chunk_start + chunk_size, L)
            for t in range(chunk_start, chunk_end):
                x_t = x[:, t].unsqueeze(-1)
                h = A_bar * h + B_bar * x_t
                y_t = torch.einsum('bid,id->bi', h, self.C)
                y_all = y_all.index_copy(1, torch.tensor(t, device=y_all.device), y_t.unsqueeze(1))
        
        x = y_all
        x = x * z_aligned
        x = x + x * self.D.unsqueeze(0).unsqueeze(0)
        
        return self.out_proj(x)


class JointMambaScanner(nn.Module):
    def __init__(self, d_model=64, d_state=16, num_directions=4):
        super().__init__()
        self.d_model = d_model
        self.num_directions = num_directions
        self.ssm_blocks = nn.ModuleList([
            SSMBlock(d_model, d_state=d_state) for _ in range(num_directions)
        ])
    
    def forward(self, x, spatial_shape):
        B, num_dirs, seq_len, d = x.shape
        x_reshaped = x.view(B * num_dirs, seq_len, d)
        
        forward_outs = torch.stack([
            self.ssm_blocks[dir_idx](x_reshaped[dir_idx * B:(dir_idx + 1) * B])
            for dir_idx in range(num_dirs)
        ], dim=0)
        forward_outs = forward_outs.view(B * num_dirs, seq_len, d)
            
        x_flipped = x_reshaped.flip(dims=[1])
        backward_outs = torch.stack([
            self.ssm_blocks[dir_idx](x_flipped[dir_idx * B:(dir_idx + 1) * B]).flip(dims=[1])
            for dir_idx in range(num_dirs)
        ], dim=0)
        backward_outs = backward_outs.view(B * num_dirs, seq_len, d)
        
        dir_outs = (forward_outs + backward_outs) / 2
        
        if spatial_shape is None:
            raise ValueError("spatial_shape must be provided to preserve 2D spatial structure")
        
        H, W = spatial_shape
        dir_outs = dir_outs.view(B, num_dirs, seq_len, d)
        dir_outs_spatial = dir_outs.view(B, num_dirs, H, W, d)
        dir_outs_spatial = dir_outs_spatial.permute(0, 1, 4, 2, 3).contiguous()
        return dir_outs_spatial


class DirectionAggregator(nn.Module):
    """Direction aggregator: aggregates multi-directional features using gated convolution"""
    def __init__(self, num_directions=4, d_model=64):
        super().__init__()
        self.num_directions = num_directions
        self.d_model = d_model
        self.gated_conv = nn.Conv2d(num_directions * d_model, d_model, kernel_size=3, padding=1)
        self.gate_conv = nn.Conv2d(num_directions * d_model, d_model, kernel_size=3, padding=1)
        self.activation = nn.GELU()
    
    def forward(self, direction_features):
        B, num_dirs, d, H, W = direction_features.shape
        direction_concat = direction_features.reshape(B, num_dirs * d, H, W)
        gate = torch.sigmoid(self.gate_conv(direction_concat))
        conv_out = self.gated_conv(direction_concat)
        return self.activation(conv_out) * gate
class WaveletMamba(nn.Module):
    def __init__(self, in_channels=128, d_model=64, 
                 kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_directions = 4
        self.kernel_size = kernel_size
        self.direction_types = ['spiral', 'radial', 'spiral', 'tangential']
        
        gabor_base_a = self._initialize_base_gabor_filters(kernel_size, 'real')
        gabor_base_b = self._initialize_base_gabor_filters(kernel_size, 'imag')
        gabor_base_a_expanded = gabor_base_a.unsqueeze(1).expand(4, in_channels, kernel_size, kernel_size)
        gabor_base_b_expanded = gabor_base_b.unsqueeze(1).expand(4, in_channels, kernel_size, kernel_size)
        gabor_init_a = gabor_base_a_expanded + torch.randn(4, in_channels, kernel_size, kernel_size) * 0.3
        gabor_init_b = gabor_base_b_expanded + torch.randn(4, in_channels, kernel_size, kernel_size) * 0.3
        self.wavelet_filters_a = nn.Parameter(gabor_init_a)
        self.wavelet_filters_b = nn.Parameter(gabor_init_b)
        self.channel_fusion_a = nn.Conv2d(in_channels * 4, 4, kernel_size=1, groups=4)
        self.channel_fusion_b = nn.Conv2d(in_channels * 4, 4, kernel_size=1, groups=4)

        self.wavelet_proj = nn.ModuleList([
            nn.Linear(2, d_model) for _ in range(4)
        ])
     
        self.direction_bias = nn.Parameter(torch.randn(4, d_model) * 0.5)
        
        self.mamba_scanner = JointMambaScanner(
            d_model=d_model,
            d_state=16,
            num_directions=self.num_directions
        )
        
        self.direction_aggregator = DirectionAggregator(
            num_directions=self.num_directions,
            d_model=d_model
        )
        
        self.output_conv = nn.Conv2d(d_model, in_channels, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        self.skip_weight = nn.Parameter(torch.tensor(0.2))
        self.skip_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.full_res_proj = nn.Conv2d(8, d_model, kernel_size=1)
    
    def _initialize_base_gabor_filters(self, kernel_size: int, component: str) -> torch.Tensor:
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        y = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        R = torch.sqrt(X**2 + Y**2 + 1e-8)
        Theta = torch.atan2(Y, X)
        
        sigma = kernel_size / 6.0
        wavelength = kernel_size / 4.0
        
        spiral_params_init = [(0.5, 0.3), (0.3, -0.3)]
        spiral_idx = 0
        
        filters = []
        for dir_idx, dir_type in enumerate(self.direction_types):
            if dir_type == 'spiral':
                a, b = spiral_params_init[spiral_idx]
                spiral_idx += 1
                b_tensor = torch.tensor(b, dtype=torch.float32)
                
                if torch.abs(b_tensor) < 1e-6:
                    angle_field = Theta + math.pi / 2
                else:
                    spiral_base_angle = torch.atan(torch.tensor(1.0, dtype=torch.float32) / b_tensor)
                    radial_factor = torch.clamp(R / (kernel_size / 2 + 1e-6), 0, 1)
                    angle_field = Theta + spiral_base_angle + 0.3 * b_tensor * radial_factor
                angle_field = torch.remainder(angle_field + 2 * math.pi, 2 * math.pi)
                
            elif dir_type == 'radial':
                angle_field = Theta
                
            elif dir_type == 'tangential':
                angle_field = Theta + math.pi / 2
                
            else:
                raise ValueError(f"Unknown direction type: {dir_type}")
            
            cos_theta = torch.cos(angle_field)
            sin_theta = torch.sin(angle_field)
            
            x_rot = X * cos_theta + Y * sin_theta
            y_rot = -X * sin_theta + Y * cos_theta
            
            gaussian = torch.exp(-(x_rot**2 + y_rot**2) / (2 * sigma**2))
            complex_wave = torch.exp(2j * math.pi * x_rot / wavelength)
            
            if component == 'real':
                gabor_filter = gaussian * torch.real(complex_wave)
            else:
                gabor_filter = gaussian * torch.imag(complex_wave)
            
            norm = torch.norm(gabor_filter) + 1e-8
            gabor_filter = gabor_filter / norm
            filters.append(gabor_filter.unsqueeze(0))
        
        return torch.cat(filters, dim=0)
    
    def _dual_tree_wavelet_decomposition(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, height, width = x.shape
        
        filters_a = self.wavelet_filters_a
        filters_b = self.wavelet_filters_b
        
        all_filters_a = filters_a.view(4 * in_channels, 1, self.kernel_size, self.kernel_size)
        all_filters_b = filters_b.view(4 * in_channels, 1, self.kernel_size, self.kernel_size)
        x_repeated = x.repeat_interleave(4, dim=1)
        tree_a_all = F.conv2d(x_repeated, all_filters_a, padding=1, stride=1, groups=4*in_channels)
        tree_b_all = F.conv2d(x_repeated, all_filters_b, padding=1, stride=1, groups=4*in_channels)
        tree_a_all = tree_a_all.view(batch_size, 4, in_channels, height, width)
        tree_b_all = tree_b_all.view(batch_size, 4, in_channels, height, width)

        B, _, C, H, W = tree_a_all.shape
        tree_a_interleaved = tree_a_all.view(B, 4 * C, H, W)
        tree_b_interleaved = tree_b_all.view(B, 4 * C, H, W)
        tree_a_fused = self.channel_fusion_a(tree_a_interleaved)
        tree_b_fused = self.channel_fusion_b(tree_b_interleaved)
        
        tree_a_fused = tree_a_fused.unsqueeze(2)
        tree_b_fused = tree_b_fused.unsqueeze(2)
        
        directional_features = torch.cat([tree_a_fused, tree_b_fused], dim=2)
        return directional_features.view(batch_size, 8, height, width)
    
    def forward(self, x: torch.Tensor, return_intermediate=False):
        B, C, H, W = x.shape
        x_skip = x
        
        wavelet_features = self._dual_tree_wavelet_decomposition(x)
        # Deferred saving: only save full feature map when needed (for full_res_proj)
        # If return_intermediate is not needed, can avoid saving full 256×256 feature map
        
        target_hw = 8
        wavelet_features_ds = F.adaptive_avg_pool2d(wavelet_features, (target_hw, target_hw))
        H_ds, W_ds = target_hw, target_hw

        wavelet_dirs_ds = wavelet_features_ds.view(B, 4, 2, H_ds, W_ds)
        wavelet_dirs_flat_ds = wavelet_dirs_ds.permute(0, 1, 3, 4, 2).contiguous().reshape(B, 4, H_ds * W_ds, 2)
        
        wavelet_dirs_flat_all = wavelet_dirs_flat_ds.view(B * 4, H_ds * W_ds, 2)
        wavelet_projected_all = torch.stack([
            self.wavelet_proj[dir_idx](wavelet_dirs_flat_all[dir_idx * B:(dir_idx + 1) * B])
            for dir_idx in range(4)
        ], dim=1)
        wavelet_projected = wavelet_projected_all.view(B, 4, H_ds * W_ds, self.d_model)
        
        direction_bias = self.direction_bias.unsqueeze(0).unsqueeze(2)
        wavelet_projected += direction_bias  # In-place operation
        
        assert wavelet_projected.shape[-1] == self.d_model, \
            f"wavelet_proj output dimension mismatch: expected {self.d_model}, got {wavelet_projected.shape[-1]}"
        
        mamba_output = self.mamba_scanner(wavelet_projected, spatial_shape=(H_ds, W_ds))
        aggregated_ds = self.direction_aggregator(mamba_output)
        
        aggregated_full = F.interpolate(aggregated_ds, size=(H, W), mode='bilinear', align_corners=False)
        # Deferred computation: only compute full_res_proj when needed (reduce memory usage)
        wavelet_features_full_proj = self.full_res_proj(wavelet_features)
        aggregated_full += 0.3 * wavelet_features_full_proj  # In-place operation
        aggregated = aggregated_full
        B, C_out, H_out, W_out = aggregated.shape
        aggregated_flat = aggregated.permute(0, 2, 3, 1).contiguous().reshape(B * H_out * W_out, C_out)
        aggregated_norm = self.norm(aggregated_flat)
        aggregated = aggregated_norm.view(B, H_out, W_out, C_out).permute(0, 3, 1, 2).contiguous()
        
        output = self.output_conv(aggregated)
        
        skip_gate = self.skip_gate(x_skip)
        output += self.skip_weight * (skip_gate * x_skip)  # In-place operation
        
        if return_intermediate:
            return output, {'wavelet_dirs': wavelet_dirs_ds, 'wavelet_features': wavelet_features_ds}
        return output


class LearnableSinusoidalCoordEncoder(nn.Module):
    def __init__(self, input_dim: int = 2, target_dim: int = 64, 
                 freq_dim: int = 32, ra_range: tuple = (0.0, 360.0), 
                 dec_range: tuple = (-90.0, 90.0)):
        super().__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.freq_dim = freq_dim
        self.ra_range = ra_range
        self.dec_range = dec_range
        
        log_freq_min = -4.0
        log_freq_max = 1.0
        freq_init = torch.exp(torch.linspace(log_freq_min, log_freq_max, freq_dim))
        
        self.ra_frequencies = nn.Parameter(freq_init.clone())
        self.dec_frequencies = nn.Parameter(freq_init.clone())
        
        encoding_dim = 4 * freq_dim
        self.projection = nn.Sequential(
            nn.Linear(encoding_dim, target_dim * 2),
            nn.LayerNorm(target_dim * 2),
            nn.ReLU(),
            nn.Linear(target_dim * 2, target_dim),
            nn.LayerNorm(target_dim)
        )
        
        if encoding_dim != target_dim:
            self.residual_proj = nn.Linear(encoding_dim, target_dim)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        ra = coordinates[:, 0]
        dec = coordinates[:, 1]
        
        ra_min, ra_max = self.ra_range
        dec_min, dec_max = self.dec_range
        
        ra_norm = (ra - ra_min) / (ra_max - ra_min) * (2 * math.pi)
        dec_norm = (dec - dec_min) / (dec_max - dec_min) * (2 * math.pi)
        ra_norm = torch.clamp(ra_norm, 0.0, 2 * math.pi)
        dec_norm = torch.clamp(dec_norm, 0.0, 2 * math.pi)
        
        ra_norm_expanded = ra_norm.unsqueeze(-1)
        dec_norm_expanded = dec_norm.unsqueeze(-1)
        
        ra_sin = torch.sin(ra_norm_expanded * self.ra_frequencies.unsqueeze(0))
        ra_cos = torch.cos(ra_norm_expanded * self.ra_frequencies.unsqueeze(0))
        ra_encoding = torch.cat([ra_sin, ra_cos], dim=-1)
        
        dec_sin = torch.sin(dec_norm_expanded * self.dec_frequencies.unsqueeze(0))
        dec_cos = torch.cos(dec_norm_expanded * self.dec_frequencies.unsqueeze(0))
        dec_encoding = torch.cat([dec_sin, dec_cos], dim=-1)
        
        coord_encoding = torch.cat([ra_encoding, dec_encoding], dim=-1)
        coord_features = self.projection(coord_encoding)
        
        if self.residual_proj is not None and not isinstance(self.residual_proj, nn.Identity):
            residual = self.residual_proj(coord_encoding)
            coord_features += residual  # In-place operation
        
        return coord_features


class MorphHead(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        self.output_dim = output_dim
        self.intermediate_feature_dim = 128
        self.shared_feature_projection = nn.Linear(self.intermediate_feature_dim, output_dim)
        self.se_gate = nn.Sequential(
            nn.Linear(output_dim + 4 + 10, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid()
        )
        self.film_gamma = nn.Linear(output_dim, output_dim)
        self.film_beta = nn.Linear(output_dim, output_dim)
        
        self.fusion_layer = nn.Sequential(
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim)
        )
        self.residual_projection = nn.Linear(output_dim, output_dim)
        
    def forward(self, predicted_class=None, color_features=None, shared_features=None):
        assert shared_features is not None and 'x2' in shared_features, "MorphHead requires shared_features['x2']"
        x2_shared = shared_features['x2']
        assert x2_shared.shape[1] == self.intermediate_feature_dim, \
            f"Expected intermediate feature dim={self.intermediate_feature_dim}, got {x2_shared.shape[1]}"
        base_features_pooled = F.adaptive_avg_pool2d(x2_shared, (1, 1)).flatten(1)
        base_features = self.shared_feature_projection(base_features_pooled)
        
        if predicted_class is not None:
            if predicted_class.dim() == 0:
                predicted_class = predicted_class.view(1)
            one_hot = F.one_hot(predicted_class.long(), num_classes=10).float()
        else:
            one_hot = torch.zeros(base_features.size(0), 10, device=base_features.device)
        color_in = color_features if color_features is not None else torch.zeros(base_features.size(0), 4, device=base_features.device)
        gate_input = torch.cat([base_features, color_in, one_hot], dim=1)
        gate = self.se_gate(gate_input)
        class_attended = base_features * gate
        
        gamma = self.film_gamma(class_attended)
        beta = self.film_beta(class_attended)
        film_applied = gamma * base_features + beta

        fused_features = torch.cat([film_applied, base_features], dim=1)
        output_features = self.fusion_layer(fused_features)
        
        residual = self.residual_projection(base_features)
        final_features = output_features + residual
        
        return final_features

class FeatureHub(nn.Module):
    def __init__(self, use_morph=True, use_wavelet_mamba=True, use_coord_encoding=True, device='cpu'):
        super().__init__()
        self.use_morph = use_morph
        self.use_wavelet_mamba = use_wavelet_mamba  # Always True (core feature extractor)
        self.use_coord_encoding = use_coord_encoding
        if isinstance(device, str):
            self.device = torch.device(device if device == 'cpu' or torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.wavelet_encoder = BaseFeatureExtractor(
            input_channels=3, 
            output_dim=256,
            use_wavelet_mamba=use_wavelet_mamba
        )
        if self.use_morph:
            self.morphology_analyzer = MorphHead(
                output_dim=256
            )
        else:
            self.morphology_analyzer = None
        if self.use_coord_encoding:
            self.coordinate_encoder = LearnableSinusoidalCoordEncoder(
            input_dim=2, 
                target_dim=64,
                freq_dim=32
        )
        else:
            # Fixed coordinate encoding (simple linear projection)
            self.coordinate_encoder = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(),
                nn.Linear(64, 64)
            )
        
        projection_input_dim = 512 if self.use_morph else 256
        self.image_to_task_projection = nn.Sequential(
            nn.Linear(projection_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )

        self.vib_mu = nn.Linear(256, 256)
        self.vib_logvar = nn.Linear(256, 256)
        
        self.register_parameter('prior_mu', nn.Parameter(torch.zeros(256)))
        self.register_parameter('prior_logvar', nn.Parameter(torch.zeros(256)))
        
        self.color_magnitude_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
    
    def forward(self, images, coordinates, predicted_class=None, return_intermediate=False):
        batch_size = images.size(0)
        
        color_magnitudes = self.color_magnitude_extractor(images)
        color_features_4d = build_color_features(color_magnitudes)
        
        need_intermediate = return_intermediate or (self.use_morph and self.morphology_analyzer is not None)
        wavelet_output = self.wavelet_encoder(images, return_intermediate=need_intermediate)
        if need_intermediate:
            wavelet_features, intermediate_features = wavelet_output
        else:
            wavelet_features = wavelet_output
            intermediate_features = None
        
        if self.use_morph and self.morphology_analyzer is not None:
            if predicted_class is None:
                with torch.no_grad():
                    g_r = color_features_4d[:, 1]
                    weak_pred = torch.zeros(batch_size, dtype=torch.long, device=images.device)
                    weak_pred[g_r > 0.8] = 3
                    weak_pred[(g_r > 0.5) & (g_r <= 0.8)] = 6
                predicted_class_for_morph = weak_pred
            else:
                predicted_class_for_morph = predicted_class
            
            morph_features = self.morphology_analyzer(
                predicted_class_for_morph, 
                color_features_4d,
                shared_features=intermediate_features
            )
        else:
            morph_features = torch.zeros(batch_size, 256, device=images.device)
        
        coord_features = self.coordinate_encoder(coordinates)
        
        if self.use_morph:
            image_features = torch.cat([wavelet_features, morph_features], dim=1)
        else:
            image_features = wavelet_features
        
        proj_features = self.image_to_task_projection(image_features)
        
        mu = self.vib_mu(proj_features)
        logvar = self.vib_logvar(proj_features)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_shared = mu + eps * std
        cls_features = z_shared
        
        result = {
            'classification_features': cls_features,
            'vib_mu': mu,
            'vib_logvar': logvar,
            'prior_mu': self.prior_mu,
            'prior_logvar': self.prior_logvar,
            'raw_features': {
                'wavelet': wavelet_features,
                'morph': morph_features,
                'coord': coord_features,
                'color': color_features_4d,
                'color_magnitudes': color_magnitudes
            }
        }
        
        if return_intermediate and isinstance(intermediate_features, dict):
            result['intermediate_features'] = intermediate_features
        
        return result

class TaskRelationshipModel(nn.Module):
    def __init__(self, num_tasks=2, feature_dim=256):
        super().__init__()
        self.num_tasks = num_tasks
        self.feature_dim = feature_dim
        
        relationship_init = torch.eye(num_tasks)
        relationship_init = relationship_init + 0.1 * torch.randn_like(relationship_init)
        self.relationship_matrix = nn.Parameter(relationship_init)
    
    def forward(self, cls_features, redshift_features):
        task_features = torch.stack([cls_features, redshift_features], dim=1)
        relationship_norm = F.softmax(self.relationship_matrix, dim=-1)
        related_features = torch.matmul(relationship_norm, task_features)
        
        enhanced_cls_features = related_features[:, 0, :]
        enhanced_redshift_features = related_features[:, 1, :]
        
        return enhanced_cls_features, enhanced_redshift_features, relationship_norm


class UnifiedTaskHeads(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        
        cls_input_dim = 256
        
        self.cls_layer1 = nn.Sequential(
            nn.BatchNorm1d(cls_input_dim),
            nn.ReLU(),
            nn.Linear(cls_input_dim, 128)
        )
        self.cls_residual1 = nn.Linear(cls_input_dim, 128)
        
        self.class_projection = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        self.cls_norm2 = nn.LayerNorm(128)
    
    def forward(self, features):
        cls_input = features['classification_features']
        
        cls_l1_out = self.cls_layer1(cls_input)
        cls_res1 = self.cls_residual1(cls_input)
        cls_l1_res = cls_l1_out + cls_res1
        cls_l1_norm = self.cls_norm2(cls_l1_res)
        
        cls_logits = self.class_projection(cls_l1_norm)
        
        return {
            'cls_logits': cls_logits
        }

COLOR_G_R_MIN = 0.2
COLOR_G_R_MAX = 1.5
COLOR_STD_THRESHOLD = 0.1
COLOR_STD_FACTOR = 0.3
COLOR_STD_FALLBACK = 0.1


class ConfidenceWeighter(nn.Module):
    """Quality weighter based on color indices"""
    
    G_R_MIN = COLOR_G_R_MIN
    G_R_MAX = COLOR_G_R_MAX
    STD_THRESHOLD = COLOR_STD_THRESHOLD
    STD_FACTOR = COLOR_STD_FACTOR
    STD_FALLBACK = COLOR_STD_FALLBACK
    
    def __init__(self):
        super().__init__()
    
    def forward(self, color_features):
        """
        Define sample quality based on color indices, returns stratified weights
        
        Args:
            color_features: [batch, 4] color features (u-g, g-r, r-i, i-z)
        
        Returns:
            sample_weights: [batch] sample weights
            quality_masks: dict quality masks
            color_quality_metric: [batch] color quality metric for monitoring
        """
        batch_size = color_features.shape[0]
        device = color_features.device
        
        g_r_color = color_features[:, 1]
        
        quality_masks = compute_color_quality_masks(g_r_color, use_percentile_fallback=False)
        redshift_high_mask = quality_masks['high']
        redshift_med_mask = quality_masks['med']
        redshift_low_mask = quality_masks['low']
        
        sample_weights = torch.ones(batch_size, device=device)
        sample_weights[redshift_high_mask] = 1.0
        sample_weights[redshift_med_mask] = 0.7
        sample_weights[redshift_low_mask] = 0.3
        
        all_covered = redshift_high_mask | redshift_med_mask | redshift_low_mask
        if not all_covered.all():
            uncovered_count = (batch_size - all_covered.sum()).item()
            import warnings
            warnings.warn(
                f"ConfidenceWeighter: {uncovered_count}/{batch_size} samples not covered by quality masks. "
                f"These samples will have weight 1.0 (default).",
                UserWarning
            )
        
        valid_color_range = redshift_high_mask | redshift_med_mask | redshift_low_mask
        if valid_color_range.any():
            color_center = (self.G_R_MIN + self.G_R_MAX) / 2.0
            color_distance = torch.abs(g_r_color - color_center)
            color_quality_metric = 1.0 - torch.clamp(color_distance / (self.G_R_MAX - self.G_R_MIN), 0.0, 1.0)
        else:
            color_quality_metric = torch.ones(batch_size, device=device)
        
        return sample_weights, quality_masks, color_quality_metric


def compute_color_quality_masks(g_r_color, use_percentile_fallback=False):
    batch_size = g_r_color.shape[0]
    device = g_r_color.device
    
    valid_color_range = (g_r_color >= COLOR_G_R_MIN) & (g_r_color <= COLOR_G_R_MAX)
    invalid_color_mask = ~valid_color_range
    
    if valid_color_range.sum() > 0:
        color_mean = g_r_color[valid_color_range].mean()
        valid_count = valid_color_range.sum().item()
        if valid_count > 1:
            color_std = g_r_color[valid_color_range].std()
        else:
            color_std = torch.tensor(COLOR_STD_FALLBACK, device=device)
        
        threshold_offset = COLOR_STD_FACTOR * color_std if color_std > COLOR_STD_THRESHOLD else COLOR_STD_FALLBACK
        high_color_threshold = color_mean + threshold_offset
        low_color_threshold = color_mean - threshold_offset
        
        redshift_high_mask = valid_color_range & (g_r_color >= high_color_threshold)
        redshift_low_mask = valid_color_range & (g_r_color <= low_color_threshold)
        redshift_med_mask = valid_color_range & (g_r_color > low_color_threshold) & (g_r_color < high_color_threshold)
    else:
        if use_percentile_fallback:
            quantiles = torch.quantile(g_r_color, torch.tensor([0.3, 0.7], device=device))
            q30, q70 = quantiles[0], quantiles[1]
            redshift_high_mask = g_r_color >= q70
            redshift_low_mask = g_r_color <= q30
            redshift_med_mask = (g_r_color > q30) & (g_r_color <= q70)
        else:
            redshift_high_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            redshift_low_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            redshift_med_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    redshift_low_mask = redshift_low_mask | invalid_color_mask
    
    all_classified = redshift_high_mask | redshift_med_mask | redshift_low_mask
    if not all_classified.all():
        uncovered = ~all_classified
        redshift_low_mask = redshift_low_mask | uncovered
    
    return {
        'high': redshift_high_mask,
        'med': redshift_med_mask,
        'low': redshift_low_mask
    }




class HellingerKantorovichDistance(nn.Module):
    """
    Hellinger-Kantorovich distance computation module.
    
    Captures both geometric transport (Wasserstein component) for position bias
    and information geometry (Fisher-Rao component) for shape bias through
    marginal measure regularization in a unified metric space.
    Uses Sinkhorn algorithm for efficient computation.
    """
    
    def __init__(self, delta: float = 1.0, lambda_reg: float = 0.5, 
                 max_iter: int = 50, eps: float = 1e-4, n_bins: int = 40):
        """
        Args:
            delta: HK distance parameter controlling balance between geometric transport and information difference.
                  Converges to Wasserstein distance when delta->0, and to Hellinger distance when delta->+inf.
            lambda_reg: Entropy regularization parameter controlling Sinkhorn convergence speed and accuracy.
            max_iter: Maximum iterations for Sinkhorn algorithm.
            eps: Convergence threshold.
            n_bins: Number of histogram bins when converting redshift values to distributions.
        """
        super().__init__()
        self.delta = delta
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.eps = eps
        self.n_bins = n_bins
    
        self.register_buffer('_cached_cost_matrix', None)
        self._cached_z_range = None
    
    def values_to_distribution(self, values: torch.Tensor, z_min: float = 1e-6, 
                               z_max: float = 2.0) -> torch.Tensor:
        """
        Convert redshift values to histogram distribution (probability mass function).
        
        Args:
            values: [batch] redshift values
            z_min: minimum redshift value
            z_max: maximum redshift value
        
        Returns:
            dist: [batch, n_bins] distribution for each sample
        """
        batch_size = values.shape[0]
        device = values.device
        
        bin_edges = torch.linspace(z_min, z_max, self.n_bins + 1, device=device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        
        sigma = (z_max - z_min) / (self.n_bins * 2.0)
        values_expanded = values.unsqueeze(1)
        bin_centers_expanded = bin_centers.unsqueeze(0)
        
        squared_diff = (values_expanded - bin_centers_expanded) ** 2
        gaussian_weights = torch.exp(-squared_diff / (2 * sigma ** 2))
        
        dist = gaussian_weights / (gaussian_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        return dist
    
    def _compute_cost_matrix(self, z_min: float, z_max: float, device: torch.device) -> torch.Tensor:
   
        if (self._cached_cost_matrix is not None and 
            self._cached_z_range == (z_min, z_max) and
            self._cached_cost_matrix.device == device):
            return self._cached_cost_matrix
        
        bin_centers = torch.linspace(z_min, z_max, self.n_bins, device=device)
        bin_centers_i = bin_centers.unsqueeze(1)
        bin_centers_j = bin_centers.unsqueeze(0)
        squared_dist = (bin_centers_i - bin_centers_j) ** 2
        cost_matrix = squared_dist + self.delta ** 2
        
        self._cached_cost_matrix = cost_matrix
        self._cached_z_range = (z_min, z_max)
        
        return cost_matrix
    
    def _normalize_distribution(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Normalize distribution (ensure probability sum equals 1).
        
        Args:
            dist: [..., n_bins] distribution tensor
        
        Returns:
            normalized_dist: [..., n_bins] normalized distribution
        """
        return dist / (dist.sum(dim=-1, keepdim=True) + 1e-8)
    
    def sinkhorn_hk_distance(self, a: torch.Tensor, b: torch.Tensor, 
                            cost_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute HK distance using Sinkhorn algorithm.
        
        Args:
            a: [batch, n] weights of first distribution
            b: [batch, m] weights of second distribution (normalized probability mass)
            cost_matrix: [n, m] cost matrix, cost_matrix[i,j] = d^2(x_i, y_j) + delta^2
        
        Returns:
            hk_distance: [batch] HK distance values
        """
        device = a.device
        batch_size = a.shape[0]
        n, m = cost_matrix.shape
        
        u = torch.ones(batch_size, n, device=device)
        v = torch.ones(batch_size, m, device=device)
        
        if torch.isnan(a).any() or torch.isinf(a).any():
            return torch.zeros(batch_size, device=device)
        if torch.isnan(b).any() or torch.isinf(b).any():
            return torch.zeros(batch_size, device=device)
        
        if torch.isnan(cost_matrix).any() or torch.isinf(cost_matrix).any():
            return torch.zeros(batch_size, device=device)
        
        cost_normalized = -cost_matrix / self.lambda_reg
        cost_normalized = torch.clamp(cost_normalized, min=-100, max=100)
        
        K = torch.exp(cost_normalized)
        
        u_norm = None
        for iteration in range(self.max_iter):
            if iteration % 5 == 0 and iteration > 0:
                u_prev_norm = u_norm
                u_norm = torch.norm(u, dim=1)
                diff = (u_norm - u_prev_norm).abs().max()
                if diff < self.eps:
                    break
            elif iteration == 0:
                u_norm = torch.norm(u, dim=1)
            
            Kv = torch.einsum('ij,bj->bi', K, v)
            u = a / (Kv + self.eps)
            
            KTu = torch.einsum('ij,bi->bj', K, u)
            v = b / (KTu + self.eps)
        
        P = torch.einsum('bi,ij,bj->bij', u, K, v)
        hk_squared = torch.einsum('bij,ij->b', P, cost_matrix)
        hk_distance = torch.sqrt(torch.clamp(hk_squared, min=0.0) + 1e-8)
        
        return hk_distance
    
    def forward(self, pred_values: torch.Tensor, target_values: torch.Tensor,
                z_min: float = 1e-6, z_max: float = 2.0, return_components: bool = False):
        """
        Compute HK distance between predicted and target values.
        
        Args:
            pred_values: [batch] predicted redshift values
            target_values: [batch] true redshift values
            z_min: minimum redshift value
            z_max: maximum redshift value
            return_components: whether to return Wasserstein and Fisher-Rao components
        
        Returns:
            hk_loss: scalar HK distance loss
            If return_components=True, returns (hk_loss, wasserstein_contrib, fisher_rao_contrib)
        """
        pred_dist = self.values_to_distribution(pred_values, z_min, z_max)
        target_dist = self.values_to_distribution(target_values, z_min, z_max)
        
        cost_matrix = self._compute_cost_matrix(z_min, z_max, pred_values.device)
        
        batch_size = pred_values.shape[0]
        
        pred_dist_norm = self._normalize_distribution(pred_dist)
        target_dist_norm = self._normalize_distribution(target_dist)
        
        valid_mask = (
            (~torch.isnan(pred_dist_norm).any(dim=1)) & 
            (~torch.isinf(pred_dist_norm).any(dim=1)) &
            (pred_dist_norm.sum(dim=1) >= 1e-8) &
            (~torch.isnan(target_dist_norm).any(dim=1)) &
            (~torch.isinf(target_dist_norm).any(dim=1)) &
            (target_dist_norm.sum(dim=1) >= 1e-8)
        )
        
        if not valid_mask.any():
            if return_components:
                return torch.tensor(0.0, device=pred_values.device), \
                       torch.tensor(0.0, device=pred_values.device), \
                       torch.tensor(0.0, device=pred_values.device)
            return torch.tensor(0.0, device=pred_values.device)
        
        if valid_mask.all():
            hk_distances = self.sinkhorn_hk_distance(pred_dist_norm, target_dist_norm, cost_matrix)
        else:
            hk_distances = torch.zeros(batch_size, device=pred_values.device)
            if valid_mask.any():
                valid_indices = torch.where(valid_mask)[0]
                hk_distances[valid_indices] = self.sinkhorn_hk_distance(
                    pred_dist_norm[valid_indices], 
                    target_dist_norm[valid_indices], 
                    cost_matrix
                )
        
        hk_loss = hk_distances.mean()
        
        if return_components:
            bin_centers = torch.linspace(z_min, z_max, self.n_bins, device=pred_values.device)
            bin_centers_i = bin_centers.unsqueeze(1)
            bin_centers_j = bin_centers.unsqueeze(0)
            squared_dist_matrix = (bin_centers_i - bin_centers_j) ** 2
            
            delta_squared_matrix = torch.full((self.n_bins, self.n_bins), self.delta ** 2, device=pred_values.device)
            
            if valid_mask.all():
                wasserstein_distances = self.sinkhorn_hk_distance(pred_dist_norm, target_dist_norm, squared_dist_matrix)
                fisher_rao_distances = self.sinkhorn_hk_distance(pred_dist_norm, target_dist_norm, delta_squared_matrix)
            else:
                wasserstein_distances = torch.zeros(batch_size, device=pred_values.device)
                fisher_rao_distances = torch.zeros(batch_size, device=pred_values.device)
                if valid_mask.any():
                    valid_indices = torch.where(valid_mask)[0]
                    wasserstein_distances[valid_indices] = self.sinkhorn_hk_distance(
                        pred_dist_norm[valid_indices], 
                        target_dist_norm[valid_indices], 
                        squared_dist_matrix
                    )
                    fisher_rao_distances[valid_indices] = self.sinkhorn_hk_distance(
                        pred_dist_norm[valid_indices], 
                        target_dist_norm[valid_indices], 
                        delta_squared_matrix
                    )
            
            return hk_loss, wasserstein_distances.mean(), fisher_rao_distances.mean()
        
        return hk_loss


def _flatten_tensor(t: torch.Tensor) -> torch.Tensor:
    """Flatten tensor to 1D, handling various shape cases"""
    if t.dim() > 1:
        t = t.squeeze()
        if t.dim() > 1:
            if t.size(0) == t.size(1):
                t = torch.diagonal(t, 0)
            else:
                t = t.view(-1)
    return t


class UnifiedLossFunction(nn.Module):
    """Unified loss function integrating HK distance and LSI-enhanced VIB"""
    
    def __init__(self, num_classes=10, feature_dim=256, 
                 lambda_redshift=0.5,
                 lambda_vib: float = 0.1,
                 use_hk: bool = True, lambda_hk: float = 0.035,
                 hk_delta: float = 1.0, hk_lambda_reg: float = 0.1,
                 use_lsi_vib: bool = True, lsi_c: float = 1.0, lsi_weight: float = 0.01):
        super().__init__()
        self.num_classes = num_classes
        self.use_hk = use_hk
        self.lambda_hk = lambda_hk
        
        self.use_lsi_vib = use_lsi_vib
        self.lsi_c_log = nn.Parameter(torch.log(torch.clamp(torch.tensor(lsi_c), min=1e-6)))
        self.lsi_weight = lsi_weight
        
        self.lambda_redshift = lambda_redshift
        self.lambda_vib = lambda_vib
        self.feature_dim = feature_dim
        
        self.gamma = 2.5
        self.alpha = 0.6
        self.alpha_class0 = 0.85
        self.confidence_weighting = ConfidenceWeighter()
        
        self._hk_delta = hk_delta
        self._hk_lambda_reg = hk_lambda_reg
        self.hk_distance = HellingerKantorovichDistance(
            delta=hk_delta,
            lambda_reg=hk_lambda_reg,
            max_iter=50,
            eps=1e-4,
            n_bins=40
        ) if use_hk else None

        self._custom_weights = None
        self._current_epoch = 0
        self._total_epochs = 80
        
    def set_hk_enabled(self, enabled: bool):
        self.use_hk = enabled
        if enabled and self.hk_distance is None:
            self.hk_distance = HellingerKantorovichDistance(
                delta=getattr(self, '_hk_delta', 1.0),
                lambda_reg=getattr(self, '_hk_lambda_reg', 0.5),
                max_iter=50,
                eps=1e-4,
                n_bins=40
            )
    
    
    def set_training_stage(self, current_epoch: int, total_epochs: int = 100):

        self._current_epoch = max(0, current_epoch)
        self._total_epochs = max(1, total_epochs)
    
    def set_weights(self, cls_weight, redshift_weight):
        """Set custom loss weights"""
        self._custom_weights = torch.tensor([cls_weight, redshift_weight])
    
    def clear_weights(self):
        """Clear custom weights, restore learnable weights"""
        self._custom_weights = None
    
    def set_regularization_weights(self, lambda_vib=None, lsi_weight=None, lambda_hk=None):
        """Dynamically adjust regularization weights"""
        if lambda_vib is not None:
            self.lambda_vib = lambda_vib
        if lsi_weight is not None:
            self.lsi_weight = lsi_weight
        if lambda_hk is not None:
            self.lambda_hk = lambda_hk
    
    def adaptive_focal_loss(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p = torch.exp(-ce_loss)
        
        mean_ce = ce_loss.mean()
        difficulty_factor = torch.clamp((mean_ce - 0.5) * 0.5, -1.0, 1.0)
        dynamic_gamma = self.gamma + difficulty_factor
        dynamic_gamma = torch.clamp(dynamic_gamma, 1.0, 5.0)
        
        alpha = torch.ones(self.num_classes, device=logits.device) * self.alpha
        alpha[0] = self.alpha_class0
        alpha = alpha[targets]
        
        focal_loss = alpha * (1 - p) ** dynamic_gamma * ce_loss
        
        return focal_loss.mean(), dynamic_gamma.item()
    
    def forward(self, outputs, targets):
        cls_loss, gamma = self.adaptive_focal_loss(
            outputs['cls_logits'], 
            targets['labels']
        )
        
        redshift_final = outputs['redshift_final']
        red_target = targets['redshift']
        
        redshift_final = _flatten_tensor(redshift_final)
        red_target = _flatten_tensor(red_target)
        
        if redshift_final.size(0) != red_target.size(0):
            raise RuntimeError(
                f"Redshift shape mismatch: redshift_final={redshift_final.shape}, "
                f"red_target={red_target.shape}"
            )
        
        red_target_safe = torch.clamp(red_target, min=1e-6)
        redshift_final_safe = torch.clamp(redshift_final, min=1e-6)
        
        log_red_target = torch.log1p(red_target_safe)
        log_red_pred = torch.log1p(redshift_final_safe)
        
        redshift_loss_base = F.mse_loss(log_red_pred, log_red_target, reduction='none')
        if 'features' not in outputs:
            raise KeyError("outputs must contain 'features' key for color-based weighting")
        if 'raw_features' not in outputs['features']:
            raise KeyError("outputs['features'] must contain 'raw_features' key for color-based weighting")
        if 'color' not in outputs['features']['raw_features']:
            raise KeyError("outputs['features']['raw_features'] must contain 'color' key for color-based weighting")
        
        color_features_for_weighting = outputs['features']['raw_features']['color']
        sample_weights, quality_masks, color_quality_metric = self.confidence_weighting(color_features_for_weighting)
        
        redshift_loss_weighted = (sample_weights * redshift_loss_base)
        redshift_loss = redshift_loss_weighted.mean()
        
        hk_loss = torch.tensor(0.0, device=cls_loss.device)
        if self.use_hk and self.hk_distance is not None:
            try:
                hk_pred_values = torch.clamp(redshift_final_safe, min=1e-6)
                hk_target_values = torch.clamp(red_target_safe, min=1e-6)
                z_max_dynamic = max(2.0, hk_pred_values.max().item() * 1.2, hk_target_values.max().item() * 1.2)
                z_max_dynamic = min(z_max_dynamic, 3.0)
                hk_loss = self.hk_distance(
                    pred_values=hk_pred_values,
                    target_values=hk_target_values,
                    z_min=1e-6,
                    z_max=z_max_dynamic
                )
            except Exception as e:
                import warnings
                warnings.warn(f"HK distance computation failed: {str(e)}. Setting HK loss to zero.", UserWarning)
                hk_loss = torch.tensor(0.0, device=cls_loss.device)



        vib_kl = torch.tensor(0.0, device=cls_loss.device)
        lsi_enhancement = torch.tensor(0.0, device=cls_loss.device)
        try:
            if 'features' in outputs and 'vib_mu' in outputs['features'] and 'vib_logvar' in outputs['features']:
                mu = outputs['features']['vib_mu']
                logvar = outputs['features']['vib_logvar']
                prior_mu = outputs['features']['prior_mu']
                prior_logvar = outputs['features']['prior_logvar']
                
                prior_var = torch.exp(prior_logvar)
                
                mu_diff = mu - prior_mu.unsqueeze(0)
                var_posterior = torch.exp(logvar)
                
                kl_per_dim = 0.5 * (
                    prior_logvar.unsqueeze(0) - logvar +
                    (var_posterior + mu_diff**2) / (prior_var.unsqueeze(0) + 1e-8) - 1.0
                )
                vib_kl = torch.mean(torch.sum(kl_per_dim, dim=1)) / self.feature_dim
                
                if self.use_lsi_vib:
                    kl_per_sample = torch.sum(kl_per_dim, dim=1) / self.feature_dim
                    lsi_c_value = torch.exp(self.lsi_c_log)
                    lsi_per_sample = torch.sqrt(lsi_c_value * torch.clamp(kl_per_sample, min=1e-8))
                    lsi_term = lsi_per_sample.mean()
                    lsi_enhancement = self.lsi_weight * lsi_term
        except Exception:
            vib_kl = torch.tensor(0.0, device=cls_loss.device)
            lsi_enhancement = torch.tensor(0.0, device=cls_loss.device)
        
        # Define causal_weights outside try-except to ensure it's always defined
        causal_weights = self._custom_weights.to(cls_loss.device) if self._custom_weights is not None else torch.tensor([0.75, 0.25], device=cls_loss.device)
        
        task_losses = torch.stack([cls_loss, redshift_loss])
        weighted_task_loss = torch.sum(causal_weights * task_losses)
        
        lam_vib = self.lambda_vib
        
        total_loss = (
            weighted_task_loss +
            (self.lambda_hk * hk_loss if self.use_hk else 0.0) +
            lam_vib * vib_kl +
            lsi_enhancement
        )
        
        causal_task_weights = causal_weights.detach()
        lsi_c_value_for_monitoring = torch.exp(self.lsi_c_log).detach() if self.use_lsi_vib else torch.tensor(0.0, device=cls_loss.device)
        
        loss_dict = {
            'cls_loss': cls_loss.detach(),
            'redshift_loss': redshift_loss.detach(),
            'focal_gamma': gamma,
            'total_loss': total_loss.detach(),
            'lsi_enhancement': lsi_enhancement.detach(),
            'lsi_c_learned': lsi_c_value_for_monitoring,
            'hk_loss': hk_loss.detach(),
            'color_quality_mean': color_quality_metric.mean().detach(),
            'causal_task_weights': causal_task_weights,
            'vib_kl': vib_kl.detach(),
        }
        
        return total_loss, loss_dict


class GalaxyModel(nn.Module):
    def __init__(self, num_classes: int = 10, device: str = 'cuda', 
                 use_morph: bool = True, use_wavelet_mamba: bool = True,
                 use_task_relationship: bool = True, use_coord_encoding: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.use_morph = use_morph
        self.use_wavelet_mamba = use_wavelet_mamba  # Always True (core feature extractor)
        self.use_task_relationship = use_task_relationship
        self.use_coord_encoding = use_coord_encoding
        if device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        self.feature_extractor = FeatureHub(
            use_morph=use_morph,
            use_wavelet_mamba=use_wavelet_mamba,
            use_coord_encoding=use_coord_encoding,
            device=self.device
        )
        
        self.task_heads = UnifiedTaskHeads(
            num_classes=num_classes
        )
        
        if self.use_task_relationship:
            self.task_relationship = TaskRelationshipModel(
                num_tasks=2,
                feature_dim=256
            )
        else:
            self.task_relationship = None
        
        # Redshift prediction branch (used in both cases)
        self.redshift_image_branch = nn.Sequential(
            nn.Linear(324, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    
        self.redshift_scale = nn.Parameter(torch.tensor(2.0))
 
        with torch.no_grad():
            if hasattr(self, 'redshift_image_branch') and self.redshift_image_branch is not None:
                self.redshift_image_branch[-1].bias.fill_(0.5)
        
        feature_dim = 256
        # HK distance will be controlled via set_hk_enabled() during training
        self.loss_function = UnifiedLossFunction(
            num_classes=num_classes,
            feature_dim=feature_dim,
            lambda_redshift=0.5,
            lambda_vib=0.30,
            use_hk=False,  # Will be enabled via set_hk_enabled() for ablation
            lambda_hk=0.035,
            hk_delta=1.0,
            hk_lambda_reg=0.5,
            use_lsi_vib=True,
            lsi_c=1.0,
            lsi_weight=0.15
        )

        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Improved weight initialization"""
        redshift_output_layer = None
        
        for name, module in self.named_modules():
            if isinstance(module, nn.LazyLinear):
                continue
            if isinstance(module, nn.Linear):
                try:
                    if module.weight.dim() >= 2:
                        nn.init.xavier_uniform_(module.weight, gain=0.8)
                    else:
                        nn.init.normal_(module.weight, mean=0.0, std=0.015)
                except Exception as e:
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
                if module.bias is not None:
                    if 'redshift_image_branch' in name and module.out_features == 1:
                        redshift_output_layer = module
                        nn.init.constant_(module.bias, 0.5)
                    else:
                        nn.init.constant_(module.bias, 0.0)
            
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, images: torch.Tensor, coordinates: torch.Tensor, 
                return_intermediate: bool = False) -> Dict[str, torch.Tensor]:
        
        batch_size = images.size(0)
        
        features = self.feature_extractor(
            images, 
            coordinates, 
            predicted_class=None,
            return_intermediate=return_intermediate
        )
        
        cls_features_vib = features['classification_features']
        wavelet_features_raw = features['raw_features']['wavelet']
        coord_features = features['raw_features']['coord']
        color_features_4d = features['raw_features']['color']
        
        if self.use_task_relationship:
            enhanced_cls_features, enhanced_redshift_features, relationship_matrix = self.task_relationship(
                cls_features=cls_features_vib,
                redshift_features=wavelet_features_raw
            )
        else:
            # Without task relationship, use original features directly
            enhanced_cls_features = cls_features_vib
            enhanced_redshift_features = wavelet_features_raw
            relationship_matrix = None
        
        enhanced_features = features.copy()
        enhanced_features['classification_features'] = enhanced_cls_features
        task_outputs = self.task_heads(enhanced_features)
        
        image_redshift_input = torch.cat([
            enhanced_redshift_features,
            coord_features,
            color_features_4d
        ], dim=1)
        
        z_raw = self.redshift_image_branch(image_redshift_input).squeeze(-1)
        z_raw = _flatten_tensor(z_raw)
        
        final_redshift = F.softplus(z_raw) * torch.clamp(self.redshift_scale, min=0.5)
        final_redshift = torch.clamp(final_redshift, min=1e-6)
        final_redshift = _flatten_tensor(final_redshift)
        
        outputs = {
            'cls_logits': task_outputs['cls_logits'],
            'redshift_final': final_redshift,
            'redshift_pred': final_redshift,
            'features': features,
            'task_outputs': task_outputs,
            'relationship_matrix': relationship_matrix
        }
        
        if return_intermediate and 'intermediate_features' in features:
            outputs['intermediate_features'] = features['intermediate_features']
        
        return outputs
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.loss_function(outputs, targets)

