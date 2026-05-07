import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Block

class PositionalEncoding(nn.Module):
    """对应文档公式(5-1)：可学习二维位置编码，体现节点(p)和分组(b)位置"""
    def __init__(self, d_model, num_nodes, num_groups):
        super().__init__()
        # pos_embedding: [num_nodes, num_groups, d_model]，对应文档e^(pos)_(p,b)
        self.pos_embedding = nn.Parameter(torch.randn(num_nodes, num_groups, d_model))
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        # x: [batch_size, num_nodes, num_groups, d_model]
        # 位置编码与特征编码相加（文档公式5-1：z = E*x + e^(pos)）
        return x * self.scale + self.pos_embedding
    
    
class SeparableSelfAttention(nn.Module):
    """对应文档维度分离自注意力：先分组维(Token)、后节点维(视图)"""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead  # 每个注意力头维度（文档D_h）

        # 分组维自注意力（对应文档MSA^se：同一视图内的Token关联，公式5-17）
        self.group_q = nn.Linear(d_model, d_model)
        self.group_k = nn.Linear(d_model, d_model)
        self.group_v = nn.Linear(d_model, d_model)

        # 节点维自注意力（对应文档MSA^sa：跨视图的相同Token关联，公式5-18）
        self.node_q = nn.Linear(d_model, d_model)
        self.node_k = nn.Linear(d_model, d_model)
        self.node_v = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)  # 文档公式(5-14)的线性映射W_o
        self.dropout = nn.Dropout(dropout)
        self.norm_group = nn.LayerNorm(d_model)  # 文档预归一化LN()
        self.norm_node = nn.LayerNorm(d_model) 

    def scaled_dot_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """对应文档公式(5-5)：缩放点积注意力计算"""
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weight = F.softmax(attn_score, dim=-1)  # 文档SM()
        attn_weight = self.dropout(attn_weight)
        return torch.matmul(attn_weight, v)  # 文档公式(5-6)的特征加权

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, num_nodes, num_groups, d_model]
        batch_size, num_nodes, num_groups, _ = x.shape

        # -------------------------- 1. 分组维自注意力（Token维，公式5-13）--------------------------
        # 维度调整：[batch_size*num_nodes, num_groups, d_model]（单视图内计算Token关联）
        x_group = x.reshape(batch_size * num_nodes, num_groups, self.d_model)
        q_g = self.group_q(self.norm_group(x_group)).reshape(batch_size * num_nodes, num_groups, self.nhead, self.d_k).transpose(1, 2)
        k_g = self.group_k(self.norm_group(x_group)).reshape(batch_size * num_nodes, num_groups, self.nhead, self.d_k).transpose(1, 2)
        v_g = self.group_v(self.norm_group(x_group)).reshape(batch_size * num_nodes, num_groups, self.nhead, self.d_k).transpose(1, 2)

        attn_g = self.scaled_dot_attn(q_g, k_g, v_g)  # [batch_size*num_nodes, nhead, num_groups, d_k]
        attn_g = attn_g.transpose(1, 2).reshape(batch_size * num_nodes, num_groups, self.d_model)  # 拼接多头
        out_g = self.dropout(self.out_proj(attn_g))
        # 残差连接（公式5-13）：x_group + 分组维注意力输出
        x_group_out = x_group + out_g
        x_group_out = x_group_out.reshape(batch_size, num_nodes, num_groups, self.d_model)  # 恢复原维度

        # -------------------------- 2. 节点维自注意力（视图维，公式5-12）--------------------------
        # 维度调整：[batch_size*num_groups, num_nodes, d_model]（跨视图计算相同Token关联）
        x_node = x_group_out.permute(0, 2, 1, 3).reshape(batch_size * num_groups, num_nodes, self.d_model)
        q_n = self.node_q(self.norm_node(x_node)).reshape(batch_size * num_groups, num_nodes, self.nhead, self.d_k).transpose(1, 2)
        k_n = self.node_k(self.norm_node(x_node)).reshape(batch_size * num_groups, num_nodes, self.nhead, self.d_k).transpose(1, 2)
        v_n = self.node_v(self.norm_node(x_node)).reshape(batch_size * num_groups, num_nodes, self.nhead, self.d_k).transpose(1, 2)

        attn_n = self.scaled_dot_attn(q_n, k_n, v_n)  # [batch_size*num_groups, nhead, num_nodes, d_k]
        attn_n = attn_n.transpose(1, 2).reshape(batch_size * num_groups, num_nodes, self.d_model)  # 拼接多头
        out_n = self.dropout(self.out_proj(attn_n))
        # 残差连接（公式5-12）：x_node + 节点维注意力输出
        x_node_out = x_node + out_n
        x_node_out = x_node_out.reshape(batch_size, num_groups, num_nodes, self.d_model).permute(0, 2, 1, 3)  # 恢复原维度

        return x_node_out  # [batch_size, num_nodes, num_groups, d_model]

class FusionVectorCrossAttention(nn.Module):
    """对应文档融合向量交叉注意力：用融合向量捕捉全维度关联（公式5-15至5-30）"""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        # 融合向量（Q）与Token（K/V）的线性映射
        self.fusion_q = nn.Linear(d_model, d_model)  # 融合向量作为Q
        self.token_k = nn.Linear(d_model, d_model)    # Token作为K
        self.token_v = nn.Linear(d_model, d_model)    # Token作为V

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, fusion_vec: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, num_nodes, num_groups, d_model]（维度分离注意力输出）
        # fusion_vec: [batch_size, 1, d_model]（全局融合向量，初始为随机或Class Token平均）
        batch_size, num_nodes, num_groups, _ = x.shape

        # 1. 维度调整：Token展平为[batch_size, num_nodes*num_groups, d_model]
        x_flat = x.reshape(batch_size, num_nodes * num_groups, self.d_model)

        # 2. 交叉注意力计算（公式5-15、5-16）
        q_f = self.fusion_q(self.norm(fusion_vec)).reshape(batch_size, 1, self.nhead, self.d_k).transpose(1, 2)
        k_t = self.token_k(self.norm(x_flat)).reshape(batch_size, num_nodes*num_groups, self.nhead, self.d_k).transpose(1, 2)
        v_t = self.token_v(self.norm(x_flat)).reshape(batch_size, num_nodes*num_groups, self.nhead, self.d_k).transpose(1, 2)

        attn_cross = self.scaled_dot_attn(q_f, k_t, v_t)  # [batch_size, nhead, 1, d_k]
        attn_cross = attn_cross.transpose(1, 2).reshape(batch_size, 1, self.d_model)  # 拼接多头
        out_cross = self.dropout(self.out_proj(attn_cross))

        # 3. 融合向量更新（公式5-30）：残差连接确保稳定性
        fusion_vec_updated = fusion_vec + out_cross
        return fusion_vec_updated

    def scaled_dot_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weight = F.softmax(attn_score, dim=-1)
        attn_weight = self.dropout(attn_weight)
        return torch.matmul(attn_weight, v)

class MLP(nn.Module):
    """对应文档公式(5-9)、(5-14)：注意力后非线性变换"""
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 残差连接（公式5-9）：x + MLP(LN(x))
        out = self.fc2(self.dropout(self.act(self.fc1(self.norm(x)))))
        return x + out


class DynamicNodeSignalFusion(nn.Module):
    """按样本自适应地融合原始节点信号。"""
    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, num_nodes, seq_len, input_dim]
        weights = F.softmax(self.score(x), dim=1)
        return torch.sum(x * weights, dim=1, keepdim=True)


class DynamicNodeTokenFusion(nn.Module):
    """按样本和token自适应地融合节点特征。"""
    def __init__(self, d_model: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or max(d_model // 2, 16)
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, num_nodes, num_groups, d_model]
        weights = F.softmax(self.score(x), dim=1)
        return torch.sum(x * weights, dim=1, keepdim=True)


class DynamicNodeFeatureFusion(nn.Module):
    """按样本自适应地融合节点级全局特征。"""
    def __init__(self, d_model: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or max(d_model // 2, 16)
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, num_nodes, d_model]
        weights = F.softmax(self.score(x), dim=1)
        return torch.sum(x * weights, dim=1)


class ReliabilityAwareNodeFusion(nn.Module):
    """结合节点特征与质量统计量的可靠性加权融合。"""
    def __init__(self, d_model: int, stat_dim: int = 4, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or max(d_model // 2, 16)
        self.score = nn.Sequential(
            nn.LayerNorm(d_model + stat_dim),
            nn.Linear(d_model + stat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = F.softmax(self.score(torch.cat([x, stats], dim=-1)), dim=1)
        fused = torch.sum(x * weights, dim=1)
        return fused, weights.squeeze(-1)


class CrossNodeCoAttentionBlock(nn.Module):
    """在节点维上做自注意力，让不同雷达节点先交互再融合。"""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, mlp_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        hidden_dim = max(d_model * mlp_ratio, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + self.drop(attn_out)
        return x + self.ffn(self.norm2(x))


def flatten_periods_as_sequence(x: torch.Tensor) -> torch.Tensor:
    """将 period 维按时间顺序拼回完整时序，避免过早压缩长时序信息。"""
    batch_size, num_nodes, seq_len, input_dim, num_periods = x.shape
    return x.permute(0, 1, 4, 2, 3).reshape(batch_size, num_nodes, seq_len * num_periods, input_dim)


class TemporalConvStem(nn.Module):
    """轻量局部时序前端，先抽取短程纹理，再交给 Transformer 建模长依赖。"""
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IQDenoiseStem(nn.Module):
    """低SNR场景下的轻量IQ去噪残差前端。"""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 16, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=5, padding=2),
            nn.Dropout(dropout),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, input_dim, seq_len]
        return x + self.res_scale * self.net(x)


def build_complex_spectrogram(x: torch.Tensor, n_fft: int = 64, hop_length: int = 16) -> torch.Tensor:
    """对 IQ 复信号做滑窗 FFT，构造简化时频谱。"""
    complex_x = torch.complex(x[..., 0], x[..., 1])
    frames = complex_x.unfold(dimension=-1, size=n_fft, step=hop_length)
    window = torch.hann_window(n_fft, device=x.device, dtype=x[..., 0].dtype).view(1, 1, 1, -1)
    spec = torch.fft.fft(frames * window, n=n_fft, dim=-1)
    return torch.log1p(torch.abs(spec).float())


class DomainFeatureFusion(nn.Module):
    """自适应融合时域和频域的节点级特征。"""
    def __init__(self, d_model: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or max(d_model, 32)
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, time_feat: torch.Tensor, freq_feat: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(torch.cat([time_feat, freq_feat], dim=-1)))
        return gate * time_feat + (1.0 - gate) * freq_feat


class GatedInteractionFusion(nn.Module):
    """门控 + 显式时频交互项的轻量升级版。"""
    def __init__(self, d_model: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or max(d_model, 32)
        self.inter_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.weight_net = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, time_feat: torch.Tensor, freq_feat: torch.Tensor) -> torch.Tensor:
        inter_feat = self.inter_proj(time_feat * freq_feat)
        weights = F.softmax(self.weight_net(torch.cat([time_feat, freq_feat, inter_feat], dim=-1)), dim=-1)
        return (
            weights[:, 0:1] * time_feat
            + weights[:, 1:2] * freq_feat
            + weights[:, 2:3] * inter_feat
        )


class CrossGatedInteractionFusion(nn.Module):
    """较复杂的交叉门控交互融合：先互相调制，再做三分支自适应融合。"""
    def __init__(self, d_model: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or max(d_model, 32)
        self.time_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.freq_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.inter_proj = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.mix_score = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, time_feat: torch.Tensor, freq_feat: torch.Tensor) -> torch.Tensor:
        time_enh = time_feat * self.time_gate(freq_feat)
        freq_enh = freq_feat * self.freq_gate(time_feat)
        inter_feat = self.inter_proj(torch.cat([time_feat * freq_feat, torch.abs(time_feat - freq_feat)], dim=-1))
        weights = F.softmax(self.mix_score(torch.cat([time_enh, freq_enh, inter_feat], dim=-1)), dim=-1)
        return (
            weights[:, 0:1] * time_enh
            + weights[:, 1:2] * freq_enh
            + weights[:, 2:3] * inter_feat
        )


class TimeFreqCrossAttentionBlock(nn.Module):
    """时域token向频域token发起单向交叉注意力，实现池化前的深度时频对齐。"""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, mlp_ratio: int = 2):
        super().__init__()
        self.time_norm = nn.LayerNorm(d_model)
        self.freq_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(dropout)
        hidden_dim = max(d_model * mlp_ratio, d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, time_tokens: torch.Tensor, freq_tokens: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.cross_attn(
            query=self.time_norm(time_tokens),
            key=self.freq_norm(freq_tokens),
            value=self.freq_norm(freq_tokens),
            need_weights=False,
        )
        time_tokens = time_tokens + self.attn_drop(attn_out)
        return time_tokens + self.ffn(self.ffn_norm(time_tokens))


class ScaleFeatureFusion(nn.Module):
    """自适应融合不同频谱尺度的特征。"""
    def __init__(self, d_model: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or max(d_model, 32)
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.score(x), dim=1)
        return torch.sum(x * weights, dim=1)


class CosineClassifier(nn.Module):
    """Normalized cosine classifier for small-sample settings."""
    def __init__(self, feat_dim: int, num_classes: int, init_scale: float = 10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.scale = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.weight, dim=-1)
        return self.scale * torch.matmul(x, w.t())


class RadarInterferenceModel_Scheme1(nn.Module):
    """方案1：维度分离注意力 + 视图维融合（对齐文档基础框架）"""
    def __init__(self, 
                 input_dim: int = 2,  # 实部+虚部（您的方案中为2）
                 input_period: int = 8,  # 雷达周期数（您的方案中为8）
                 patch_size: int = 16,  # Token分块尺寸（您的方案中为16）
                 stride: int = 16,  # 卷积步长，若要重叠分块可调整此值
                 seq_len: int = 2000,  # 单脉冲周期采样点（您的方案中为2000）
                 num_nodes: int = 3,  # 雷达视图数（您的方案中为3）
                 d_model: int = 128,  # 特征维度D
                 mlp_ratio: int = 4,  # 隐藏层维度倍率（文档5-14）
                 nhead: int = 8,      # 注意力头数A
                 num_blocks: int = 4, # 编码模块堆叠数（文档H）
                 num_classes: int = 10,  # 干扰样式类别数
                 dropout: float = 0.1):
        super().__init__()
        self.num_groups = (seq_len - patch_size) // stride + 1  # 计算分块数量，考虑步长
        self.d_model = d_model

        # 1. 脉冲周期线性映射（您的方案：8周期→1维，此处用线性层实现）
        self.period_linear = nn.Linear(input_period, 1)  # 输入含8周期，输出1维

        # 2. 信号分块与编码（使用1D卷积进行patch embedding，通道数为input_dim）
        self.patch_embed = nn.Conv1d(in_channels=input_dim, out_channels=d_model, kernel_size=patch_size, stride=stride)

        # 3. 位置编码（文档公式5-1）
        self.pos_enc = PositionalEncoding(d_model, num_nodes, self.num_groups)

        # 4. 编码模块堆叠（文档若干编码模块，含维度分离注意力+MLP）
        self.blocks = nn.ModuleList([
            nn.Sequential(
                SeparableSelfAttention(d_model, nhead, dropout),
                MLP(d_model, d_model * mlp_ratio, dropout)  # 隐藏层维度设为d_model*mlp_ratio
            ) for _ in range(num_blocks)
        ])

        # 5. 视图维融合（您的方案：线性层/平均，此处提供两种选择）
        self.node_fusion = DynamicNodeTokenFusion(d_model)
        # self.node_fusion = lambda x: x.mean(dim=1, keepdim=True)  # 平均融合（注释切换）

        # 6. 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入x: [batch_size, num_nodes=3, seq_len=2000, input_dim=2, num_periods=8]
        # （注：您的原始样本维度(3,12000,2,8,2000)，此处调整为[batch_size, num_nodes, seq_len, input_dim, num_periods]以便计算）
        batch_size, num_nodes, seq_len, input_dim, num_periods = x.shape

        # -------------------------- 1. 脉冲周期线性映射（8→1）--------------------------
        x = self.period_linear(x.permute(0,1,2,3,4)).squeeze(-1)  # [batch_size, num_nodes, seq_len, input_dim]

        # -------------------------- 2. 调整维度以适应1D卷积 [batch_size*num_nodes, input_dim, seq_len] --------------------------
        x = x.permute(0, 1, 3, 2).reshape(batch_size * num_nodes, input_dim, seq_len)

        # -------------------------- 3. 信号分块与编码（使用1D卷积）--------------------------
        x = self.patch_embed(x)  # [batch_size*num_nodes, d_model, self.num_groups]
        x = x.permute(0, 2, 1)  # 调整为 [batch_size*num_nodes, self.num_groups, d_model]
        x = x.reshape(batch_size, num_nodes, self.num_groups, self.d_model)  # 恢复为 [batch_size, num_nodes, self.num_groups, d_model]

        # -------------------------- 4. 位置编码（文档公式5-1）--------------------------
        x = self.pos_enc(x)

        # -------------------------- 5. 编码模块堆叠（文档H个Block）--------------------------
        for block in self.blocks:
            x = block(x)  # [batch_size, num_nodes, num_groups, d_model]

        # -------------------------- 6. 视图维融合（您的方案）--------------------------
        # 按视图维融合：[batch_size, 1, num_groups, d_model]
        x = self.node_fusion(x).squeeze(1)
        # Token维平均：[batch_size, d_model]
        x = x.mean(dim=1)

        # -------------------------- 7. 分类--------------------------
        return self.classifier(x)  # [batch_size, num_classes]
    
class RadarInterferenceModel_Scheme2(nn.Module):
    """方案2：维度分离注意力 + 全局融合向量（对齐文档完整框架）"""
    def __init__(self, 
                 input_dim: int = 2,  # 实部+虚部（您的方案中为2）
                 input_period: int = 8,  # 雷达周期数（您的方案中为8）
                 patch_size: int = 16,  # Token分块尺寸（您的方案中为16）
                 stride: int = 16,  # 卷积步长，若要重叠分块可调整此值
                 seq_len: int = 2000,  # 单脉冲周期采样点（您的方案中为2000）
                 num_nodes: int = 3,  # 雷达视图数（您的方案中为3）
                 d_model: int = 128,  # 特征维度D
                 mlp_ratio: int = 2,
                 nhead: int = 8,
                 num_blocks: int = 4,
                 num_classes: int = 10,
                 dropout: float = 0.1):
        super().__init__()
        self.num_groups = (seq_len - patch_size) // stride + 1  # 计算分块数量，考虑步长
        self.d_model = d_model

        # 1. 脉冲周期线性映射（您的方案：8周期→1维，此处用线性层实现）
        self.period_linear = nn.Linear(input_period, 1)  # 输入含8周期，输出1维

        # 2. 信号分块与编码（使用1D卷积进行patch embedding，通道数为input_dim）
        self.patch_embed = nn.Conv1d(in_channels=input_dim, out_channels=d_model, kernel_size=patch_size, stride=stride)

        # 3. 位置编码（文档公式5-1）
        self.pos_enc = PositionalEncoding(d_model, num_nodes, self.num_groups)

        # 4. 编码模块堆叠（新增融合向量交叉注意力）
        self.blocks = nn.ModuleList([
            nn.Sequential(
                SeparableSelfAttention(d_model, nhead, dropout),
                MLP(d_model, d_model * mlp_ratio, dropout)  # 隐藏层维度设为d_model*4
            ) for _ in range(num_blocks)
        ])

        # 5. 全局融合向量相关（文档核心组件）
        self.fusion_vec_init = nn.Parameter(torch.randn(1, 1, d_model))  # 初始融合向量（文档r'）
        self.cross_attn = FusionVectorCrossAttention(d_model, nhead, dropout)  # 交叉注意力

        # 6. 特征融合（方案2：Token平均特征 + 融合向量）
        self.feature_fusion = nn.Linear(d_model * 2, d_model)  # 拼接后线性融合

        # 7. 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入x: [batch_size, num_nodes=3, seq_len=2000, input_dim=2, num_periods=8]
        batch_size, num_nodes, seq_len, input_dim, num_periods = x.shape

        # -------------------------- 1. 脉冲周期线性映射（8→1）--------------------------
        x = self.period_linear(x.permute(0,1,2,3,4)).squeeze(-1)  # [batch_size, num_nodes, seq_len, input_dim]

        # -------------------------- 2. 调整维度以适应1D卷积 [batch_size*num_nodes, input_dim, seq_len] --------------------------
        x = x.permute(0, 1, 3, 2).reshape(batch_size * num_nodes, input_dim, seq_len)

        # -------------------------- 3. 信号分块与编码（使用1D卷积）--------------------------
        x = self.patch_embed(x)  # [batch_size*num_nodes, d_model, self.num_groups]
        x = x.permute(0, 2, 1)  # 调整为 [batch_size*num_nodes, self.num_groups, d_model]
        x = x.reshape(batch_size, num_nodes, self.num_groups, self.d_model)  # 恢复为 [batch_size, num_nodes, self.num_groups, d_model]

        # -------------------------- 4. 位置编码（文档公式5-1）--------------------------
        x = self.pos_enc(x)

        # -------------------------- 5. 编码模块+融合向量更新（文档H个Block）--------------------------
        # 初始化融合向量（广播到batch_size）
        fusion_vec = self.fusion_vec_init.expand(batch_size, 1, self.d_model)  # [batch_size, 1, d_model]
        for block in self.blocks:
            # 维度分离注意力+MLP
            x = block(x)
            # 交叉注意力更新融合向量（文档公式5-30）
            fusion_vec = self.cross_attn(x, fusion_vec)

        # -------------------------- 6. 特征融合（方案2：Token平均 + 融合向量）--------------------------
        # Token维平均：[batch_size, num_nodes, d_model]
        token_avg = x.mean(dim=2)
        # 视图维平均：[batch_size, d_model]
        token_avg = token_avg.mean(dim=1)
        # 融合向量展平：[batch_size, d_model]
        fusion_vec_flat = fusion_vec.squeeze(1)
        # 拼接融合：[batch_size, d_model*2] → [batch_size, d_model]
        x_fused = self.feature_fusion(torch.cat([token_avg, fusion_vec_flat], dim=1))

        # -------------------------- 7. 分类--------------------------
        return self.classifier(x_fused)  # [batch_size, num_classes]

class RadarInterferenceModel_Scheme3(nn.Module):
    """方案3：3节点预处理阶段融合→1DCNN分块→基础Transformer"""
    def __init__(self, fusion_type="weighted", input_dim=2, input_period=8, patch_size=16, stride=16, seq_len=2000, num_nodes=3,
                 d_model=128, mlp_ratio=4, nhead=8, num_blocks=4, num_classes=12, dropout=0.1):
        super().__init__()
        self.fusion_type = fusion_type  # "avg"（平均）或 "weighted"（可学习加权）
        self.num_groups = (seq_len - patch_size) // stride + 1  # Token数
        self.d_model = d_model

        # 1. 脉冲周期线性映射（8周期→1维，对应您的预处理逻辑）
        self.period_linear = nn.Linear(input_period, 1)

        # 2. 节点融合（基础级：平均/可学习加权）
        if self.fusion_type == "weighted":
            self.node_fusion = DynamicNodeSignalFusion(input_dim)  # 3节点→1节点（动态权重）
        # 平均融合无需参数，forward中直接计算均值

        # 3. 1DCNN分块（对应文档“分组展平”，替换线性层）
        self.patch_embed = nn.Conv1d(in_channels=input_dim, out_channels=d_model, 
                                     kernel_size=patch_size, stride=stride)

        # 4. 位置编码（文档公式5-1，融合后num_nodes=1）
        self.pos_enc = PositionalEncoding(d_model, num_nodes=1, num_groups=self.num_groups)

        # 5. 基础Transformer Block（文档若干编码模块）
        self.blocks = nn.ModuleList([
            Block(d_model, nhead, mlp_ratio, proj_drop=dropout, attn_drop=dropout, drop_path=dropout, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(num_blocks)])

        # 6. 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # 输入x: (N, 3, 2000, 2, 8) → (N, 3节点, 2000采样点, 2实虚部, 8周期)
        N, _, seq_len, input_dim, _ = x.shape

        # 1. 脉冲周期线性映射（8→1）：(N,3,2000,2,8)→(N,3,2000,2)
        x = self.period_linear(x.permute(0,1,2,3,4)).squeeze(-1)

        # 2. 节点融合（基础级）：(N,3,2000,2)→(N,1,2000,2)
        if self.fusion_type == "weighted":
            # 维度调整：(N,3,2000,2)→(N,2000,2,3)→线性融合→(N,2000,2,1)→(N,1,2000,2)
            x = self.node_fusion(x)
        else:  # 平均融合
            x = x.mean(dim=1, keepdim=True)  # (N,3,2000,2)→(N,1,2000,2)

        # 3. 1DCNN分块：(N,1,2000,2)→(N*1,2,2000)→(N*1,d_model,num_groups)
        x = x.reshape(N*1, input_dim, seq_len)  # 适配1DCNN输入：(batch, channel, seq_len)
        x = self.patch_embed(x)  # (N, d_model, num_groups)
        x = x.permute(0, 2, 1)  # (N, num_groups, d_model) → Transformer输入格式

        # 4. 位置编码：(N, num_groups, d_model)→(N, num_groups, d_model)
        x = self.pos_enc(x.unsqueeze(1)).squeeze(1)  # pos_enc需(num_nodes=1)维度，临时扩展

        # 5. Transformer特征提取
        for blk in self.blocks:
            x = blk(x)

        # 6. 全局平均池化+分类
        x = x.mean(dim=1)  # (N, d_model)
        return self.classifier(x)

# -------------------------- 方案2：共享Transformer提取特征→可学习加权融合 --------------------------
class RadarInterferenceModel_Scheme4(nn.Module):
    """方案4：3节点共享Transformer→特征加权融合"""
    def __init__(self, 
                input_dim: int = 2,  # 实部+虚部
                 input_period: int = 8,  # 雷达周期数
                 patch_size: int = 16,  # Token分块尺寸
                 stride: int = 16,  # 卷积步长
                 seq_len: int = 2000,  # 单脉冲周期采样点
                 num_nodes: int = 3,  # 雷达视图数
                 d_model: int = 128,  # 特征维度D
                 mlp_ratio: int = 4,  # 隐藏层维度倍率
                 nhead: int = 8,      # 注意力头数A
                 num_blocks: int = 4, # 编码模块堆叠数
                 num_classes: int = 10,  # 干扰样式类别数
                 dropout: float = 0.1):
        super().__init__()
        self.num_groups = (seq_len - patch_size) // stride + 1
        self.d_model = d_model
        self.num_nodes = num_nodes

        # 1. 脉冲周期线性映射
        self.period_linear = nn.Linear(input_period, 1)

        # 2. 1DCNN分块（3节点共享）
        self.patch_embed = nn.Conv1d(in_channels=input_dim, out_channels=d_model, 
                                     kernel_size=patch_size, stride=stride)

        # 3. 位置编码（文档公式5-1，num_nodes=3）
        self.pos_enc = PositionalEncoding(d_model, num_nodes=num_nodes, num_groups=self.num_groups)

        # 4. 共享Transformer Block（3节点共享编码模块，文档第12段）
        self.shared_blocks = nn.ModuleList([
            Block(d_model, nhead, mlp_ratio, proj_drop=dropout, attn_drop=dropout, drop_path=dropout, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(num_blocks)])

        # 5. 节点特征加权融合（可学习权重，文档第31段“融合向量”思路）
        self.node_fusion = DynamicNodeTokenFusion(d_model)

        # 6. 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # 输入x: (N, 3, 2000, 2, 8)
        N, num_nodes, seq_len, input_dim, _ = x.shape

        # 1. 脉冲周期映射：(N,3,2000,2,8)→(N,3,2000,2)
        x = self.period_linear(x.permute(0,1,2,3,4)).squeeze(-1)

        # 2. 1DCNN分块：(N,3,2000,2)→(N*3,2,2000)→(N*3,d_model,num_groups)
        x = x.permute(0,1,3,2).reshape(N*num_nodes, input_dim, seq_len)
        x = self.patch_embed(x)  # (N*3, d_model, num_groups)
        x = x.permute(0, 2, 1)  # (N*3, num_groups, d_model)

        # 3. 位置编码：(N*3, num_groups, d_model)→(N,3,num_groups,d_model)
        x = x.reshape(N, num_nodes, self.num_groups, self.d_model)
        x = self.pos_enc(x)

        # 4. 共享Transformer提取特征
        x = x.reshape(N*num_nodes, self.num_groups, self.d_model)  # 共享参数需展平节点维
        
        for blk in self.shared_blocks:
            x = blk(x)
        
        x = x.reshape(N, num_nodes, self.num_groups, self.d_model)  # 恢复节点维

        # 5. 节点特征融合：(N,3,num_groups,d_model)→(N,1,num_groups,d_model)→(N,num_groups,d_model)
        x = self.node_fusion(x).squeeze(1)

        # 6. 分类
        x = x.mean(dim=1)  # (N, d_model)
        return self.classifier(x)

# -------------------------- 方案5：独立Transformer提取特征→加权融合 --------------------------
class RadarInterferenceModel_Scheme5(nn.Module):
    """方案5创新版：完整时序 + 时频双分支 + 类中心友好的判别特征头。"""
    def __init__(self, 
                input_dim: int = 2,
                 input_period: int = 8,
                 patch_size: int = 16,
                 stride: int = 16,
                 seq_len: int = 2000,
                 num_nodes: int = 3,
                 d_model: int = 128,
                 mlp_ratio: int = 4,
                 nhead: int = 8,
                 num_blocks: int = 4,
                 num_classes: int = 10,
                 dropout: float = 0.1,
                 use_denoise_stem: bool = False,
                 use_tf_cross_attn: bool = False,
                 use_node_co_attn: bool = False,
                 use_node_reliability: bool = False,
                 domain_fusion_type: str = "gate",
                 cls_head: str = "linear",
                 cosine_scale: float = 10.0):

        super().__init__()
        self.effective_seq_len = seq_len * input_period
        self.num_groups = (self.effective_seq_len - patch_size) // stride + 1
        self.spec_n_fft = min(max(32, patch_size * 8), self.effective_seq_len)
        self.spec_hop = max(self.spec_n_fft // 4, 4)
        self.spec_groups = (self.effective_seq_len - self.spec_n_fft) // self.spec_hop + 1
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.use_denoise_stem = use_denoise_stem
        self.use_tf_cross_attn = use_tf_cross_attn
        self.use_node_co_attn = use_node_co_attn
        self.use_node_reliability = use_node_reliability
        self.domain_fusion_type = domain_fusion_type
        self.cls_head = cls_head
        stem_dim = max(d_model // 2, 32)

        if self.use_denoise_stem:
            self.denoise_stems = nn.ModuleList([
                IQDenoiseStem(input_dim=input_dim, hidden_dim=max(16, d_model // 8), dropout=min(0.1, dropout))
                for _ in range(num_nodes)
            ])

        # 时域分支
        self.node_stems = nn.ModuleList([
            TemporalConvStem(input_dim=input_dim, hidden_dim=stem_dim, dropout=dropout)
            for _ in range(num_nodes)
        ])
        self.patch_embeds = nn.ModuleList([
            nn.Conv1d(in_channels=stem_dim, out_channels=d_model,
                      kernel_size=patch_size, stride=stride)
            for _ in range(num_nodes)
        ])
        self.pos_encs = nn.ModuleList([
            PositionalEncoding(d_model, num_nodes=1, num_groups=self.num_groups)
            for _ in range(num_nodes)
        ])
        self.node_blocks = nn.ModuleList([
            nn.Sequential(*[
                Block(d_model, nhead, mlp_ratio, proj_drop=dropout, attn_drop=dropout, drop_path=dropout, qkv_bias=True, norm_layer=nn.LayerNorm)
                for _ in range(num_blocks)
            ]) for _ in range(num_nodes)
        ])

        # 频域分支
        self.spec_embeds = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.spec_n_fft),
                nn.Linear(self.spec_n_fft, d_model),
                nn.GELU(),
            ) for _ in range(num_nodes)
        ])
        self.spec_pos_encs = nn.ModuleList([
            PositionalEncoding(d_model, num_nodes=1, num_groups=self.spec_groups)
            for _ in range(num_nodes)
        ])
        self.spec_blocks = nn.ModuleList([
            nn.Sequential(*[
                Block(d_model, nhead, mlp_ratio, proj_drop=dropout, attn_drop=dropout, drop_path=dropout, qkv_bias=True, norm_layer=nn.LayerNorm)
                for _ in range(max(1, num_blocks - 1))
            ]) for _ in range(num_nodes)
        ])
        if self.use_tf_cross_attn:
            self.time_freq_cross_blocks = nn.ModuleList([
                TimeFreqCrossAttentionBlock(d_model, nhead, dropout=dropout, mlp_ratio=mlp_ratio)
                for _ in range(num_nodes)
            ])

        # 节点内时频融合 + 节点间融合
        if domain_fusion_type == "gated_interaction":
            fusion_factory = lambda: GatedInteractionFusion(d_model)
        elif domain_fusion_type == "cross_gated_interaction":
            fusion_factory = lambda: CrossGatedInteractionFusion(d_model)
        else:
            fusion_factory = lambda: DomainFeatureFusion(d_model)
        self.domain_fusions = nn.ModuleList([fusion_factory() for _ in range(num_nodes)])
        if self.use_node_co_attn:
            self.node_co_attn = CrossNodeCoAttentionBlock(d_model, nhead, dropout=dropout, mlp_ratio=mlp_ratio)
        if self.use_node_reliability:
            self.node_fusion = ReliabilityAwareNodeFusion(d_model, stat_dim=4)
        else:
            self.node_fusion = DynamicNodeFeatureFusion(d_model)

        # 判别特征头：供中心损失约束的 embedding
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if cls_head == "cosine":
            self.classifier = CosineClassifier(d_model, num_classes, init_scale=cosine_scale)
        else:
            self.classifier = nn.Linear(d_model, num_classes)

    def _extract_node_features(self, x):
        N, num_nodes, _, _, _ = x.shape
        node_features = []
        node_stats = []

        x = flatten_periods_as_sequence(x)
        if self.use_denoise_stem:
            denoised_nodes = []
            for i in range(num_nodes):
                node_x = self.denoise_stems[i](x[:, i].permute(0, 2, 1)).permute(0, 2, 1)
                denoised_nodes.append(node_x.unsqueeze(1))
            x = torch.cat(denoised_nodes, dim=1)
        spec = build_complex_spectrogram(x, n_fft=self.spec_n_fft, hop_length=self.spec_hop)

        for i in range(num_nodes):
            time_x = x[:, i].permute(0, 2, 1)
            time_x = self.node_stems[i](time_x)
            time_x = self.patch_embeds[i](time_x).permute(0, 2, 1)
            time_x = self.pos_encs[i](time_x.unsqueeze(1)).squeeze(1)
            time_x = self.node_blocks[i](time_x)

            freq_x = self.spec_embeds[i](spec[:, i])
            freq_x = self.spec_pos_encs[i](freq_x.unsqueeze(1)).squeeze(1)
            freq_x = self.spec_blocks[i](freq_x)
            if self.use_tf_cross_attn:
                time_x = self.time_freq_cross_blocks[i](time_x, freq_x)

            time_feat = time_x.mean(dim=1)
            freq_feat = freq_x.mean(dim=1)

            fused_feat = self.domain_fusions[i](time_feat, freq_feat)
            node_features.append(fused_feat.unsqueeze(1))
            node_signal = x[:, i]
            complex_sig = torch.complex(node_signal[..., 0], node_signal[..., 1])
            amp = torch.abs(complex_sig)
            power_mean = torch.mean(amp ** 2, dim=1)
            amp_std = torch.std(amp, dim=1)
            spec_i = spec[:, i]
            spec_mean = torch.mean(spec_i, dim=(1, 2)) + 1e-6
            spec_peak_ratio = torch.amax(spec_i, dim=(1, 2)) / spec_mean
            spec_prob = spec_i / torch.sum(spec_i, dim=(1, 2), keepdim=True).clamp_min(1e-6)
            spec_entropy = -torch.sum(spec_prob * torch.log(spec_prob.clamp_min(1e-6)), dim=(1, 2))
            spec_entropy = spec_entropy / math.log(float(spec_i.shape[1] * spec_i.shape[2]))
            node_stats.append(torch.stack([power_mean, amp_std, spec_peak_ratio, 1.0 - spec_entropy], dim=1).unsqueeze(1))

        node_features = torch.cat(node_features, dim=1)
        node_stats = torch.cat(node_stats, dim=1)
        return node_features, node_stats

    def forward_features(self, x):
        node_features, node_stats = self._extract_node_features(x)
        fused_x = node_features
        if self.use_node_co_attn:
            fused_x = self.node_co_attn(fused_x)
        if self.use_node_reliability:
            fused_x, _ = self.node_fusion(fused_x, node_stats)
        else:
            fused_x = self.node_fusion(fused_x)
        return self.embedding_head(fused_x)

    def forward_with_features(self, x):
        features = self.forward_features(x)
        return self.classifier(features), features

    def forward_with_details(self, x):
        node_features, node_stats = self._extract_node_features(x)
        if self.use_node_co_attn:
            node_features = self.node_co_attn(node_features)
        if self.use_node_reliability:
            fused_global, node_weights = self.node_fusion(node_features, node_stats)
        else:
            fused_global = self.node_fusion(node_features)
            node_weights = None
        features = self.embedding_head(fused_global)
        return self.classifier(features), features, node_features, node_weights

    def forward(self, x):
        return self.classifier(self.forward_features(x))

# -------------------------- 方案4：独立Transformer+分类器→投票融合 --------------------------
class NodeClassifier(nn.Module):
    """方案6：单个节点的“Transformer+分类器”（用于独立训练）"""
    def __init__(self, 
                input_dim: int = 2,  # 实部+虚部
                 input_period: int = 8,  # 雷达周期数
                 patch_size: int = 16,  # Token分块尺寸
                 stride: int = 16,  # 卷积步长
                 seq_len: int = 2000,  # 单脉冲周期采样点
                 d_model: int = 128,  # 特征维度D
                 mlp_ratio: int = 4,  # 隐藏层维度倍率
                 nhead: int = 8,      # 注意力头数A
                 num_blocks: int = 4, # 编码模块堆叠数
                 num_classes: int = 10,  # 干扰样式类别数
                 dropout: float = 0.1):
        super().__init__()
        self.num_groups = (seq_len - patch_size) // stride + 1
        self.d_model = d_model

        # 1. 脉冲周期线性映射
        self.period_linear = nn.Linear(input_period, 1)

        # 2. 1DCNN分块
        self.patch_embed = nn.Conv1d(in_channels=input_dim, out_channels=d_model, 
                                     kernel_size=patch_size, stride=stride)

        # 3. 位置编码（num_nodes=1）
        self.pos_enc = PositionalEncoding(d_model, num_nodes=1, num_groups=self.num_groups)

        # 4. Transformer Block
        self.blocks = nn.ModuleList([
            Block(d_model, nhead, mlp_ratio, proj_drop=dropout, attn_drop=dropout, drop_path=dropout, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(num_blocks)])

        # 5. 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # 输入x: (N, 1, 2000, 2, 8)（单个节点数据）
        N, _, seq_len, input_dim, _ = x.shape

        # 1. 脉冲周期映射：(N,1,2000,2,8)→(N,1,2000,2)
        x = self.period_linear(x.permute(0,1,2,3,4)).squeeze(-1)

        # 2. 1DCNN分块：(N,1,2000,2)→(N,2,2000)→(N,d_model,num_groups)
        x = x.reshape(N, input_dim, seq_len)
        x = self.patch_embed(x).permute(0,2,1)  # (N,num_groups,d_model)

        # 3. 位置编码
        x = self.pos_enc(x.unsqueeze(1)).squeeze(1)

        # 4. Transformer提取特征
        for blk in self.blocks:
            x = blk(x)

        # 5. 分类
        x = x.mean(dim=1)  # (N,d_model)
        return self.classifier(x)

class RadarInterferenceModel_Scheme6(nn.Module):
    """方案6：3个独立NodeClassifier→投票融合（仅测试阶段用，训练需单独训练每个NodeClassifier）"""
    def __init__(self, node_models):
        super().__init__()
         
        self.node_models = self.node_models = nn.ModuleList(node_models) # 3个预训练好的NodeClassifier

    def forward(self, x, mode="vote"):
        # 输入x: (N, 3, 2000, 2, 8)→拆分3个节点数据
        N = x.shape[0]
        node_logits = []

        # 每个节点模型输出logits
        for i, model in enumerate(self.node_models):
            node_x = x[:, i:i+1, :, :, :]  # (N,1,2000,2,8)
            logits = model(node_x)
            node_logits.append(logits.unsqueeze(1))  # (N,1,num_classes)

        node_logits = torch.cat(node_logits, dim=1)  # (N,3,num_classes)

        # 决策融合：投票或概率平均
        if mode == "vote":
            # 投票：取每个样本3个模型预测的多数类
            preds = torch.argmax(node_logits, dim=2)  # (N,3)
            final_preds = torch.mode(preds, dim=1)[0]  # (N,)
            return final_preds  # 投票结果（用于计算精度）
        else:
            # 直接输出三个节点的概率（用于损失计算，训练单个模型时用）
            return node_logits


if __name__ == "__main__":
    # 测试模型前向传播
    batch_size = 256
    num_nodes = 3
    input_dim = 2
    num_periods = 8
    seq_len = 2000
    x = torch.randn(batch_size, num_nodes, seq_len, input_dim, num_periods)  # 维度调整为[batch_size, num_nodes, seq_len, input_dim, num_periods]

    # 2. 初始化方案1模型并推理
    model1 = RadarInterferenceModel_Scheme5(num_classes=6)  # 假设6种干扰样式
    output1 = model1(x)
    print("方案1输出形状:", output1.shape)  # [12000, 6]

    # 3. 初始化方案2模型并推理
    model2 = RadarInterferenceModel_Scheme6(num_classes=6)
    output2 = model2(x)  # 输出3个节点的概率
    print("方案2输出形状:", output2.shape)  # [12000, 6]
