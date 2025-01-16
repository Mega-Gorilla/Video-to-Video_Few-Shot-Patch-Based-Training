import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

class DiscriminatorN_IN(nn.Module):
    """
    PatchGANベースの判別器の実装
    入力画像を小さなパッチに分割して、各パッチが本物か偽物かを判定する
    Instance Normalizationを使用して学習を安定化
    """
    def __init__(
        self,
        input_channels: int = 3,          # 入力チャネル数（RGB画像なら3）
        num_filters: int = 64,            # 基本的なフィルター数
        n_layers: int = 3,                # 畳み込み層の数
        use_noise: bool = False,          # ノイズ付加の有無
        noise_sigma: float = 0.2,         # ノイズの強度
        norm_layer: str = 'instance_norm', # 正規化層のタイプ
        use_bias: bool = True             # バイアス項の使用有無
    ):
        super().__init__()
        
        self.use_noise = use_noise
        self.noise_sigma = noise_sigma
        
        # 正規化層の選択
        # batch_norm: ミニバッチ単位での正規化
        # instance_norm: 個々の特徴マップ単位での正規化
        if norm_layer == 'batch_norm':
            norm = nn.BatchNorm2d
        elif norm_layer == 'instance_norm':
            norm = nn.InstanceNorm2d
        else:
            norm = None
            
        # 最初の畳み込みブロック
        # 正規化なしで、LeakyReLUを活性化関数として使用
        self.initial = self._make_block(
            input_channels,   # 入力チャネル（RGB）
            num_filters,      # 出力チャネル
            4, 2, 1,         # カーネルサイズ、ストライド、パディング
            use_bias,        # バイアスの使用
            None,            # 最初の層は正規化なし
            nn.LeakyReLU(0.2, True)  # 負の値に対して小さな勾配を持つ活性化関数
        )
        
        # 中間層の構築
        # 層を重ねるごとにフィルター数を2倍に増やす（最大でnum_filters*8まで）
        self.intermediate = nn.ModuleList()
        curr_filters = num_filters
        for i in range(1, n_layers):
            next_filters = min(curr_filters * 2, num_filters * 8)
            self.intermediate.append(
                self._make_block(
                    curr_filters,    # 現在のフィルター数
                    next_filters,    # 次のフィルター数（2倍）
                    4, 2, 1,        # カーネルサイズ、ストライド、パディング
                    use_bias,
                    norm,           # 正規化層を使用
                    nn.LeakyReLU(0.2, True)
                )
            )
            curr_filters = next_filters
            
        # 出力前の畳み込み層
        # ストライド1で空間サイズを維持
        next_filters = min(curr_filters * 2, num_filters * 8)
        self.pre_output = self._make_block(
            curr_filters,
            next_filters,
            4, 1, 1,      # ストライド1でサイズ維持
            use_bias,
            norm,
            nn.LeakyReLU(0.2, True)
        )
        
        # 最終出力層
        # 1チャネルの判別マップを生成
        self.output = self._make_block(
            next_filters,
            1,            # 出力は1チャネル（判別マップ）
            4, 1, 1,
            use_bias,
            None,         # 最終層は正規化なし
            None          # 活性化関数なし
        )
        
        # 重みの初期化
        self.apply(self._init_weights)
        
    def _init_weights(self, m: nn.Module):
        """ネットワークの重みを正規分布で初期化"""
        if isinstance(m, nn.Conv2d):
            # 平均0、標準偏差0.02の正規分布で初期化
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
                
    def _make_block(
        self,
        in_channels: int,      # 入力チャネル数
        out_channels: int,     # 出力チャネル数
        kernel_size: int,      # カーネルサイズ
        stride: int,           # ストライド
        padding: int,          # パディング
        bias: bool,            # バイアスの使用
        norm: Optional[nn.Module],    # 正規化層
        activation: Optional[nn.Module]  # 活性化関数
    ) -> nn.Sequential:
        """畳み込みブロックを作成する補助関数"""
        layers = [
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size, stride, padding, bias=bias
            )
        ]
        
        # 正規化層と活性化関数を追加（指定されている場合）
        if norm:
            layers.append(norm(out_channels))
        if activation:
            layers.append(activation)
            
        return nn.Sequential(*layers)
        
    def forward(self, x: Tensor) -> Tuple[Tensor, None]:
        """
        順伝播処理
        学習時にはオプションでノイズを付加
        returns:
            - 判別マップ: 各パッチが本物である確率
            - None: 互換性のため
        """
        # 学習時のノイズ付加（オプション）
        if self.use_noise and self.training:
            noise = torch.randn_like(x) * self.noise_sigma
            x = x + noise
            
        # 各層を順番に通す
        out = self.initial(x)
        for layer in self.intermediate:
            out = layer(out)
        out = self.pre_output(out)
        out = self.output(out)
        
        return out, None  # 2つ目の戻り値はコードの互換性のため