# models/generator.py
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional

class UpsamplingLayer(nn.Module):
    """アップサンプリングレイヤー
    - 画像サイズを2倍に拡大する
    """
    def __init__(self, channels: int):
        super().__init__()
        self.layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)

class ResNetBlock(nn.Module):
    """ResNetブロック
    - 2つの畳み込み層と正規化層を含む残差ブロック
    - スキップコネクションで入力をブロックの出力に加算
    """
    def __init__(
        self,
        channels: int,
        norm_layer: Optional[str] = 'instance_norm',
        use_bias: bool = False
    ):
        super().__init__()
        
        # 正規化層の選択（BatchNormまたはInstanceNorm）
        norm = None
        if norm_layer == 'batch_norm':
            norm = nn.BatchNorm2d
        elif norm_layer == 'instance_norm':
            norm = nn.InstanceNorm2d
            
        # 第1畳み込みブロック: ReLU -> Conv -> Norm
        layers = [
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=use_bias),
        ]
        if norm:
            layers.append(norm(channels))
            
        # 第2畳み込みブロック: ReLU -> Conv -> Norm
        layers.extend([
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=use_bias)
        ])
        if norm:
            layers.append(norm(channels))
            
        self.block = nn.Sequential(*layers)
        
    def forward(self, x: Tensor) -> Tensor:
        # スキップコネクション: 入力 + ブロックの出力
        return x + self.block(x)

class GeneratorJ(nn.Module):
    """スタイル変換用のGenerator
    - U-Net的な構造を持つ
    - ダウンサンプリング -> ResNetブロック -> アップサンプリング
    - スキップコネクションで詳細情報を保持
    """
    def __init__(
        self,
        input_channels: int = 3,          # 入力チャネル数（RGB=3）
        filters: List[int] = [32, 64, 128, 128, 128, 64],  # 各層のフィルター数
        norm_layer: str = 'instance_norm', # 正規化層の種類
        use_bias: bool = False,           # バイアス項の使用
        resnet_blocks: int = 7,           # ResNetブロックの数
        tanh: bool = True,                # 出力層にtanhを使用するか
        append_smoothers: bool = True,     # スムージング層を追加するか
        input_size: int = 256,            # 入力画像サイズ
    ):
        super().__init__()
        
        self.input_size = input_size
        self.append_smoothers = append_smoothers
        
        # 正規化層の選択
        norm = None
        if norm_layer == 'batch_norm':
            norm = nn.BatchNorm2d
        elif norm_layer == 'instance_norm':
            norm = nn.InstanceNorm2d
            
        # 初期畳み込み層 (7x7カーネル)
        self.initial_conv = self._make_conv_block(
            input_channels, filters[0], 7, 1, 3,
            use_bias, norm, nn.LeakyReLU(0.2, inplace=False)
        )
        
        # ダウンサンプリング層 (解像度を1/4に)
        self.downsample1 = self._make_conv_block(
            filters[0], filters[1], 3, 2, 1,
            use_bias, norm, nn.LeakyReLU(0.2, inplace=False)
        )
        self.downsample2 = self._make_conv_block(
            filters[1], filters[2], 3, 2, 1,
            use_bias, norm, nn.LeakyReLU(0.2, inplace=False)
        )
        
        # ResNetブロック (特徴抽出・変換)
        self.resnet_blocks = nn.ModuleList([
            ResNetBlock(filters[2], norm_layer, use_bias)
            for _ in range(resnet_blocks)
        ])
        
        # アップサンプリング層 (解像度を4倍に)
        # スキップコネクションのため、入力チャネル数は2倍
        self.upsample2 = self._make_upconv_block(
            filters[2] + filters[2], filters[4], 4, 2, 1,
            use_bias, norm, nn.ReLU(inplace=False)
        )
        self.upsample1 = self._make_upconv_block(
            filters[4] + filters[1], filters[4], 4, 2, 1,
            use_bias, norm, nn.ReLU(inplace=False)
        )
        
        # 最終畳み込み層（スキップコネクションのため入力チャネル数が多い）
        conv11_channels = filters[0] + filters[4] + input_channels
        self.conv11 = nn.Sequential(
            nn.Conv2d(conv11_channels, filters[5], 7, 1, 3, bias=use_bias),
            nn.ReLU(inplace=False)
        )
        
        # オプショナルなスムージング層
        if self.append_smoothers:
            self.smoothers = nn.Sequential(
                nn.Conv2d(filters[5], filters[5], 3, padding=1, bias=use_bias),
                nn.ReLU(inplace=False),
                nn.BatchNorm2d(filters[5]),
                nn.Conv2d(filters[5], filters[5], 3, padding=1, bias=use_bias),
                nn.ReLU(inplace=False)
            )
            
        # 出力層（RGB画像を生成）
        output_layers = [nn.Conv2d(filters[5], 3, 1, bias=True)]
        if tanh:
            output_layers.append(nn.Tanh())  # [-1, 1]の範囲に正規化
        self.output = nn.Sequential(*output_layers)
        
        # 重みの初期化
        self.apply(self._init_weights)
        
    def _init_weights(self, m: nn.Module):
        """ネットワークの重みを初期化"""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
                
    def _make_conv_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias: bool,
        norm: Optional[nn.Module],
        activation: Optional[nn.Module]
    ) -> nn.Sequential:
        """畳み込みブロックを作成
        Conv -> Norm -> Activation の順
        """
        layers = [
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size, stride, padding, bias=bias
            )
        ]
        
        if norm:
            layers.append(norm(out_channels))
        if activation:
            layers.append(activation)
            
        return nn.Sequential(*layers)
        
    def _make_upconv_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias: bool,
        norm: Optional[nn.Module],
        activation: Optional[nn.Module]
    ) -> nn.Sequential:
        """アップサンプリングブロックを作成
        Upsample -> Conv -> Norm -> Activation の順
        """
        layers = [
            UpsamplingLayer(in_channels),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
        ]
        
        if norm:
            layers.append(norm(out_channels))
        if activation:
            layers.append(activation)
            
        return nn.Sequential(*layers)
        
    def forward(self, x: Tensor) -> Tensor:
        """順伝播処理
        1. 特徴抽出（ダウンサンプリング）
        2. 特徴変換（ResNetブロック）
        3. 画像生成（アップサンプリング）
        """
        # 初期特徴抽出
        conv0 = self.initial_conv(x)
        
        # ダウンサンプリング（特徴抽出）
        conv1 = self.downsample1(conv0)
        conv2 = self.downsample2(conv1)
        
        # ResNetブロックによる特徴変換
        out = conv2
        for block in self.resnet_blocks:
            out = block(out)
            
        # アップサンプリング（画像生成）
        # スキップコネクションで詳細情報を結合
        out = self.upsample2(torch.cat([out, conv2], dim=1))
        out = self.upsample1(torch.cat([out, conv1], dim=1))
        out = self.conv11(torch.cat([out, conv0, x], dim=1))
        
        # オプショナルなスムージング
        if self.append_smoothers:
            out = self.smoothers(out)
            
        # 最終出力（RGBイメージ）
        out = self.output(out)
        
        return out