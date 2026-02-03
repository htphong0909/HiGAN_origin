import torch
from torch import Tensor
import torch.nn as nn
import torchvision.models as models
# Các module custom (Transformer, Loss, ResNet đặc biệt)
from models.transformer import *
from einops import rearrange, repeat  # Thư viện giúp thao tác shape tensor dễ dàng
from models.loss import Proxy_Anchor
from models.resnet_dilation import resnet18 as resnet18_dilation
import torch.fft as fft

class Mix_TR(nn.Module):
    """
    Mix_TR: Model kết hợp nội dung văn bản in (Printed Content) và phong cách viết tay (Handwriting Style).
    Kiến trúc gồm:
    1. Style Encoder: Trích xuất đặc trưng ảnh phong cách (ResNet + Transformer Encoder).
    2. Content Encoder: Mã hóa ảnh nội dung (CNN).
    3. Style Disentanglement: Tách style thành 2 hướng Vertical (dọc) và Horizontal (ngang) dùng Proxy Loss.
    4. Decoder: Trộn Content và Style thông qua cơ chế Attention.
    """
    def __init__(self, nb_classes, d_model=256, nhead=8, num_encoder_layers=2, num_head_layers=1, num_decoder_layers=3,
                 dim_feedforward=2048, dropout=0.1, activation="relu", return_intermediate_dec=False,
                 normalize_before=True, fft_threshold=8):
        super(Mix_TR, self).__init__()
        
        # --- Cấu hình Encoder cơ sở (Base Encoder) ---
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        # Encoder chung để xử lý đặc trưng style ban đầu
        self.base_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, None)

        # --- Hai nhánh Encoder riêng biệt cho Style ---
        # 1. Vertical Head: Học các đặc trưng theo chiều dọc (độ cao chữ, độ nghiêng...)
        vertical_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.vertical_head = TransformerEncoder(encoder_layer, num_head_layers, vertical_norm)

        # 2. Horizontal Head: Học các đặc trưng theo chiều ngang (khoảng cách ký tự, nét nối...)
        horizontal_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.horizontal_head = TransformerEncoder(encoder_layer, num_head_layers, horizontal_norm)

        # --- Cấu hình Decoder (Nơi trộn Content + Style) ---
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        
        # Decoder 1: Trộn style dọc
        vertical_decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.vertical_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, vertical_decoder_norm,
                                                   return_intermediate=return_intermediate_dec)
        
        # Decoder 2: Trộn style ngang
        horizontal_decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.horizontal_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, horizontal_decoder_norm,
                                                     return_intermediate=return_intermediate_dec)
        
        # --- Positional Encoding ---
        # Mã hóa vị trí 1D cho chuỗi text (Content)
        self.add_position1D = PositionalEncoding(dropout=0.1, dim=d_model) 
        # Mã hóa vị trí 2D cho ảnh style (Style)
        self.add_position2D = PositionalEncoding2D(dropout=0.1, d_model=d_model) 

        # --- MLP cho Proxy Loss (Projection Heads) ---
        # Dùng để chiếu feature về không gian metric learning
        self.vertical_pro_mlp = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512))
        self.horizontal_pro_mlp = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512))
        
        self._reset_parameters()

        # --- Style Encoder Backbone (ResNet18) ---
        # Load pre-trained ResNet18
        self.Feat_Encoder = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        # Sửa lớp đầu vào để nhận ảnh xám (1 channel) thay vì RGB (3 channels)
        self.Feat_Encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Loại bỏ các lớp cuối (FC, Pooling) để lấy feature map
        self.Feat_Encoder.layer4 = nn.Identity()
        self.Feat_Encoder.fc = nn.Identity()
        self.Feat_Encoder.avgpool = nn.Identity()
        # Thêm lớp Convolution có dilation (giãn nở) để tăng vùng nhìn (receptive field)
        self.style_dilation_layer = resnet18_dilation().conv5_x
        
        # --- Content Encoder ---
        # Một mạng CNN nhỏ dựa trên các lớp đầu của ResNet18 để mã hóa ảnh nội dung
        self.content_encoder = nn.Sequential(*([nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] +list(models.resnet18(weights='ResNet18_Weights.DEFAULT').children())[1:-2]))

        # --- Loss Functions (Metric Learning) ---
        # Proxy Anchor Loss giúp gom nhóm các style của cùng 1 tác giả lại gần nhau
        self.vertical_proxy = Proxy_Anchor(nb_classes=nb_classes, sz_embed=d_model, proxy_mode='only_real')
        self.horizontal_proxy = Proxy_Anchor(nb_classes=nb_classes, sz_embed=d_model, proxy_mode='only_real')

        self.fft_threshold = fft_threshold

    def _reset_parameters(self):
        # Khởi tạo trọng số Xavier cho các tham số (giúp hội tụ tốt hơn)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def random_double_sampling(self, x, ratio=0.25):
        """
        Hàm lấy mẫu ngẫu nhiên (Shuffle) token trong chuỗi.
        Mục đích: Tạo cặp positive (anchor, positive) cho contrastive learning.
        x: [Length, Batch, N_group, Dim]
        """
        L, B, N, D = x.shape 
        x = rearrange(x, "L B N D -> B N L D")
        noise = torch.rand(B, N, L, device=x.device)  # Tạo noise [0, 1]
        ids_shuffle = torch.argsort(noise, dim=2)     # Sắp xếp noise để lấy index ngẫu nhiên

        anchor_tokens, pos_tokens = int(L*ratio), int(L*2*ratio)
        ids_keep_anchor, ids_keep_pos = ids_shuffle[:, :, :anchor_tokens], ids_shuffle[:, :, anchor_tokens:pos_tokens]
        
        # Lấy mẫu anchor và positive từ chuỗi gốc dựa trên index ngẫu nhiên
        x_anchor = torch.gather(
            x, dim=2, index=ids_keep_anchor.unsqueeze(-1).repeat(1, 1, 1, D))
        x_pos = torch.gather(
            x, dim=2, index=ids_keep_pos.unsqueeze(-1).repeat(1, 1, 1, D))
        return x_anchor, x_pos
    

    def random_vertical_sample(self, x, ratio=0.5):
        """
        Lấy mẫu ngẫu nhiên theo chiều DỌC (Height).
        Ý nghĩa: Khi xáo trộn hoặc lấy mẫu ngẫu nhiên các hàng (Height), cấu trúc dọc bị phá vỡ,
        buộc mô hình phải dựa vào thông tin HÀNG NGANG (Horizontal) còn sót lại -> Học đặc trưng Horizontal.
        Input x: [(H*W), B, D] -> flattened sequence
        """
        x = rearrange(x, "( H W ) B D -> B H W D", H=4) # Khôi phục không gian 2D
        B, H, W, D = x.shape
        noise = torch.rand(B, H, device=x.device) # Noise theo chiều cao
        ids_shuffle = torch.argsort(noise, dim=1)
        tokens = int(H*ratio)
        
        # Lấy các hàng ngẫu nhiên
        x_sample = torch.gather(
            x, dim=1, index=ids_shuffle[:, :tokens].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, W, D))
        x_sample = rearrange(x_sample, 'B H W D -> ( H W ) B D') # Flatten lại
        
        return x_sample
    

    def random_horizontal_sample(self, x, ratio=0.5):
        """
        Lấy mẫu ngẫu nhiên theo chiều NGANG (Width).
        Ý nghĩa: Khi lấy mẫu ngẫu nhiên các cột, cấu trúc ngang bị phá vỡ,
        buộc mô hình phải dựa vào thông tin DỌC (Vertical) -> Học đặc trưng Vertical.
        """
        x = rearrange(x, "( H W ) B D -> B W H D", H=4) # Lưu ý thứ tự trục: B, Width, Height, Dim
        B, W, H, D = x.shape
        noise = torch.rand(B, W, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        tokens = int(W*ratio)
        
        # Lấy các cột ngẫu nhiên
        x_sample = torch.gather(
            x, dim=1, index=ids_shuffle[:, :tokens].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, D))
        x_sample = rearrange(x_sample, 'B W H D -> ( H W ) B D')
        
        return x_sample

    def forward(self, style, content, wid):
        """
        Input:
            style: Ảnh phong cách (B, 1, H_img, W_img)
            content: Ảnh nội dung text (B, 1, H_c, W_c)
            wid: Writer ID (nhãn tác giả) dùng cho loss
        """
        batch_size, in_planes, h, w = style.shape
        
        # ==========================================
        # 1. Style Encoding Path
        # ==========================================
        # Qua ResNet backbone
        style = self.Feat_Encoder(style) 
        # Reshape lại output của ResNet (lưu ý c=256, h=4 là giả định kích thước feature map)
        style = rearrange(style, 'n (c h w) ->n c h w', c=256, h=4).contiguous()
        
        # Qua lớp Dilation Conv để tăng vùng nhìn (Receptive Field)
        # Output shape: [B, 512, 4, W_feat] (Channel tăng lên 512 từ layer này)
        style = self.style_dilation_layer(style)    
        
        # Thêm thông tin vị trí 2D
        style = self.add_position2D(style)
        
        # Flatten không gian (H, W) thành chuỗi (Sequence) để đưa vào Transformer
        # Shape: [(H*W), B, C] - chuẩn input của Transformer PyTorch (Seq, Batch, Dim)
        style = rearrange(style, 'n c h w ->(h w) n c').contiguous()
        
        # Qua Base Encoder
        base_style = self.base_encoder(style)   # [4*W_feat, B, 512]

        # Tách thành 2 nhánh feature riêng biệt
        vertical_style = self.vertical_head(base_style)   # Nhánh dọc [Sequence, B, 512]
        horizontal_style = self.horizontal_head(base_style) # Nhánh ngang

        # ==========================================
        # 2. Style Disentanglement (Tách style) & Loss Calculation
        # ==========================================
        
        # --- Nhánh Vertical Style Loss ---
        # Lấy mẫu theo chiều NGANG (horizontal sample) -> Phá vỡ liên kết ngang -> Để lại thông tin DỌC.
        # Ví dụ: Độ cao của từng chữ cái.
        vertical_style_proxy = self.random_horizontal_sample(vertical_style)    # [2*W, B, 512] (giả sử ratio 0.5)
        vertical_style_proxy = self.vertical_pro_mlp(vertical_style_proxy)      # Qua MLP Projection
        vertical_style_proxy = torch.mean(vertical_style_proxy, dim=0)          # Global Pooling -> Vector đại diện [B, 512]
        vertical_style_loss = self.vertical_proxy(vertical_style_proxy, wid)    # Tính Loss phân loại style

        # --- Nhánh Horizontal Style Loss ---
        # Lấy mẫu theo chiều DỌC (vertical sample) -> Phá vỡ liên kết dọc -> Để lại thông tin NGANG.
        # Ví dụ: Khoảng cách giữa các ký tự, nét nối liền (ligature).
        horizontal_style_proxy = self.random_vertical_sample(horizontal_style)
        horizontal_style_proxy = self.horizontal_pro_mlp(horizontal_style_proxy)
        horizontal_style_proxy = torch.mean(horizontal_style_proxy, dim=0)
        horizontal_style_loss = self.horizontal_proxy(horizontal_style_proxy, wid)

        # ==========================================
        # 3. Content Encoding Path
        # ==========================================
        # content Input: [B, T_chars, H, W] -> T_chars là số lượng ký tự/ảnh con
        # Gộp Batch và Time (T) lại để xử lý song song qua CNN
        content = rearrange(content, 'n t h w ->(n t) 1 h w').contiguous()
        content = self.content_encoder(content) # Extract feature
        
        # Tách lại Batch và Time, gộp Spatial thành 1 chiều Feature
        # Output: [Time, Batch, Feature_Dim]
        content = rearrange(content, '(n t) c h w ->t n (c h w)', n=batch_size).contiguous() 
        content = self.add_position1D(content) # Thêm vị trí 1D cho chuỗi ký tự
        
        # ==========================================
        # 4. Decoding (Fusion Content + Style)
        # ==========================================
        # Decoder hoạt động theo cơ chế Cross-Attention:
        # Query = Content, Key/Value = Style
        
        # Bước 1: Trộn style ngang vào content
        style_hs = self.horizontal_decoder(content, horizontal_style, tgt_mask=None)
        
        # Bước 2: Trộn style dọc vào kết quả trên
        # Input của vertical_decoder là output của horizontal_decoder (style_hs[0])
        hs = self.vertical_decoder(style_hs[0], vertical_style, tgt_mask=None)
        
        # hs[0] shape: [Time, Batch, Dim]. Permute về [Batch, Time, Dim] để return
        return hs[0].permute(1, 0, 2).contiguous(), vertical_style_loss, horizontal_style_loss 
    
    def generate(self, style, content):
        """
        Hàm dùng cho Inference (Test). 
        Logic giống hệt forward nhưng bỏ qua bước tính Loss và Sampling ngẫu nhiên.
        """
        batch_size, in_planes, h, w = style.shape
        
        # --- Encode Style ---
        style = self.Feat_Encoder(style)
        style = rearrange(style, 'n (c h w) ->n c h w', c=256, h=4).contiguous()
        style = self.style_dilation_layer(style)    # [B, 512, 4, W]
        style = self.add_position2D(style)
        style = rearrange(style, 'n c h w ->(h w) n c').contiguous()
        
        base_style = self.base_encoder(style)   # [4*W, B, 512]
        vertical_style = self.vertical_head(base_style)
        horizontal_style = self.horizontal_head(base_style)

        # Lưu ý: Trong comment gốc có phần random shuffle đã bị comment đi (disable).
        # Khi generate ta muốn giữ nguyên toàn bộ thông tin style, không xáo trộn.

        # --- Encode Content ---
        content = rearrange(content, 'n t h w ->(n t) 1 h w').contiguous()
        content = self.content_encoder(content)
        content = rearrange(content, '(n t) c h w ->t n (c h w)', n=batch_size).contiguous() 
        content = self.add_position1D(content)
        
        # --- Decode (Mix) ---
        style_hs = self.horizontal_decoder(content, horizontal_style, tgt_mask=None)
        hs = self.vertical_decoder(style_hs[0], vertical_style, tgt_mask=None)
        
        # Return kết quả cuối cùng: [Batch, Time, Dim]
        return hs[0].permute(1, 0, 2).contiguous()