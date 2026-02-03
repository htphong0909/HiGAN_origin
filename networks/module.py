import torch
from torch import nn
from networks.block import Conv2dBlock, ActFirstResBlock, DeepBLSTM, DeepGRU, DeepLSTM
from networks.utils import _len2mask, init_weights


class StyleEncoder_origin(nn.Module):
    def __init__(self, style_dim=32, resolution=16, max_dim=256, in_channel=1, init='N02',
                 SN_param=False, norm='none', share_wid=True):
        super(StyleEncoder_origin, self).__init__()
        self.reduce_len_scale = 16 # Ảnh sẽ bị thu nhỏ chiều ngang 16 lần
        self.share_wid = share_wid
        self.style_dim = style_dim

        # --- GIAI ĐOẠN 1: XÂY DỰNG BACKBONE (Trích xuất đặc trưng cơ bản) ---
        nf = resolution # nf: số channel bắt đầu (ví dụ: 16 hoặc 32)
        cnn_f = [
            nn.ConstantPad2d(2, -1), # Pad thêm để giữ size khi dùng Conv 5x5
            Conv2dBlock(in_channel, nf, 5, 1, 0, norm='none', activation='none')
        ]
        
        # Vòng lặp này đi qua 2 lần MaxPool -> Giảm size 4 lần (2x2)
        for i in range(2):
            nf_out = min([int(nf * 2), max_dim])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', norm, sn=SN_param)]
            cnn_f += [nn.ReflectionPad2d((1, 1, 0, 0))]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', norm, sn=SN_param)]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)] # Downsample lần 1, 2
            nf = min([nf_out, max_dim])

        df = nf
        # Thêm 1 lần MaxPool nữa -> Tổng cộng giảm size 8 lần
        for i in range(1):
            df_out = min([int(df * 2), max_dim])
            cnn_f += [ActFirstResBlock(df, df, None, 'lrelu', norm, sn=SN_param)]
            cnn_f += [ActFirstResBlock(df, df_out, None, 'lrelu', norm, sn=SN_param)]
            cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)] # Downsample lần 3
            df = min([df_out, max_dim])

        df_out = min([int(df * 2), max_dim])
        cnn_f += [ActFirstResBlock(df, df, None, 'lrelu', norm, sn=SN_param)]
        cnn_f += [ActFirstResBlock(df, df_out, None, 'lrelu', norm, sn=SN_param)]
        self.cnn_backbone = nn.Sequential(*cnn_f)

        # --- GIAI ĐOẠN 2: STYLE EXTRACTION (Nén thành vector phong cách) ---
        # Lớp này dùng stride=2 -> Tổng cộng từ đầu đến đây giảm size 16 lần (H/16, W/16)
        cnn_e = [
            nn.ReflectionPad2d((1, 1, 0, 0)),
            Conv2dBlock(df_out, df, 3, 2, 0, norm=norm, activation='lrelu', activation_first=True)
        ]
        self.cnn_wid = nn.Sequential(*cnn_e)
        
        # Mạng MLP nhỏ để tinh chỉnh vector style
        self.linear_style = nn.Sequential(
            nn.Linear(df, df),
            nn.LeakyReLU()
        )
        self.mu = nn.Linear(df, style_dim) # Vector trung bình
        self.logvar = nn.Linear(df, style_dim) # Vector độ lệch (dùng trong VAE)

        # Khởi tạo trọng số (Logvar bias thấp để bắt đầu với variance nhỏ, tăng độ ổn định)
        if init != 'none': init_weights(self, init)
        torch.nn.init.constant_(self.logvar.weight.data, 0.)
        torch.nn.init.constant_(self.logvar.bias.data, -10.)

    def forward(self, img, img_len, wid_cnn_backbone=None, vae_mode=False):
        # 1. Đi qua Backbone
        # Shape: (B, 1, H, W) -> (B, 512, H/8, W/8)
        if self.share_wid:
            feat = wid_cnn_backbone(img)
        else:
            feat = self.cnn_backbone(img)

        # 2. Qua cnn_wid (Downsample lần cuối)
        # Shape: (B, 512, H/8, W/8) -> (B, 256, H/16, W/16)
        # .squeeze(-2) loại bỏ chiều Height (giả sử H lúc này = 1) -> (B, 256, W/16)
        img_len = img_len // self.reduce_len_scale
        out_e = self.cnn_wid(feat).squeeze(-2) 

        # 3. Xử lý Masking (Loại bỏ phần padding trắng của ảnh để không tính vào style)
        # img_len_mask: (B, 1, W/16)
        img_len_mask = _len2mask(img_len, out_e.size(-1)).unsqueeze(1).float().detach()
        
        # Tính trung bình đặc trưng theo chiều ngang (Global Average Pooling có Mask)
        # (B, 256, W/16) -> (B, 256)
        style = (out_e * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)
        
        # 4. Qua lớp Linear tinh chỉnh
        # Shape: (B, 256)
        style = self.linear_style(style)
        
        # 5. Output đầu ra
        # mu: (B, style_dim) - Ví dụ (B, 96)
        mu = self.mu(style)
        
        if vae_mode:
            # logvar: (B, style_dim)
            logvar = self.logvar(style)
            # Lấy mẫu (Reparameterization trick): z = mu + std * epsilon
            encode_z = self.sample(mu, logvar)
            return encode_z, mu, logvar
        else:
            return mu



#---------------------------------------------------
from models.resnet_dilation import resnet18 as resnet18_dilation
from einops import rearrange

class StyleEncoder(nn.Module):
    def __init__(self, style_dim=32, resolution=16, max_dim=256, in_channel=1, init='N02',
                 SN_param=False, norm='none', share_wid=True):
        super(StyleEncoder, self).__init__()
        self.style_dim = style_dim
        
        # --- GIAI ĐOẠN 1: BACKBONE RESNET (Theo thiết kế của DiffBrush) ---
        # Sử dụng ResNet18 làm bộ trích xuất đặc trưng cơ sở [cite: 140, 280]
        self.backbone = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.backbone.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Loại bỏ layer4 và các lớp phía sau để giữ feature map thay vì vector phân loại [cite: 189]
        self.backbone.layer4 = nn.Identity()
        self.backbone.fc = nn.Identity()
        self.backbone.avgpool = nn.Identity()
        
        # Thêm lớp Dilation để tăng Receptive Field như DiffBrush [cite: 190]
        # Giúp mô hình "nhìn" được các nét chữ dài và rộng hơn
        self.style_dilation_layer = resnet18_dilation().conv5_x 
        
        # --- GIAI ĐOẠN 2: ADAPTATION LAYER (Khớp với shape của bạn) ---
        # Sau dilation, channel thường là 512, ta nén về max_dim (ví dụ 256)
        self.adapter = nn.Sequential(
            nn.Conv2d(512, max_dim, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling đưa về (B, max_dim, 1, 1)
        )

        # --- GIAI ĐOẠN 3: OUTPUT HEADS (Giống style encoder của bạn) ---
        self.linear_style = nn.Sequential(
            nn.Linear(max_dim, max_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.mu = nn.Linear(max_dim, style_dim)
        self.logvar = nn.Linear(max_dim, style_dim)

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, img, vae_mode=False):
        # 1. Trích xuất đặc trưng qua ResNet + Dilation [cite: 112, 190]
        x = self.backbone.conv1(img)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        
        # Qua lớp giãn nở để bắt được inter-word patterns (mô hình dòng chữ) [cite: 190, 201]
        feat = self.style_dilation_layer(x) 
        
        # 2. Nén không gian về vector phẳng (B, max_dim)
        style_vec = self.adapter(feat).view(feat.size(0), -1)
        
        # 3. Tinh chỉnh phong cách
        style_refined = self.linear_style(style_vec)
        
        # 4. Đầu ra
        mu = self.mu(style_refined)
        
        if vae_mode:
            logvar = self.logvar(style_refined)
            encode_z = self.sample(mu, logvar)
            return encode_z, mu, logvar
        else:
            return mu




class WriterIdentifier(nn.Module):
    def __init__(self, n_writer=284, resolution=16, max_dim=256, in_channel=1, init='N02',
                 SN_param=False, dropout=0.0, norm='bn'):

        super(WriterIdentifier, self).__init__()
        self.reduce_len_scale = 16

        ######################################
        # Construct Backbone
        ######################################
        nf = resolution
        cnn_f = [nn.ConstantPad2d(2, -1),
                 Conv2dBlock(in_channel, nf, 5, 1, 0,
                             norm='none',
                             activation='none')]
        for i in range(2):
            nf_out = min([int(nf * 2), max_dim])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', norm, sn=SN_param, dropout=dropout / 2)]
            cnn_f += [nn.ReflectionPad2d((1, 1, 0, 0))]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', norm, sn=SN_param, dropout=dropout / 2)]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)]
            nf = min([nf_out, max_dim])

        df = nf
        for i in range(1):
            df_out = min([int(df * 2), max_dim])
            cnn_f += [ActFirstResBlock(df, df, None, 'lrelu', norm, sn=SN_param, dropout=dropout)]
            cnn_f += [ActFirstResBlock(df, df_out, None, 'lrelu', norm, sn=SN_param, dropout=dropout)]
            cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)]
            df = min([df_out, max_dim])

        df_out = min([int(df * 2), max_dim])
        cnn_f += [ActFirstResBlock(df, df, None, 'lrelu', norm, sn=SN_param, dropout=dropout / 2)]
        cnn_f += [ActFirstResBlock(df, df_out, None, 'lrelu', norm, sn=SN_param, dropout=dropout / 2)]
        self.cnn_backbone = nn.Sequential(*cnn_f)

        ######################################
        # Construct WriterIdentifier
        ######################################
        cnn_w = [nn.ReflectionPad2d((1, 1, 0, 0)),
                 Conv2dBlock(df_out, df, 3, 2, 0,
                             norm=norm,
                             activation='lrelu',
                             activation_first=True)]
        self.cnn_wid = nn.Sequential(*cnn_w)
        self.linear_wid = nn.Sequential(
            nn.Linear(df, df),
            nn.LeakyReLU(),
            nn.Linear(df, n_writer),
        )

        if init != 'none':
            init_weights(self, init)

    def forward(self, img, img_len):
        feat = self.cnn_backbone(img)
        img_len = img_len // self.reduce_len_scale
        out_w = self.cnn_wid(feat).squeeze(-2)
        img_len_mask = _len2mask(img_len, out_w.size(-1)).unsqueeze(1).float().detach()
        wid_feat = (out_w * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)
        wid_logits = self.linear_wid(wid_feat)
        return wid_logits



class Recognizer(nn.Module):
    # resolution: 32  max_dim: 512  in_channel: 1  norm: 'none'  init: 'N02'  dropout: 0.  n_class: 72  rnn_depth: 0
    def __init__(self, n_class, resolution=16, max_dim=256, in_channel=1, norm='none',
                 init='none', rnn_depth=1, dropout=0.0, bidirectional=True):
        super(Recognizer, self).__init__()
        self.len_scale = 8
        self.use_rnn = rnn_depth > 0
        self.bidirectional = bidirectional

        ######################################
        # Construct Backbone
        ######################################
        nf = resolution
        cnn_f = [nn.ConstantPad2d(2, -1),
                 Conv2dBlock(in_channel, nf, 5, 1, 0,
                             norm='none',
                             activation='none')]
        for i in range(2):
            nf_out = min([int(nf * 2), max_dim])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'relu', norm, 'zero', dropout=dropout / 2)]
            cnn_f += [nn.ZeroPad2d((1, 1, 0, 0))]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'relu', norm, 'zero', dropout=dropout / 2)]
            cnn_f += [nn.ZeroPad2d(1)]
            cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)]
            nf = min([nf_out, max_dim])

        df = nf
        for i in range(2):
            df_out = min([int(df * 2), max_dim])
            cnn_f += [ActFirstResBlock(df, df, None, 'relu', norm, 'zero', dropout=dropout)]
            cnn_f += [ActFirstResBlock(df, df_out, None, 'relu', norm, 'zero', dropout=dropout)]
            if i < 1:
                cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)]
            else:
                cnn_f += [nn.ZeroPad2d((1, 1, 0, 0))]
            df = min([df_out, max_dim])

        ######################################
        # Construct Classifier
        ######################################
        cnn_c = [nn.ReLU(),
                 Conv2dBlock(df, df, 3, 1, 0,
                             norm=norm,
                             activation='relu')]

        self.cnn_backbone = nn.Sequential(*cnn_f)
        self.cnn_ctc = nn.Sequential(*cnn_c)
        if self.use_rnn:
            if bidirectional:
                self.rnn_ctc = DeepBLSTM(df, df, rnn_depth, bidirectional=True)
            else:
                self.rnn_ctc = DeepLSTM(df, df, rnn_depth)
        self.ctc_cls = nn.Linear(df, n_class)

        if init != 'none':
            init_weights(self, init)

    def forward(self, x, x_len=None):
        cnn_feat = self.cnn_backbone(x)
        cnn_feat2 = self.cnn_ctc(cnn_feat)
        ctc_feat = cnn_feat2.squeeze(-2).transpose(1, 2)
        if self.use_rnn:
            if self.bidirectional:
                ctc_len = x_len // (self.len_scale  + 1e-8)
            else:
                ctc_len = None
            ctc_feat = self.rnn_ctc(ctc_feat, ctc_len)
        logits = self.ctc_cls(ctc_feat)
        if self.training:
            logits = logits.transpose(0, 1).log_softmax(2)
            logits.requires_grad_(True)
        return logits
