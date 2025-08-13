import torch
from torch import nn
import torch.nn.functional as F
from lib.swin_transformer import SwinTransformer
from lib.DCTlayer import MultiSpectralAttentionLayer

class BasicConv2d(nn.Module):
    # conv + bn
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)

        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class MSD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSD, self).__init__()

        self.rgb_feat = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.depth_feat = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.thermal_feat = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.rgb_score = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7,
                      padding=7 // 2, bias=False),
            nn.Sigmoid()
        )

        self.depth_score = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7,
                      padding=7 // 2, bias=False),
            nn.Sigmoid()
        )

        self.thermal_score = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7,
                      padding=7 // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        rgb_feat = self.rgb_feat(x[0])
        thermal_feat = self.thermal_feat(x[1])
        depth_feat = self.depth_feat(x[2])

        max_rgb, _ = torch.max(rgb_feat, dim=1, keepdim=True)
        avg_rgb = torch.mean(rgb_feat, dim=1, keepdim=True)
        rgb_q = self.rgb_score(torch.cat([max_rgb, avg_rgb], dim=1))

        max_t, _ = torch.max(thermal_feat, dim=1, keepdim=True)
        avg_t = torch.mean(thermal_feat, dim=1, keepdim=True)
        thermal_q = self.thermal_score(torch.cat([max_t, avg_t], dim=1))

        max_d, _ = torch.max(depth_feat, dim=1, keepdim=True)
        avg_d = torch.mean(depth_feat, dim=1, keepdim=True)
        depth_q = self.depth_score(torch.cat([max_d, avg_d], dim=1))

        return [rgb_feat, thermal_feat, depth_feat], [rgb_q, thermal_q, depth_q]


class CrossIQ(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CrossIQ, self).__init__()

        self.query_rgb = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_rgb = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_rgb = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.query_depth = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_depth = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_depth = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.query_thermal = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_thermal = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_thermal = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.gating = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.out_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, s):
        B, C, H, W = x[0].shape

        qr = self.query_rgb(x[0]) * s[0]
        kr = self.key_rgb(x[0]) * s[0]
        vr = self.value_rgb(x[0])

        qt = self.query_thermal(x[1]) * s[1]
        kt = self.key_thermal(x[1]) * s[1]
        vt = self.value_thermal(x[1])

        qd = self.query_depth(x[2]) * s[2]
        kd = self.key_depth(x[2]) * s[2]
        vd = self.value_depth(x[2])

        def flatten(x): return x.view(B, x.size(1), -1)

        qr, kr, vr = flatten(qr), flatten(kr), flatten(vr)
        qd, kd, vd = flatten(qd), flatten(kd), flatten(vd)
        qt, kt, vt = flatten(qt), flatten(kt), flatten(vt)

        scale = qr.size(1) ** 0.5

        attn_rgb = F.softmax((qr.transpose(1, 2) @ kd + qr.transpose(1, 2) @ kt) / scale, dim=-1)
        attn_depth = F.softmax((qd.transpose(1, 2) @ kr + qd.transpose(1, 2) @ kt) / scale, dim=-1)
        attn_thermal = F.softmax((qt.transpose(1, 2) @ kr + qt.transpose(1, 2) @ kd) / scale, dim=-1)

        weighted_rgb = (attn_rgb @ vr.transpose(1, 2)).transpose(1, 2).view(B, -1, H, W)
        weighted_depth = (attn_depth @ vd.transpose(1, 2)).transpose(1, 2).view(B, -1, H, W)
        weighted_thermal = (attn_thermal @ vt.transpose(1, 2)).transpose(1, 2).view(B, -1, H, W)

        gated_rgb = self.gating(weighted_rgb) * weighted_rgb
        gated_depth = self.gating(weighted_depth) * weighted_depth
        gated_thermal = self.gating(weighted_thermal) * weighted_thermal

        fused = gated_rgb + gated_depth + gated_thermal
        output = self.out_proj(fused)

        return output

class FASPP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FASPP, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )

        self.branch1_0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1_1 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 3), dilation=3),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(3, 0), dilation=3)
        )

        self.branch2_0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch2_1 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 5), dilation=5),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(5, 0), dilation=5)
        )

        self.branch3_0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch3_1 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 7), dilation=7),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(7, 0), dilation=7)
        )

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.conv_cat = BasicConv2d(5 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

        self.attn = MultiSpectralAttentionLayer(out_channel, 7, 7)

    def forward(self, x):
        size = x.shape[2:]
        x0 = self.branch0(x)

        x1_0 = self.branch1_0(x)
        x1 = self.branch1_1(x1_0 + x0)

        x2_0 = self.branch2_0(x)
        x2 = self.branch2_1(x2_0 + x1)

        x3_0 = self.branch3_0(x)
        x3 = self.branch3_1(x3_0 + x2)

        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', align_corners=False)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3, global_feat), 1))

        x = self.relu(self.attn(x_cat) + self.conv_res(x))

        return x



class RR(nn.Module):
    def __init__(self,in_ch,inc_ch):
        super(RR, self).__init__()

        self.conv0 = nn.Conv2d(in_ch,inc_ch,3,padding=1)

        self.conv1 = nn.Conv2d(inc_ch,64,3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3 = nn.Conv2d(64,64,3,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv4 = nn.Conv2d(64,64,3,padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64,64,3,padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64,1,3,padding=1)

        self.conv_out = nn.Conv2d(192 + 1,1,3,padding=1)
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self,x):

        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx,hx4),1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx,hx3),1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx,hx2),1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx,hx1),1))))

        residual = self.conv_d0(d1)

        return self.conv_out(x + residual)

class TriFuse(nn.Module):
    def __init__(self,pretrained=False):
        super(TriFuse, self).__init__()
        self.swin1 = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        self.ReLU = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))


        self.msd1 = MSD(in_channels=128, out_channels=64)
        self.msd2 = MSD(in_channels=256, out_channels=128)
        self.msd3 = MSD(in_channels=512, out_channels=256)
        self.msd4 = MSD(in_channels=1024, out_channels=512)

        self.ciq1 = CrossIQ(in_channels=64, out_channels=64)
        self.ciq2 = CrossIQ(in_channels=128, out_channels=128)
        self.ciq3 = CrossIQ(in_channels=256, out_channels=256)
        self.ciq4 = CrossIQ(in_channels=512, out_channels=512)

        self.decoder1 = FASPP(in_channel=128, out_channel=32)
        self.decoder2 = FASPP(in_channel=256, out_channel=64)
        self.decoder3 = FASPP(in_channel=512, out_channel=128)
        self.decoder4 = FASPP(in_channel=512, out_channel=256)

        self.final_1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )
        self.final_1_d = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )
        self.final_1_t = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )

        self.final_4_vdt = nn.Sequential(
            Conv(256, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )

        self.final_3_vdt = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )

        self.final_2_vdt = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, 1, 1, bn=False, relu=False)
        )

        self.final_1_vdt = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, 1, 1, bn=False, relu=False)
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        self.rr = RR(192 + 1, 64)

    def forward(self, rgb, t, d):
        score_list_t, score_PE = self.swin1(t)
        score_list_rgb, score_PE = self.swin1(rgb)
        score_list_d, score_PE = self.swin1(d)

        x1 = [score_list_rgb[0],score_list_t[0],score_list_d[0]]
        x2 = [score_list_rgb[1],score_list_t[1],score_list_d[1]]
        x3 = [score_list_rgb[2],score_list_t[2],score_list_d[2]]
        x4 = [score_list_rgb[3],score_list_t[3],score_list_d[3]]


        x4e, x4s = self.msd4(x4)
        x4u = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(xe_) for xe_ in x4e]

        x3e = [x3[i] + x4u[i] for i in range(3)]
        x3e, x3s = self.msd3(x3e)
        x3u = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(xe_) for xe_ in x3e]

        x2e = [x2[i] + x3u[i] for i in range(3)]
        x2e, x2s = self.msd2(x2e)
        x2u = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(xe_) for xe_ in x2e]

        x1e = [x1[i] + x2u[i] for i in range(3)]
        x1e, x1s = self.msd1(x1e)
        x1u = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(xe_) for xe_ in x1e]


        x1e_v = x1u[0]
        x1e_pred = self.final_1(x1e_v)
        x1e_pred = self.up2(x1e_pred)

        x1e_t = x1u[1]
        x1e_pred_t = self.final_1_t(x1e_t)
        x1e_pred_t = self.up2(x1e_pred_t)

        x1e_d = x1u[2]
        x1e_pred_d = self.final_1_d(x1e_d)
        x1e_pred_d = self.up2(x1e_pred_d)

        x1_vdt = self.ciq1(x1e, x1s)
        x1_vdt = self.up2(x1_vdt)
        x2_vdt = self.ciq2(x2e, x2s)
        x2_vdt = self.up2(x2_vdt)
        x3_vdt = self.ciq3(x3e, x3s)
        x3_vdt = self.up2(x3_vdt)
        x4_vdt = self.ciq4(x4e, x4s)
        x4_vdt = self.up2(x4_vdt)

        x4_vdt_f = self.decoder4(x4_vdt)
        x4_vdt_f = self.up2(x4_vdt_f)
        x4_pred_vdt = self.final_4_vdt(x4_vdt_f)

        x3_vdt = torch.cat((x3_vdt, x4_vdt_f), dim=1)
        x3_vdt_f = self.decoder3(x3_vdt)
        x3_vdt_f = self.up2(x3_vdt_f)
        x3_pred_vdt = self.final_3_vdt(x3_vdt_f)

        x2_vdt = torch.cat((x2_vdt, x3_vdt_f), dim=1)
        x2_vdt_f = self.decoder2(x2_vdt)
        x2_vdt_f = self.up2(x2_vdt_f)
        x2_pred_vdt = self.final_2_vdt(x2_vdt_f)

        x1_vdt = torch.cat((x1_vdt, x2_vdt_f), dim=1)
        x1_vdt_f = self.decoder1(x1_vdt)
        x1_vdt_f = self.up2(x1_vdt_f)
        x1_pred_vdt = self.final_1_vdt(x1_vdt_f)
        x2_pred_vdt = self.up2(x2_pred_vdt)
        x3_pred_vdt = self.up4(x3_pred_vdt)
        x4_pred_vdt = self.up8(x4_pred_vdt)
        x_pred = [x1_pred_vdt, x2_pred_vdt, x3_pred_vdt, x4_pred_vdt]

        x_refine = torch.cat(
            [x1_pred_vdt, self.up4(x1e[0] * x1s[0]), self.up4(x1e[1] * x1s[1]), self.up4(x1e[2] * x1s[2])], dim=1)
        x_refine = self.rr(x_refine)

        return x1e_pred, x1e_pred_t, x1e_pred_d, x_pred, x_refine

    def load_pretrained_model(self):
        self.swin1.load_state_dict(torch.load('./pretrained/swin_base_patch4_window12_384_22k.pth')['model'],strict=False)
        print('loading pretrained model success!')



class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x



