import torch
import torch.nn as nn
from torch.nn import init

from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from mmcv.runner import _load_checkpoint


class Block(nn.Module):
    def __init__(self, in_dim, out_dim, stride, ClaSiLU):
        super(Block, self).__init__()
        self.stride = stride
        self.conv_pw = nn.Sequential(
            nn.SyncBatchNorm(in_dim),
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
        )

        self.conv_dw = nn.Sequential(
            nn.SyncBatchNorm(out_dim),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=stride, padding=1, groups=out_dim),
        )

        self.ClaSiLU_gate = None
        if ClaSiLU:
            self.ClaSiLU_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.SyncBatchNorm(in_dim),
                nn.Conv2d(in_dim, out_dim, kernel_size=1),
            )

        self.down = nn.AvgPool2d(3, 2, 1) if stride == 2 else nn.Identity()

    def forward(self, x):
        def ClaSiLU_activation(y, x):
            return y * torch.sigmoid(y + self.ClaSiLU_gate(x))
        y1 = self.conv_pw(x)
        if self.stride == 2:
            y1_1, y1_2 = torch.chunk(y1, dim=1, chunks=2)
            y1 = torch.cat((y1_1+x, y1_2+x), dim=1)
        else:
            y1 = y1 + x
        y1 = ClaSiLU_activation(y1,x)     
        y2 = self.conv_dw(y1) + self.down(y1)
        return y2

class Classfier(nn.Module):
    def __init__(self, final_dim, num_classes, distillation=True):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, final_dim, bias=False),
            nn.BatchNorm1d(final_dim),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(final_dim, num_classes),
        )
        self.distillation = distillation
        if distillation:
            self.classifier_dist = nn.Sequential(
            nn.Linear(final_dim, final_dim, bias=False),
            nn.BatchNorm1d(final_dim),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(final_dim, num_classes),
        )
    def forward(self, x):
        if self.distillation:
            x = self.classifier(x), self.classifier_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.classifier(x)
        return x

class RepMobile(nn.Module):
    def __init__(self, num_classes=1000, dims=[128,256,512,1024], layers=[2,2,6,2], ClaSiLU=[1,1,1,1], pretrained=None, distillation=False, init_cfg=None,out_indices=[]):
        super(RepMobile, self).__init__()
        block = Block
        self.stem = nn.Sequential(            
            nn.Conv2d(3, dims[0]//2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(dims[0]//2),                 
            nn.ReLU(True),
        )

        self.layer0 = nn.Sequential(
            block(dims[0]//2, dims[0], 2, ClaSiLU[0]),
            *[block(dims[0], dims[0], 1, ClaSiLU[0]) for _ in range(layers[0]-1)],
        )
        self.layer1 = nn.Sequential(
            block(dims[0], dims[1], 2, ClaSiLU[1]),
            *[block(dims[1], dims[1], 1, ClaSiLU[1]) for _ in range(layers[1]-1)],
        )
        self.layer2 = nn.Sequential(
            block(dims[1], dims[2], 2, ClaSiLU[2]),
            *[block(dims[2], dims[2], 1, ClaSiLU[2]) for _ in range(layers[2]-1)],
        )
        self.layer3 = nn.Sequential(
            block(dims[2], dims[3], 2, ClaSiLU[3]),
            *[block(dims[3], dims[3], 1, ClaSiLU[3]) for _ in range(layers[3]-1)],
        )

        # final_dim = max(1024, dims[3])
        # self.head = nn.Sequential(
        #     nn.Conv2d(dims[3], final_dim, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.SyncBatchNorm(final_dim),
        #     nn.ReLU(True),
        # )
        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.classifier = Classfier(final_dim, num_classes, distillation)
        
        self.init_params()
        self.init_cfg = init_cfg
        assert(self.init_cfg is not None)
        self.out_indices = out_indices
        self.init_weights()
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)
            logger.info(f"Miss {missing_keys}")
            logger.info(f"Unexpected {unexpected_keys}")  
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.SyncBatchNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, x):
        out = self.stem(x)
        outs = []
        out = self.layer0(out)
        outs.append(out)
        out = self.layer1(out)
        outs.append(out)
        out = self.layer2(out)
        outs.append(out)
        out = self.layer3(out)
        outs.append(out)     
        return outs


@BACKBONES.register_module()
def RepMobile_L(pretrained=False, num_classes = 1000, distillation=False, init_cfg=None, out_indices=[], **kwargs):   
    return RepMobile(dims=[96,192,384,768], layers=[4,8,20,8], ClaSiLU=[1,1,1,1],init_cfg=init_cfg, distillation=distillation, out_indices=out_indices)

@BACKBONES.register_module()
def RepMobile_M(pretrained=False, num_classes = 1000, distillation=False, init_cfg=None, out_indices=[], **kwargs):   
    return RepMobile(dims=[80,160,320,640], layers=[3,6,15,3], ClaSiLU=[1,1,1,1],init_cfg=init_cfg, distillation=distillation, out_indices=out_indices)

@BACKBONES.register_module()
def RepMobile_S(pretrained=False, num_classes = 1000, distillation=False, init_cfg=None, out_indices=[], **kwargs):   
    return RepMobile(dims=[64,128,256,512], layers=[3,6,12,3], ClaSiLU=[1,1,1,1], init_cfg=init_cfg, distillation=distillation, out_indices=out_indices)
