import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import roi_align
from torchvision.transforms.functional import hflip


def D(p, z, version='simplified', hidden_dim=1): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=hidden_dim) # l2-normalize 
        z = F.normalize(z, dim=hidden_dim) # l2-normalize 
        return -(p*z).sum(dim=hidden_dim).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=hidden_dim).mean()
    else:
        raise Exception



class projection_1x1Conv(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim, 1),
            nn.BatchNorm2d(hidden_dim)
        )
        self.num_layers = 3

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 


class prediction_1x1Conv(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Conv2d(hidden_dim, out_dim, 1)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 


class FPNLayersChooser(nn.Module):
    '''layer list: 0,1,2,3,pool'''
    def __init__(self, layer_list=[0]):
        super().__init__()
        self.layer_list = [str(i) for i in layer_list]

    def forward(self, x):
        if len(self.layer_list) == 1:
            return x[self.layer_list[0]]
        else:
            return tuple(x[i] for i in self.layer_list)


class PixSiam(nn.Module):
    def __init__(self,
                 backbone=resnet_fpn_backbone('resnet18',
                                              pretrained=False,
                                              trainable_layers=5)):
        super().__init__()

        self.backbone = backbone
        self.projector = projection_1x1Conv(backbone.out_channels)

        self.encoder = nn.Sequential(  # f encoder
            self.backbone,
            FPNLayersChooser(),
            self.projector
        )
        self.predictor = prediction_1x1Conv()

        # fpn stride
        self.stride = 4  # [4, 8, 16, 32, 64] TODO

    def crop_trans(self, crop1, crop2):
        z0_1, z1_1 = crop1[:, :2], crop1[:, 2:]
        z0_2, z1_2 = crop2[:, :2], crop2[:, 2:]
        max0, min1 = torch.max(z0_1, z0_2), torch.min(z1_1, z1_2)
        o0_1, o1_1 = max0 - z0_1, min1 - z0_1
        o0_2, o1_2 = max0 - z0_2, min1 - z0_2
        r0_1, r1_1, r0_2, r1_2 = [i.float()/self.stride
                                  for i in [o0_1, o1_1, o0_2, o1_2]]
        return torch.cat((r0_1, r1_1), 1), torch.cat((r0_2, r1_2), 1)

    def align(self, x, crop, flip):
        # flip align
        noflip_x = x * (~flip[:, None, None, None].expand_as(x))
        flip_x = hflip(x) * (flip[:, None, None, None].expand_as(x))
        x = noflip_x + flip_x
        # crop align
        idxs = torch.arange(crop.shape[0])[:, None].to(crop.device)
        crop = torch.cat((idxs, crop), 1)
        x = roi_align(x, crop, tuple([x.size(-2), x.size(-1)]))
        return x

    def forward(self, x1, x2):
        im1, crop1, flip1 = x1
        im2, crop2, flip2 = x2
        f, h = self.encoder, self.predictor
        z1, z2 = f(im1), f(im2)
        # TODO check correctness
        crop1, crop2 = self.crop_trans(crop1, crop2)
        a1, a2 = self.align(z1, crop1, flip1), self.align(z2, crop2, flip2)
        p1, p2 = h(a1), h(a2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return L


if __name__ == "__main__":
    model = PixSiam()
    x1 = torch.randn((2, 3, 224, 224))
    x2 = torch.randn_like(x1)

    model.forward(x1, x2).backward()
    print("forward backwork check")

    z1 = torch.randn((200, 2560))
    z2 = torch.randn_like(z1)
    import time
    tic = time.time()
    print(D(z1, z2, version='original'))
    toc = time.time()
    print(toc - tic)
    tic = time.time()
    print(D(z1, z2, version='simplified'))
    toc = time.time()
    print(toc - tic)














# from torchvision.models import resnet50
# from functools import partial

# def resnet_tuple_out_warper(m):
#     def new_forward(self, x):
#         out = []
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         out.append(x)
#         x = self.layer2(x)
#         out.append(x)
#         x = self.layer3(x)
#         out.append(x)
#         x = self.layer4(x)
#         out.append(x)

#         return tuple(out)

#     m._forward_impl = partial(new_forward, m)
#     return m


# if __name__ == "__main__":
#     m1 = resnet50()
#     print(m1(torch.rand(1,3,224,224)).shape)
#     m2 = resnet_tuple_out_warper(m1)
#     print([i.shape for i in m2(torch.rand(1,3,224,224))])