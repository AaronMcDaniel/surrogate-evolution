import timm
import torch
import torch.nn as nn


class TrEstimator(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        base_model_name = cfg['base_model_name']
        input_depth = 2
        self.weight_scale = cfg['weight_scale']

        self.base_model = timm.create_model(base_model_name,
                                            features_only=True,
                                            in_chans=input_depth,
                                            pretrained=pretrained)

        out_ch = self.base_model.feature_info.channels()[-1]
        self.conv_heatmap = nn.Conv2d(out_ch, 1, kernel_size=1, bias=True)
        self.conv_offset = nn.Conv2d(out_ch, 2, kernel_size=1, bias=True)

    def freeze_encoder(self):
        self.base_model.freeze_encoder()

    def unfreeze_encoder(self):
        self.base_model.unfreeze_encoder()

    def forward(self, prev_frame, cur_frame):
        inputs = torch.stack([prev_frame, cur_frame], dim=1)
        x = self.base_model(inputs)
        x = x[-1]

        x = x[:, :, 2:-2, 2:-2]

        x_hm = self.conv_heatmap(x)
        m = torch.exp(self.weight_scale * torch.sigmoid(x_hm))
        heatmap = m / torch.sum(m, dim=(2, 3), keepdim=True)
        offsets = self.conv_offset(x)

        return heatmap, offsets


def tsm(tensor, duration, dilation=1):
    # tensor [N*T, C, H, W]
    size = tensor.size()
    tensor = tensor.view((-1, duration) + size[1:])
    # tensor [N, T, C, H, W]
    shift_size = size[1] // 8
    mix_tensor, peri_tensor = tensor.split([shift_size, 7 * shift_size], dim=2)
    mix_tensor = mix_tensor[:, (1, 0), :, :]
    return torch.cat((mix_tensor, peri_tensor), dim=2).view(size)


def add_tsm_to_module(obj, duration, dilation=1):
    orig_forward = obj.forward

    def updated_forward(*args, **kwargs):
        a = (tsm(args[0], duration=duration, dilation=dilation), ) + args[1:]
        return orig_forward(*a, **kwargs)

    obj.forward = updated_forward

    return obj


class TrEstimatorTsm(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        base_model_name = cfg['base_model_name']
        input_depth = 1
        self.weight_scale = cfg['weight_scale']

        self.base_model = timm.create_model(base_model_name,
                                            features_only=True,
                                            in_chans=input_depth,
                                            pretrained=pretrained)
        duration = 2
        if 'resnet' in base_model_name:
            for l in self.base_model.layer1:
                add_tsm_to_module(l.conv1, duration)

            for l in self.base_model.layer2:
                add_tsm_to_module(l.conv1, duration)

            for l in self.base_model.layer3:
                add_tsm_to_module(l.conv1, duration)

            for l in self.base_model.layer4:
                add_tsm_to_module(l.conv1, duration)

        out_ch = self.base_model.feature_info.channels()[-1]
        self.conv_heatmap = nn.Conv2d(out_ch*2, 1, kernel_size=1, bias=True)
        self.conv_offset = nn.Conv2d(out_ch*2, 2, kernel_size=1, bias=True)


    def forward(self, prev_frame, cur_frame):
        inputs = torch.stack([prev_frame, cur_frame], dim=1)[:, :, None, :, :]  # BxTxCxHxW
        B, T, C, H, W = inputs.shape
        x = inputs.view(B*T, C, H, W)
        x = self.base_model(x)
        x = x[-1]

        # x = inputs.view(B, T, x.shape[1], x.shape[2], x.shape[3])
        # x = x[:, 1]  # select the cur frame output
        x = x.view(B, -1, x.shape[2], x.shape[3])

        x = x[:, :, 2:-2, 2:-2]

        x_hm = self.conv_heatmap(x)
        m = torch.exp(self.weight_scale * torch.sigmoid(x_hm))
        heatmap = m / torch.sum(m, dim=(2, 3), keepdim=True)
        offsets = self.conv_offset(x)

        return heatmap, offsets


def print_summary():
    import pytorch_model_summary
    model = TrEstimatorTsm(cfg={'base_model_name': 'resnet18', 'weight_scale': 2})
    heatmap, offsets = model(torch.zeros((2, 1024, 1024)), torch.zeros((2, 1024, 1024)))
    print(heatmap.shape, offsets.shape)

    print(pytorch_model_summary.summary(model, torch.zeros((2, 1024, 1024)), torch.zeros((2, 1024, 1024)),
                                        max_depth=2, show_hierarchical=False))


if __name__ == "__main__":
    print_summary()
