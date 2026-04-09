"""
Self-contained TransNAS-Bench-101 network builder.
Combines cell_ops.py, cell_micro.py, net_macro.py into one file
to avoid the original project's complex dependency chain.
"""
import torch
import torch.nn as nn


# ===================== Cell Operations =====================

OPS = {
    '0': lambda C_in, C_out, stride, affine, track_running_stats: Zero(C_in, C_out, stride),
    '1': lambda C_in, C_out, stride, affine, track_running_stats: Identity() if (
                stride == 1 and C_in == C_out) else FactorizedReduce(C_in, C_out, stride, affine, track_running_stats),
    '2': lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, (1, 1), stride, (0, 0),
                                                                             (1, 1), affine, track_running_stats),
    '3': lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, (3, 3), stride, (1, 1),
                                                                             (1, 1), affine, track_running_stats)
}


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine, track_running_stats,
                 activation='relu'):
        super(ReLUConvBN, self).__init__()
        if activation == 'leaky':
            ops = [nn.LeakyReLU(0.2, False)]
        elif activation == 'relu':
            ops = [nn.ReLU(inplace=False)]
        else:
            raise ValueError(f"invalid activation {activation}")
        ops += [nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
                nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats)]
        self.ops = nn.Sequential(*ops)
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride

    def forward(self, x):
        return self.ops(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super(Zero, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x):
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.)
            else:
                return x[:, :, ::self.stride, ::self.stride].mul(0.)
        else:
            shape = list(x.shape)
            shape[1], shape[2], shape[3] = self.C_out, (shape[2] + 1) // self.stride, (shape[3] + 1) // self.stride
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, stride, affine, track_running_stats):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        assert C_out % 2 == 0, 'C_out : {:}'.format(C_out)
        C_outs = [C_out // 2, C_out - C_out // 2]
        self.convs = nn.ModuleList()
        for i in range(2):
            self.convs.append(nn.Conv2d(C_in, C_outs[i], 1, stride=stride, padding=0, bias=False))
        self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        self.bn = nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats)

    def forward(self, x):
        x = self.relu(x)
        y = self.pad(x)
        out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


# ===================== Micro Cell =====================

class MicroCell(nn.Module):
    expansion = 1

    def __init__(self, cell_code, C_in, C_out, stride, affine=True, track_running_stats=True):
        super(MicroCell, self).__init__()
        assert cell_code != 'basic'
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)

        self.node_num = len(cell_code)
        self.edges = nn.ModuleList()
        self.nodes = list(range(len(cell_code)))
        assert self.nodes == list(map(len, cell_code))
        self.from_nodes = [list(range(i)) for i in self.nodes]
        self.from_ops = [list(range(n * (n - 1) // 2, n * (n - 1) // 2 + n))
                         for n in range(self.node_num)]
        self.stride = stride

        for node in self.nodes:
            for op_idx, from_node in zip(cell_code[node], self.from_nodes[node]):
                if from_node == 0:
                    edge = OPS[op_idx](C_in, C_out, self.stride, affine, track_running_stats)
                else:
                    edge = OPS[op_idx](C_out, C_out, 1, affine, track_running_stats)
                self.edges.append(edge)

        self.cell_code = cell_code
        self.C_in = C_in
        self.C_out = C_out

    def forward(self, inputs):
        node_features = [inputs]
        for node_idx in self.nodes:
            if node_idx == 0:
                continue
            node_feature_list = [self.edges[from_op](node_features[from_node]) for from_op, from_node in
                                 zip(self.from_ops[node_idx], self.from_nodes[node_idx])]
            node_feature = torch.stack(node_feature_list).sum(0)
            node_features.append(node_feature)
        return node_features[-1]


# ===================== ResNet Basic Block =====================

class ResNetBasicblock(nn.Module):
    expansion = 1

    def __init__(self, cell_code, inplanes, planes, stride=1, affine=True, track_running_stats=True, activation='relu'):
        super(ResNetBasicblock, self).__init__()
        assert cell_code == 'basic'
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_a = ReLUConvBN(inplanes, planes, 3, stride, 1, 1, affine, track_running_stats, activation=activation)
        self.conv_b = ReLUConvBN(planes, planes, 3, 1, 1, 1, affine, track_running_stats, activation=activation)
        self.downsample = None
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes * self.expansion, affine, track_running_stats),
            )

    def forward(self, inputs):
        feature = self.conv_a(inputs)
        feature = self.conv_b(feature)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        return residual + feature


# ===================== MacroNet =====================

class MacroNet(nn.Module):
    def __init__(self, net_code, structure='full', input_dim=(224, 224), num_classes=75):
        super(MacroNet, self).__init__()
        assert structure in ['full', 'drop_last', 'backbone'], 'unknown structure: %s' % repr(structure)
        self.structure = structure
        self._read_net_code(net_code)
        self.inplanes = self.base_channel
        self.feature_dim = [input_dim[0] // 4, input_dim[1] // 4]

        self.stem = nn.Sequential(
            nn.Conv2d(3, self.base_channel // 2, kernel_size=3, stride=2, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(self.base_channel // 2, affine=True, track_running_stats=True),
            ReLUConvBN(self.base_channel // 2, self.base_channel, 3, 2, 1, 1, True, True)
        )

        self.layers = []
        for i, layer_type in enumerate(self.macro_code):
            layer_type = int(layer_type)
            target_channel = self.inplanes * 2 if layer_type % 2 == 0 else self.inplanes
            stride = 2 if layer_type > 2 else 1
            self.feature_dim = [self.feature_dim[0] // stride, self.feature_dim[1] // stride]
            layer = self._make_layer(self.cell, target_channel, 2, stride, True, True)
            self.add_module(f"layer{i}", layer)
            self.layers.append(f"layer{i}")

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) if structure in ['drop_last', 'full'] else None
        self.head = nn.Linear(self.inplanes, num_classes) if structure in ['full'] else None

        self._kaiming_init()

    def forward(self, x):
        x = self.stem(x)
        for i, layer_name in enumerate(self.layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
        if self.structure in ['full', 'drop_last']:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        if self.structure == 'full':
            x = self.head(x)
        return x

    def _make_layer(self, cell, planes, num_blocks, stride=1, affine=True, track_running_stats=True):
        layers = [cell(self.micro_code, self.inplanes, planes, stride, affine, track_running_stats)]
        self.inplanes = planes * cell.expansion
        for _ in range(1, num_blocks):
            layers.append(cell(self.micro_code, self.inplanes, planes, 1, affine, track_running_stats))
        return nn.Sequential(*layers)

    def _read_net_code(self, net_code):
        net_code_list = net_code.split('-')
        self.base_channel = int(net_code_list[0])
        self.macro_code = net_code_list[1]
        if net_code_list[-1] == 'basic':
            self.micro_code = 'basic'
            self.cell = ResNetBasicblock
        else:
            self.micro_code = [''] + net_code_list[2].split('_')
            self.cell = MicroCell

    def _kaiming_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
