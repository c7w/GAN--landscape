import jittor as jt
import jittor.nn as nn

class SegFormerHeadMLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super(SegFormerHeadMLP, self).__init__()
        self.fc = nn.Linear(input_dim, embed_dim)

    def execute(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.fc(x)
        return x

class SegFormerHead(nn.Module):
    def __init__(self, in_channels, embedding_dim, hidden_dim, num_classes,
                 feature_strides, **kwargs):
        super(SegFormerHead, self).__init__(**kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]

        self.in_channels = in_channels
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c1 = nn.Linear(c1_in_channels, embedding_dim)
        self.linear_c2 = nn.Linear(c2_in_channels, embedding_dim)
        self.linear_c3 = nn.Linear(c3_in_channels, embedding_dim)
        self.linear_c4 = nn.Linear(c4_in_channels, embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm(hidden_dim),
            nn.ReLU(),
        )

        self.linear_pred = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def execute(self, inputs):
        # x = self._transform_inputs(inputs)  # len=4, 1/4, 1/8, 1/16, 1/32
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = jt.nn.interpolate(_c4, size=c1.size()[2:])

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = jt.nn.interpolate(_c3, size=c1.size()[2:])

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = jt.nn.interpolate(_c2, size=c1.size()[2:])

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(jt.concat([_c4, _c3, _c2, _c1], dim=1))
        x = self.linear_pred(_c)
        return x
