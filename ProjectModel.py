import torch.nn as nn
import torchvision


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Use VGG-16 feature-extractor for encoding
        self.extractor = torchvision.models.vgg16(pretrained=True).features[-1]

    def forward(self, data):
        return self.extractor(data)


class ASPPModule(nn.Module):
    def __init__(self, input_size: int, kernel_size: int, padding: int, dilation: int):
        super(ASPPModule, self).__init__()
        self.conv = nn.Conv2d(input_size, 512, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, data):
        data = self.conv(data)
        return self.relu(data)


class ASPP(nn.Module):
    def __init__(self):
        super().__init__()
        module1 = ASPPModule(3, 1, 0, 1)
        module2 = ASPPModule(512, 3, 6, 6)
        module3 = ASPPModule(512, 3, 12, 12)
        module4 = ASPPModule(512, 3, 18, 18)
        sequence = nn.Sequential(nn.Conv2d(512, 128, 1, stride=1, bias=False),
                                     nn.ReLU())
        self.model = nn.Sequential(module1, module2, module3, module4, sequence)

    def forward(self, data):
        return self.model(data)


class DecoderModule(nn.Module):
    def __init__(self, input_size: int, output_size: int, kernel_size: int):
        super(DecoderModule, self).__init__()
        # Kroner et al. state the importance of bilinear upsampling
        convolution = nn.Conv2d(input_size, output_size, kernel_size=kernel_size, padding=1)
        relu = nn.ReLU()
        self.model = nn.Sequential(convolution, relu)

    def forward(self, data):
        return self.model(data)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        upsampling = nn.Upsample(scale_factor=2)
        module1 = DecoderModule(128, 64, 3)
        module2 = DecoderModule(64, 32, 3)
        module3 = DecoderModule(32, 16, 3)
        output = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.model = nn.Sequential(upsampling, module1, module2, module3, output)

    def forward(self, data):
        return self.model(data)


class EncoderASPPDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        # Freeze parameters: We only want to train ASPP and our decoder
        self.encoder.requires_grad_(False)
        self.aspp = ASPP()
        self.decoder = Decoder()

    def forward(self, data):
        d = self.encoder(data)
        d = self.aspp(d)
        return self.decoder(d)
