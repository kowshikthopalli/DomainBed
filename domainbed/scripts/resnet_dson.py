import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from dson import DomainSpecificOptimizedNorm2d as DSON2d
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class DownSample(nn.Module):

    def __init__(self, inplanes, planes, stride=1,kernel_size=1,num_domains=3,bias=False): 
        super(DownSample, self).__init__()   
        self.conv = nn.Conv2d(inplanes, planes,
                            kernel_size=kernel_size, stride=stride, bias=bias)
        self.bn   =  DSON2d(planes,num_domains)
    def forward(self, x,domain_label):
        out = self.conv(x)
        out,l= self.bn(out,domain_label)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,num_domains=3):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = DSON2d(planes,num_domains)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = DSON2d(planes,num_domains)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x,domain_label):
        # f,domain_label =x[0],x[1]
        # x=f
        residual = x

        out = self.conv1(x)
        out,l = self.bn1(out,domain_label)
        out = self.relu(out)

        out = self.conv2(out)
        out,l = self.bn2(out,domain_label)

        if self.downsample is not None:
            residual = self.downsample(x,domain_label)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,num_domains=3):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = DSON2d(planes,num_domains)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = DSON2d(planes,num_domains)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = DSON2d(planes * 4,num_domains)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x,domain_label):
        residual = x

        out = self.conv1(x)
        out,l = self.bn1(out,domain_label)
        out = self.relu(out)

        out = self.conv2(out)
        out,l = self.bn2(out,domain_label)
        out = self.relu(out)

        out = self.conv3(out)
        out,l = self.bn3(out,domain_label)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, input_channels=3,num_domains=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.num_domains = num_domains
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = DSON2d(64,self.num_domains)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=2)

        self.dropout = nn.Dropout2d(p=0.5,inplace=True)

        #print "block.expansion=",block.expansion
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            # elif isinstance(m, DSON2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownSample(self.inplanes, planes * block.expansion,kernel_size=1,\
                 stride=stride,num_domains=self.num_domains,bias=False)
            # nn.Sequential(
            #     nn.Conv2d(self.inplanes, planes * block.expansion,
            #               kernel_size=1, stride=stride, bias=False),
            #     DSON2d(planes * block.expansion,self.num_domains),
            # )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,num_domains=self.num_domains))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers # module list

    def forward(self,x,domain_label):
        self.domain_label=domain_label
        x = self.conv1(x)
        x,label = self.bn1(x,domain_label)
        x = self.relu(x)
        x = self.maxpool(x)
        for l in self.layer1:
            x= l(x,domain_label)
        for l in self.layer2:
            x= l(x,domain_label)
        for l in self.layer3:
            x= l(x,domain_label)
        for l in self.layer4:
            x= l(x,domain_label)


        # x = self.layer1(x) # list comprehension
        # x = self.layer2((x,domain_label))
        # x = self.layer3((x,domain_label))
        # x = self.layer4((x,domain_label))

        x = self.avgpool(x)
        x = self.dropout(x)
        #print "avepool: ",x.data.shape
        x = x.view(x.size(0), -1)
        #print "view: ",x.data.shape
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']),strict=False)
    return model



def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']),strict=False)
    return model

if __name__ == "__main__":
    resnet1 = resnet18(True,input_channels=3,num_domains=3)
    inp = torch.randn(10,3,224,224)
    resnet1= resnet1.eval()
    with torch.no_grad():

        print(resnet1(inp,1).size())

    #oprn a pickle file with
