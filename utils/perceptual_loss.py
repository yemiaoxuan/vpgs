import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        # 使用 VGG19 模型
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).cuda()
        
        # 定义更多的特征块
        blocks = []
        # VGG19 的特征层分块
        blocks.extend([vgg.features[:4].eval()])     # 第一个块 conv1 (0-3)
        blocks.extend([vgg.features[4:9].eval()])    # 第二个块 conv2 (4-8)
        blocks.extend([vgg.features[9:18].eval()])   # 第三个块 conv3 (9-17)
        # blocks.extend([vgg.features[18:27].eval()])  # 第四个块 conv4 (18-26)
        # blocks.extend([vgg.features[27:36].eval()])  # 第五个块 conv5 (27-35)
        
        del vgg
        torch.cuda.empty_cache()
        
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
                
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, input, target, normalize=True):
        if normalize:
            input = (input - self.mean) / self.std
            target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
            
        loss = 0.0
        x = input
        y = target
        
        for block in self.blocks:
            with torch.cuda.amp.autocast(enabled=True):
                x = block(x)
                y = block(y)
                loss += torch.nn.functional.l1_loss(x, y)
                
            # 清理中间变量
            x = x.detach()
            y = y.detach()
            torch.cuda.empty_cache()
            
        return loss