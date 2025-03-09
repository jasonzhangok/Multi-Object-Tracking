import torch
import torch.nn as nn
import time

def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)

# 定义一个简单的 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # 输入通道 3，输出通道 16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 输入通道 16，输出通道 32
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # 假设输入图像大小为 32x32
        self.fc2 = nn.Linear(128, 10)  # 10 个类别

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)  # 16x16
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)  # 8x8
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def prune_channels(model, prune_ratio):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):  # 只剪枝卷积层
            weight = module.weight.data
            out_channels = weight.shape[0]  # 输出通道数
            prune_channels = int(out_channels * prune_ratio)  # 计算需要剪枝的通道数

            # 计算通道重要性（使用 L1 范数）
            importance = weight.abs().sum(dim=(1, 2, 3))  # 计算每个通道的 L1 范数
            sorted_idx = importance.argsort()  # 按重要性排序
            prune_idx = sorted_idx[:prune_channels]  # 选择最不重要的通道

            # 移除不重要的通道
            mask = torch.ones(out_channels, dtype=bool)
            mask[prune_idx] = False  # 标记需要剪枝的通道
            module.weight.data = module.weight.data[mask]  # 剪枝权重
            if module.bias is not None:
                module.bias.data = module.bias.data[mask]  # 剪枝偏置

            # 更新下一层的输入通道数
            if isinstance(module, nn.Conv2d):
                next_convs = []
                for _, next_module in model.named_modules():
                    if isinstance(next_module, nn.Conv2d) and next_module.in_channels == out_channels:
                        next_convs.append(next_module)
                if next_convs:
                    for next_conv in next_convs:
                        # 确保下一层的输入通道数与剪枝后的通道数一致
                        if next_conv.weight.data.shape[1] == out_channels:
                            next_conv.weight.data = next_conv.weight.data[:, mask]  # 更新下一层的输入通道
                        else:
                            print(f"Skipping {name}: next layer input channels do not match.")
                else:
                    print(f"Skipping {name}: no next conv layer found.")

    return model

def update_layer(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 获取卷积层的输出通道数
            out_channels = module.weight.data.shape[0]
            # 假设输入图像大小为 32x32，经过两次池化后大小为 8x8
            fc_input_features = out_channels * 8 * 8
            # 更新全连接层的输入特征数
            if hasattr(model, 'fc1'):
                model.fc1 = nn.Linear(fc_input_features, 128)
        if isinstance(module, nn.Conv2d):  # 找到卷积层
            out_channels = module.weight.data.shape[0]  # 剪枝后的输出通道数
            # 找到对应的 BN 层
            for bn_name, bn_module in model.named_modules():
                if isinstance(bn_module, nn.BatchNorm2d) and bn_name == name.replace("conv", "bn"):
                    # 更新 BN 层的 running_mean 和 running_var
                    bn_module.running_mean = bn_module.running_mean[:out_channels]
                    bn_module.running_var = bn_module.running_var[:out_channels]
                    # 更新 BN 层的权重和偏置
                    bn_module.weight.data = bn_module.weight.data[:out_channels]
                    bn_module.bias.data = bn_module.bias.data[:out_channels]

    return model

# 初始化模型
model = SimpleCNN()
input_tensor = torch.randn(100, 3, 32, 32)  # 假设输入图像大小为 32x32

# 测试剪枝前的推理时间
start_time = time.time()
with torch.no_grad():
    _ = model(input_tensor)
end_time = time.time()
print(f"剪枝前的推理时间: {end_time - start_time:.6f} 秒")
model_structure(model)
# 定义剪枝比例
prune_ratio = 0.2  # 剪枝 20% 的通道

# 执行剪枝
pruned_model = prune_channels(model, prune_ratio)
pruned_model = update_layer(pruned_model)
# 测试剪枝后的推理时间
start_time = time.time()
with torch.no_grad():
    _ = pruned_model(input_tensor)
end_time = time.time()
print(f"剪枝后的推理时间: {end_time - start_time:.6f} 秒")
model_structure(pruned_model)