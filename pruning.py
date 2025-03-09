import torch
import platform
from pathlib import Path
from models.yolo import Model
import pathlib
from utils.torch_utils import profile  # 用于计算 FLOPs
import torch
import torch.nn.utils.prune as prune
import time

# 设置平台兼容性
plt = platform.system()
if plt != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

# 获取 YOLOv5 根目录
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory

def prune_weights(model, prune_ratio):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):  # 只剪枝卷积层
            weight = module.weight.data
            num_weights = weight.numel()  # 权重总数
            prune_weights = int(num_weights * prune_ratio)  # 计算需要剪枝的权重数量
            # 计算权重重要性（使用 L1 范数）
            importance = weight.abs()  # 计算每个权重的 L1 范数
            sorted_idx = importance.view(-1).argsort()  # 按重要性排序
            prune_idx = sorted_idx[:prune_weights]  # 选择最不重要的权重

            # 移除不重要的权重
            mask = torch.ones_like(weight)
            mask.view(-1)[prune_idx] = 0  # 标记需要剪枝的权重
            module.weight.data = weight * mask  # 剪枝权重

    return model

def model_structure(model):
    eps = 1e-5
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
        flatten_variable = w_variable.view(-1)
        zero_count = 0
        for i in flatten_variable:
            if abs(i - 0.0) < eps:
                zero_count = zero_count + 1
                i = 0
        each_para = each_para - zero_count
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)

# 使用稀疏计算
def sparse_forward(model, x):
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                # 将权重转换为稀疏张量
                sparse_weight = module.weight.to_sparse()
                x = torch.nn.functional.conv2d(x, sparse_weight, module.bias, module.stride, module.padding, module.dilation, module.groups)
            else:
                x = module(x)
    return x


def test_time(model,test_data,way):
    if(way == 0):
        # 计算推理时间
        start_time = time.time()
        with torch.no_grad():
            _ = model(test_data)  # 处理 100 个测试用例
        end_time = time.time()

        # 计算平均推理时间
        total_time = end_time - start_time
        print(f"总推理时间: {total_time:.4f} 秒")
        return total_time
    else:
        start_time = time.time()
        with torch.no_grad():
            _ = sparse_forward(model,test_data)  # 处理 100 个测试用例
        end_time = time.time()
        total_time = end_time - start_time
        print(f"总推理时间: {total_time:.4f} 秒")
        return total_time



if __name__ == "__main__":
    model = Model(ROOT / "models/yolov5s.yaml")  # 使用 YOLOv5s 配置文件
    model.eval()
    test_data = torch.randn(100, 3, 640, 640)

    # 加载训练好的权重
    checkpoint = torch.load(ROOT / "best.pt", map_location="mps")  # 加载权重文件
    model.load_state_dict(checkpoint["model"].state_dict())  # 加载模型权重
    # model_structure(model)

    # 计算剪枝前的推理时间
    start_time = time.time()
    with torch.no_grad():
        _ = model(test_data)  # 处理 100 个测试用例
    end_time = time.time()
    total_time = end_time - start_time
    print(f"剪枝前的总推理时间: {total_time:.4f} 秒")

    prune_ratio = 0.2

    # 执行剪枝
    pruned_model = prune_weights(model, prune_ratio)

    # 计算剪枝后的推理时间
    start_time = time.time()
    with torch.no_grad():
        _ = sparse_forward(pruned_model, test_data)  # 处理 100 个测试用例
    end_time = time.time()
    total_time_pruned = end_time - start_time
    print(f"剪枝后的总推理时间: {total_time_pruned:.4f} 秒")

    # 计算加速比
    speedup = total_time / total_time_pruned
    print(f"加速比: {speedup:.2f}x")


    # time1 = test_time(model, test_data,0)
    #
    # prune_ratio = 0.2
    #
    # # 执行剪枝
    # pruned_model = prune_weights(model, prune_ratio)
    # time2 = test_time(model, test_data,1)
    # print(time2 - time1)
    # # model_structure(model)


    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.Conv2d):  # 只剪枝卷积层
    #         prune.ln_structured(module, name="weight", amount=0.1, n=2, dim=0)
