import torch

# 是否编译模型的标志
compile_model = False

# 是否强制使用CPU
force_cpu = False

# 选择设备：如果强制使用CPU则使用CPU，否则如果有CUDA可用则使用CUDA
device = torch.device(
    "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
)

# 定义模型参数
hidden_size = 1024  # 隐藏层大小
hidden2_size = int(hidden_size *1.125 ) # 第二隐藏层大小，比第一个隐藏层稍大
encoder_layers = 3  # 编码器层数
decoder_layers = 3  # 解码器层数
lr = 0.01  # 学习率
epochs = 75  # 训练轮数
print_every = 1  # 每训练多少轮打印一次信息
plot_every = 1  # 每训练多少轮绘图一次
max_length = 40  # 序列的最大长度
optimer = 'SGD'

input_id = 1
output_id = 0

# 输出设备信息以确认
print(f"Using device: {device}")
