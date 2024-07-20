import torch
from loguru import logger
import sys
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
hidden2_size = 1536  # 第二隐藏层大小，比第一个隐藏层稍大
num_layers = 4  # 编码器层数
lr = 0.01  # 学习率
num_heads = 4
epochs = 5  # 训练轮数
dropout = 0.8 # dropout率
print_every = 1  # 每训练多少轮打印一次信息
save_every = 3  # 每训练多少轮保存一次
max_length = 30  # 序列的最大长度
optimer = "RMSprop"

input_id = 1
output_id = 0



logger.remove()

logger.add(
    sys.stdout,
    level="TRACE",
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "  # 绿色的时间戳
        "<level>{level}</level> "  # 根据日志级别着色
        "<level>{message}</level>"  # 根据日志级别着色的日志消息
    )
)

logger.debug('使用的设备：'+str(device))
logger.debug(f'训练轮数：{epochs}')
logger.debug(f'LSTM层数：{num_layers *2}')
logger.debug(f'注意力头数：{num_heads}')