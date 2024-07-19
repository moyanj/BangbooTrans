# pylint: disable=W0622
import torch
import torch.optim as optim
import os
import time
import json
import random

from model import EncoderRNN, DecoderRNN, criterion
import dataset
import config
from config import logger

# 创建模型实例
encoder = EncoderRNN(
    dataset.input_lang.n_chars,
    config.hidden_size,
    config.hidden2_size,
    config.encoder_layers,
    config.dropout,
    config.device,
).to(config.device)
encoder.train()

decoder = DecoderRNN(
    config.hidden_size,
    dataset.output_lang.n_chars,
    config.hidden2_size,
    config.decoder_layers,
    config.dropout,
    config.device,
).to(config.device)
decoder.train()


# 如果需要编译模型
if config.compile_model:
    encoder = torch.compile(encoder)
    decoder = torch.compile(decoder)

# 创建优化器
optimizer = getattr(optim, config.optimer)
encoder_optimizer = optimizer(encoder.parameters(), lr=config.lr)
decoder_optimizer = optimizer(decoder.parameters(), lr=config.lr)


def _train(
    input_tensor,
    target_tensor,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
):
    # 初始化隐藏状态
    encoder_hidden = encoder.init_hidden()

    # 清空梯度
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # 获取输入和目标序列的长度
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # 初始化编码器的输出
    encoder_outputs = torch.zeros(
        config.max_length, encoder.hidden_size, device=config.device
    )

    loss = 0  # 初始化损失

    # 编码输入序列
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        if ei < config.max_length:  # 防止越界
            encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor(
        [[dataset.SOS_token]], device=config.device
    )  # 解码器的初始输入为SOS_token
    decoder_hidden = encoder_hidden  # 将编码器的隐藏状态传递给解码器

    # 解码目标序列
    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)  # 获取概率最高的词
        decoder_input = topi.squeeze().detach()  # 不回传梯度

        # 计算损失
        loss += criterion(decoder_output, target_tensor[di])

        # 如果解码器输出EOS_token，则提前停止解码
        if decoder_input.item() == dataset.EOS_token:
            break

    loss.backward()  # 反向传播计算梯度

    # 更新模型参数
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length  # 返回平均损失

def train(n_iters, print_every, save_every, model_dir):
    total_loss = 0
    losses = []
    for iter in range(1, n_iters + 1):
        # Shuffle dataset pairs at the beginning of each epoch
        if iter % len(dataset.pairs) == 1:
            random.shuffle(dataset.pairs)

        training_pair = dataset.pairs[(iter - 1) % len(dataset.pairs)]
        input_tensor = dataset.tensor_from_sentence(
            dataset.input_lang, training_pair[config.input_id]
        )
        target_tensor = dataset.tensor_from_sentence(
            dataset.output_lang, training_pair[config.output_id]
        )

        loss = _train(
            input_tensor,
            target_tensor,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            criterion,
        )
        total_loss += loss
        losses.append(loss)
        if iter % print_every == 0:
            logger.info(f"({iter / n_iters * 100:.2f}%) Epoch: {iter} Loss: {total_loss / print_every:.10f}")
            total_loss = 0
            
        if iter % save_every == 0:
            save(os.path.join(model_dir,str(iter)))

    return losses


def count_parameters(model):
    """计算模型参数的数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_para():
    """获取编码器和解码器的参数数量"""
    encoder_params = count_parameters(encoder)
    decoder_params = count_parameters(decoder)
    total_params = encoder_params + decoder_params

    logger.debug(f"Encoder 参数量: {encoder_params}")
    logger.debug(f"Decoder 参数量: {decoder_params}")
    logger.debug(f"总参数量: {total_params}")
    return encoder_params, decoder_params

def save(model_dir):
    os.makedirs(model_dir,exist_ok=True)
    """保存模型和相关信息"""
    torch.save(encoder.state_dict(), os.path.join(model_dir, "encoder.pth"))
    torch.save(decoder.state_dict(), os.path.join(model_dir, "decoder.pth"))
    torch.save(
        encoder_optimizer.state_dict(), os.path.join(model_dir, "encoder_optimizer.pth")
    )
    torch.save(
        decoder_optimizer.state_dict(), os.path.join(model_dir, "decoder_optimizer.pth")
    )

    dataset.input_lang.dump(open(os.path.join(model_dir, "input_vocab.json"), "w"))
    dataset.output_lang.dump(open(os.path.join(model_dir, "output_vocab.json"), "w"))

def create_modle():
    logger.info('开始训练')
    model_dir = os.path.join("models", str(int(time.time())))
    os.makedirs(os.path.join(model_dir,'checkpoint'))
    e_para, d_para = get_para()

    metadata = {
        "parameters": {
            "decoder": d_para,
            "encoder": e_para,
            "total": d_para + e_para,
            "M": (d_para + e_para) / 1024 // 1024,
        },
        "epochs": config.epochs,
        "lr": config.lr,
        "max_length": config.max_length,
        "data_size": len(dataset.pairs),
        "hidden_dim": config.hidden_size,
        "hidden_dim2": config.hidden2_size,
        "seed": torch.initial_seed(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "optimizer": type(encoder_optimizer).__name__,  # 优化器名称
        "criterion": type(criterion).__name__,  # 损失器名称
        "device": str(config.device),
        "compile": config.compile_model,
        "num_layers": config.encoder_layers,
        "dropout": config.dropout,
    }
    losses = train(config.epochs, config.print_every, config.save_every, os.path.join(model_dir,'checkpoint'))
    metadata['loss'] = losses
    json.dump(metadata, open(os.path.join(model_dir,'metadata.json'),'w'),indent=4,ensure_ascii=False)
    save(model_dir)
    logger.success(f'模型保存至：{model_dir}')

create_modle()