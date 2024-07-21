# pylint: disable=W0622
import torch
import torch.optim as optim
import torch.nn as nn
import os
import time
import json
from torch.utils.tensorboard import SummaryWriter

from model import EncoderRNN, DecoderRNN, Bangboo
import dataset
import config
from config import logger

# 创建模型实例
encoder = EncoderRNN(
    dataset.dataset.input_lang.n_chars,
    config.embed_dim,
    config.hidden_dim,
    config.hidden_dim2,
    config.num_layers,
    config.dropout,
    config.num_heads,
).to(config.device)

decoder = DecoderRNN(
    config.hidden_dim2,
    config.embed_dim,
    config.hidden_dim3,
    dataset.dataset.output_lang.n_chars,
    config.num_layers,
    config.dropout,
    config.num_heads,
).to(config.device)

model = Bangboo(encoder, decoder, config.device).to(config.device)
# 创建优化器
optimizer = getattr(optim, config.optimer)
optimizer = optimizer(model.parameters(), lr=config.lr)

# 创建损失函数
criterion = nn.CrossEntropyLoss()

if config.use_tensorboard:       
    writer = SummaryWriter(
        config.tensorboard_path,
        comment=config.tensorboard_comment
    )
global_step = 0

def train_epoch(model, dataloader, criterion, optimizer, epoch):
    global global_step
    model.train()
    total_loss = 0
    for i, (src, trg) in enumerate(dataloader):
        src.to(config.device)
        trg.to(config.device)
        
        optimizer.zero_grad()
        
        output = model(src, trg, config.teacher_forcing_ratio)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)

        optimizer.step()

        total_loss += loss
        global_step += 1
        if config.use_tensorboard:
            writer.add_scalar("Loss/train-Setp", loss, global_step)

        if (((epoch - 1) * len(dataloader) + i) % config.print_every ) == 0:
            logger.info(f"Epoch: {epoch}/{config.epochs} Step: {i+1} Loss: {loss :.8f}")
        print(global_step)
    return total_loss / len(dataloader)


def train(epochs, model, dataloader, criterion, optimizer):
    losses = []
    for epoch in range(epochs):
        loss = train_epoch(model, dataloader, criterion, optimizer, epoch)
        losses.append(loss)
        
    if config.use_tensorboard:
        writer.close()
    
    return losses


def count_parameters(model):
    """计算模型参数的数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_para():
    """获取编码器和解码器的参数数量"""
    params = count_parameters(model)

    logger.debug(f"参数量: {params}")
    logger.debug(f"参数量：{params // 1000000}M")
    return params


def save(model_dir):
    os.makedirs(model_dir, exist_ok=True)
    """保存模型和相关信息"""

    torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))
    
    dataset.dataset.input_lang.dump(
        open(os.path.join(model_dir, "input_vocab.json"), "w")
    )
    dataset.dataset.output_lang.dump(
        open(os.path.join(model_dir, "output_vocab.json"), "w")
    )


def create_modle():
    logger.info("开始训练")
    model_dir = os.path.join("models", str(int(time.time())))
    os.makedirs(os.path.join(model_dir, "checkpoint"))
    paras = get_para()

    metadata = {
        "parameters": {
            "total": paras,
            "M": (paras) // 1000000,
        },
        "epochs": config.epochs,
        "lr": config.lr,
        "teacher_forcing_ratio": config.teacher_forcing_ratio,
        "max_length": config.max_length,
        "clip": config.clip,
        "data_size": len(dataset.dataset),
        "batch_size": config.batch_size,
        "hidden_dim": config.hidden_dim,
        "hidden_dim2": config.hidden_dim2,
        "hidden_dim3": config.hidden_dim3,
        "embed_dim": config.embed_dim,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "dropout": config.dropout,
        "seed": torch.initial_seed(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "optimizer": config.optimer,  # 优化器名称
        "criterion": config.loss_func,  # 损失器名称
        "device": str(config.device),
        "compile": config.compile_model,
    }
    losses = train(config.epochs, model, dataset.dataloader, criterion, optimizer)
    #metadata["loss"] = losses

    json.dump(
        metadata,
        open(os.path.join(model_dir, "metadata.json"), "w"),
        indent=4,
        ensure_ascii=False,
    )

    model.eval()
    save(model_dir)
    logger.success(f"模型保存至：{model_dir}")


create_modle()
