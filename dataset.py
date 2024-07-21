import torch
import json
import os
import config
from config import logger
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

SOS_token = 0
EOS_token = 1
UNK_token = 2


class Lang:
    def __init__(self, name):
        self.name = name
        self.char2index = {}
        self.index2char = {
            SOS_token: "SOS",
            EOS_token: "EOS",
            UNK_token: "UNK",
        }
        self.n_chars = 3  # 初始化包含SOS、EOS和UNK字符

    def add_sentence(self, sentence):
        """将句子中的每个字符添加到词典中"""
        for char in sentence:
            self.add_char(char)

    def add_char(self, char):
        """将字符添加到词典中"""
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.index2char[self.n_chars] = char
            self.n_chars += 1

    def dump(self, fp):
        """将语言对象保存到文件"""
        data = {
            "name": self.name,
            "n_chars": self.n_chars,
            "char2index": self.char2index,
            "index2char": self.index2char,
        }
        json.dump(data, fp, ensure_ascii=False)

    @classmethod
    def load(cls, fp):
        """从文件加载语言对象"""
        data = json.load(fp)
        lang = cls(data["name"])
        lang.char2index = data["char2index"]
        lang.index2char = data["index2char"]
        lang.n_chars = data["n_chars"]
        return lang


class Seq2SeqDataset(Dataset):
    def __init__(self, path, input_id, output_id, max_len):
        self.path = path

        self.input_lang = Lang("input")
        self.output_lang = Lang("output")

        self.input_id = input_id
        self.output_id = output_id
        self.max_len = max_len  # 添加max_len参数

        self.file_list = []
        self.size = 0

        self.file_data_indices = []
        self.current_index = 0
        for file_name in os.listdir(self.path):
            file_path = os.path.join(self.path, file_name)
            if os.path.isdir(file_path):
                continue

            with open(file_path, "r") as file:
                num_lines = sum(1 for line in file)
                self.file_data_indices.append(
                    (self.current_index, self.current_index + num_lines, file_path)
                )
                self.current_index += num_lines

                file.seek(0)  # 重置文件指针
                for line in file:
                    pair = line.strip().split(",")
                    self.input_lang.add_sentence(pair[self.input_id])
                    self.output_lang.add_sentence(pair[self.output_id])

    def __len__(self):
        """返回数据集的大小"""
        return self.file_data_indices[-1][1]

    def __getitem__(self, idx):
        """根据索引返回数据"""
        for start_idx, end_idx, file_path in self.file_data_indices:
            if start_idx <= idx < end_idx:
                line_idx = idx - start_idx
                with open(file_path, "r") as file:
                    for i, line in enumerate(file):
                        if i == line_idx:
                            sample = line.strip()
                            pair = sample.split(",")
                            input_tensor = tensor_from_sentence(
                                self.input_lang, pair[self.input_id], self.max_len
                            )
                            output_tensor = tensor_from_sentence(
                                self.output_lang, pair[self.output_id], self.max_len
                            )
                            return input_tensor, output_tensor


def indexes_from_sentence(lang, sentence, max_len):
    """将句子转换为索引列表，并进行截断或填充"""
    indexes = [lang.char2index.get(char, UNK_token) for char in sentence]
    indexes.append(EOS_token)
    if len(indexes) > max_len:
        indexes = indexes[:max_len]  # 截断
    else:
        indexes.extend([EOS_token] * (max_len - len(indexes)))  # 填充
    return indexes


def tensor_from_sentence(lang, sentence, max_len):
    """将句子转换为张量"""
    indexes = indexes_from_sentence(lang, sentence, max_len)
    return torch.tensor(indexes, dtype=torch.long).to(config.device)


def collate_fn(batch):
    """自定义的collate_fn函数，用于对数据进行填充"""
    inputs, outputs = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=EOS_token)
    outputs_padded = pad_sequence(outputs, batch_first=True, padding_value=EOS_token)
    return inputs_padded, outputs_padded


dataset = Seq2SeqDataset("dataset/", 1, 0, max_len=20)  # 指定最大长度max_len
dataloader = DataLoader(
    dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
)

logger.info(f"输入语言字符数: {dataset.input_lang.n_chars}")
logger.info(f"输出语言字符数: {dataset.output_lang.n_chars}")
logger.info(f"数据量：{len(dataset)}")
