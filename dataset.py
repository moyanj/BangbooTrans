import torch
import json
import os
import config
from config import logger

SOS_token = 0
EOS_token = 1
UNK_token = 2


class Lang:
    def __init__(self, name):
        self.name = name
        self.char2index = {}
        self.index2char = {SOS_token: "SOS", EOS_token: "EOS", UNK_token: "UNK"}
        self.n_chars = 3  # 初始化包含SOS、EOS和UNK字符

    def add_sentence(self, sentence):
        for char in sentence:
            self.add_char(char)

    def add_char(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.index2char[self.n_chars] = char
            self.n_chars += 1

    def dump(self, fp):
        data = {
            "name": self.name,
            "n_chars": self.n_chars,
            "char2index": self.char2index,
            "index2char": self.index2char,
        }
        json.dump(data, fp, ensure_ascii=False)

    @classmethod
    def load(cls, fp):
        data = json.load(fp)
        cla = cls(data["name"])
        cla.char2index = data["char2index"]
        cla.index2char = data["index2char"]
        cla.n_chars = data["n_chars"]
        return cla


def indexes_from_sentence(lang, sentence):
    return [lang.char2index.get(char, UNK_token) for char in sentence]


def tensor_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1).to(config.device)


def get_datas(path):
    logger.info('加载数据集中。。。')
    for pat in os.listdir(path):
        if ".csv" in pat:
            for line in open(os.path.join(path, pat)).read().split("\n"):
                yield line.split(",")


input_lang = Lang("Input")
output_lang = Lang("Output")
pairs = []

dataset = get_datas("dataset/")

for pair in dataset:
    pairs.append(pair)
    input_lang.add_sentence(pair[config.input_id])
    output_lang.add_sentence(pair[config.output_id])

logger.info(f"输入语言字符数: {input_lang.n_chars}")
logger.info(f"输出语言字符数: { output_lang.n_chars }")
logger.info(f'数据量：{len(pairs)}')