import json
import torch

SOS_token = 0
EOS_token = 1
UNK_token = 2


class Lang:
    def __init__(self, name, device):
        self.name = name
        self.device = device
        self.char2index = {}
        self.index2char = {SOS_token: "SOS", EOS_token: "EOS", UNK_token: "UNK"}
        self.n_chars = 3  # 初始化包含SOS、EOS和UNK字符

    @classmethod
    def load(cls, fp, device):
        data = json.load(fp)
        cla = cls(data["name"], device)
        cla.char2index = data["char2index"]
        cla.index2char = data["index2char"]
        cla.n_chars = data["n_chars"]
        return cla
