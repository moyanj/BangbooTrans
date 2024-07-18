import torch
import json
import config

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
            "index2index": self.index2char,
        }
        json.dump(data, fp)

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


input_lang = Lang("邦布")
output_lang = Lang("中文")
pairs = []
    
dataset = open("邦布语数据集.csv").read().split("\n")
    
for line in dataset:
    pair = line.split(",")
    pairs.append(pair)
    input_lang.add_sentence(pair[config.input_id])
    output_lang.add_sentence(pair[config.output_id])

print("Input language: ", input_lang.name)
print("Number of characters: ", input_lang.n_chars)
print("Output language: ", output_lang.name)
print("Number of characters: ", output_lang.n_chars)
  