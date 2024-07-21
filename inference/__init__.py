#pylint:disable=W0612
#pylint:disable=E1102
from model import EncoderRNN, DecoderRNN, Bangboo
from . import dataset
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import json

class Inference:
    def __init__(
        self, model_name, model_base="models", compile_model=False, force_cpu=False
    ):
        model_path = os.path.join(model_base, model_name)
        self.model_path = model_path

        self.compile = compile_model

        self.device = torch.device(
            "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.metadata = json.load(open(os.path.join(model_path, "metadata.json")))

        self.input_lang = dataset.Lang.load(
            open(os.path.join(model_path, "input_vocab.json")), self.device
        )
        self.output_lang = dataset.Lang.load(
            open(os.path.join(model_path, "output_vocab.json")), self.device
        )

        self.encoder = EncoderRNN(
            self.input_lang.n_chars,
            self.metadata['embed_dim'],
            self.metadata["hidden_dim"],
            self.metadata["hidden_dim2"],
            self.metadata["num_layers"],
            self.metadata["dropout"],
            self.metadata["num_heads"],
        ).to(self.device)
        
        self.decoder = DecoderRNN(
            self.metadata["hidden_dim2"],
            self.metadata["embed_dim"],
            self.metadata["hidden_dim3"],
            self.output_lang.n_chars,
            self.metadata["num_layers"],
            self.metadata["dropout"],
            self.metadata['num_heads'],
        ).to(self.device)
        
        self.model = Bangboo(self.encoder, self.decoder, self.device).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(model_path, "model.pth"), map_location=self.device))
        
        if compile_model and hasattr(self.model, 'compile'):
            self.model.compile()

    def indexes_from_sentence(self, lang, sentence):
        """将句子转换为索引列表，并进行截断或填充"""
        max_len = self.metadata['max_length']
        indexes = [lang.char2index.get(char, dataset.UNK_token) for char in sentence]
        indexes.append(dataset.EOS_token)
        if len(indexes) > max_len:
            indexes = indexes[:max_len]  # 截断
        else:
            indexes.extend([dataset.EOS_token] * (max_len - len(indexes)))  # 填充
        return indexes

    def tensor_from_sentence(self, lang, sentence, batch_size=1):
        """将句子转换为张量，适合直接输入LSTM"""
        indexes = self.indexes_from_sentence(lang, sentence)
        # 将索引列表转换为张量，并增加维度 (seq_len, 1)
        tensor = torch.tensor(indexes, dtype=torch.long).to(self.device)#.unsqueeze(1)
        # 增加 batch_size 维度 (seq_len, batch_size)
        # tensor = tensor.repeat(1, batch_size)
        return tensor
        
    def eval(self, input_sentence):
        # 将句子转换为索引张量 (seq_len, batch_size)
        input_tensor = self.tensor_from_sentence(self.input_lang, input_sentence).to(self.device)
        self.model.predict(input_tensor,self.metadata['max_length'])
        return "".join(decoded_chars)