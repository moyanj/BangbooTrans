from model import EncoderRNN, DecoderRNN
import dataset
import os
import torch
import json

class Inference:
    def __init__(self, model_name,model_base='Models'):
        model_path = os.path.join(model_base,model_name)
        self.metadata = json.load(open(os.path.join(model_path, 'metadata.json')))
        
        self.input_lang = dataset.Lang.load(open(os.path.join(model_path,'input_vocab.json')))
        self.output_lang = dataset.Lang.load(open(os.path.join(model_path,'output_vocab.json')))
        
        self.encoder = EncoderRNN(self.input_lang.n_chars, self.metadata['hidden_size'], self.metadata['hidden_size'], self.metadata['num_layers'])
        self.encoder.load_state_dict(torch.load(os.path.join(model_path,'encoder.pth')))
        
        self.decoder = DecoderRNN(self.metadata['hidden_size'], self.output_lang.n_chars, self.metadata['hidden2_size'], self.metadata['num_layers'])
        self.decoder.load_state_dict(torch.load(os.path.join(model_path,'decoder.pth')))
        
infer = Inference("onnxTest")        