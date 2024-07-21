from model import EncoderRNN, DecoderRNN
from . import dataset
import os
import torch
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
            self.metadata["hidden_dim"],
            self.metadata["hidden_dim2"],
            self.metadata["num_layers"],
            self.metadata["dropout"],
            self.metadata["num_heads"],
            self.device,
        )
        self.encoder.load_state_dict(
            torch.load(
                os.path.join(model_path, "encoder.pth"), map_location=self.device
            )
        )
        self.encoder.eval()

        self.decoder = DecoderRNN(
            self.metadata["hidden_dim"],
            self.output_lang.n_chars,
            self.metadata["hidden_dim2"],
            self.metadata["num_layers"],
            self.metadata["dropout"],
            self.device,
        )
        self.decoder.load_state_dict(
            torch.load(
                os.path.join(model_path, "decoder.pth"), map_location=self.device
            )
        )
        self.decoder.eval()

        if compile_model:
            self.encoder.compile()
            self.decoder.compile()

    def indexes_from_sentence(self, lang, sentence):
        return [lang.char2index.get(char, dataset.UNK_token) for char in sentence]

    def tensor_from_sentence(self, lang, sentence):
        indexes = self.indexes_from_sentence(lang, sentence)
        indexes.append(dataset.EOS_token)
        return torch.tensor(indexes, dtype=torch.long).view(-1, 1).to(self.device)

    def eval_steam(self, sentence, max_length=200):
        with torch.no_grad():
            input_tensor = self.tensor_from_sentence(self.input_lang, sentence)
            input_length = input_tensor.size()[0]

            encoder_hidden = self.encoder.init_hidden()

            encoder_outputs = torch.zeros(
                max_length, self.encoder.hidden_size, device=self.device
            )
            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(
                    input_tensor[ei], encoder_hidden
                )
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor(
                [[dataset.SOS_token]], device=self.device
            )  # SOS
            decoder_hidden = encoder_hidden
            decoded_chars = []
            for di in range(max_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden
                )
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == dataset.EOS_token:
                    decoded_chars.append("EOS")
                    break
                else:
                    decoded_char = self.output_lang.index2char.get(
                        str(topi.item()), "UNK"
                    )
                    decoded_chars.append(decoded_char)
                    yield decoded_char  # yield each character as it is decoded

                decoder_input = topi.squeeze().detach().unsqueeze(0)

            yield ""  # indicate end of sequence if streaming

    def eval(self, *args, **kwargs):
        return "".join(self.eval_steam(*args, **kwargs))
