from abc import ABC

import torch.nn as nn
import torch
from transformers import RobertaModel, RobertaConfig
from torch_geometric.nn.conv import GATConv


config = RobertaConfig.from_pretrained('roberta-base')


class GATConvWithAttention(GATConv, ABC):
    def forward(self, x, edge_index, edge_attr=None, size=None, return_attention_weights=True):
        # change GAT，input node attention weight
        out, attention_weights = super().forward(x, edge_index, edge_attr, size, return_attention_weights)
        return out, attention_weights


class RobertaPartialEncoder(nn.Module):
    def __init__(self, roberta_model_name, start_layer, end_layer):
        super().__init__()
        # split roberta
        self.roberta = RobertaModel.from_pretrained(roberta_model_name, config=config)
        self.encoder_layers = self.roberta.encoder.layer[start_layer:end_layer]

    def forward(self, hidden_states, attention_mask):
        # change mask weigth
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=hidden_states.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # conpute layer value
        for layer in self.encoder_layers:
            layer_outputs = layer(hidden_states, attention_mask=extended_attention_mask)
            hidden_states = layer_outputs[0]

        return hidden_states


class RobertaGAT(nn.Module):
    def __init__(self, roberta_model_name, num_classes):
        super(RobertaGAT, self).__init__()
        # load robert
        self.roberta = RobertaModel.from_pretrained(roberta_model_name, config=config)
        # roberta分层
        self.roberta_embeddings = self.roberta.embeddings
        self.roberta_first_half = RobertaPartialEncoder('roberta-base',
                                                        0, 6)
        self.roberta_second_half = RobertaPartialEncoder('roberta-base', 6, 12)
        # load GAT
        self.gat = GATConvWithAttention(self.roberta.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, edge_index):
        outputs = self.roberta_embeddings(input_ids=input_ids)

        outputs = self.roberta_first_half(outputs, attention_mask=attention_mask)

        outputs = outputs
        attention_mask = attention_mask
        outputs = self.roberta_second_half(outputs, attention_mask=attention_mask)

        sentence_embeddings = outputs[:, 0, :]

        gat_output, attention_weights = self.gat(sentence_embeddings, edge_index)
        return gat_output, attention_weights


def load_model():
    model = RobertaGAT(roberta_model_name="roberta-base", num_classes=5)
    model.load_state_dict(
        torch.load(r'C:\Users\AI\IdeaProjects\dlut-research-service\src\main\flask\model\RoBERTaGAT\model.pth',
                   map_location='cuda:0'), strict=False)
    model.eval()
    return model