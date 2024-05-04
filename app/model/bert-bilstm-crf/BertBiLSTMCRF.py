import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from torchcrf import CRF


class BertBiLSTMCRF(nn.Module):
    def __init__(self, bert_model, num_tags, lstm_hidden_size, lstm_layers=1):
        super(BertBiLSTMCRF, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(0.2)
        self.weight = torch.tensor([0.2, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.5, 0.5]).to("cuda:0")
        # BiLSTM
        lstm_output_size = lstm_hidden_size * 2
        self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers,
                            bidirectional=True,
                            batch_first=True)
        self.classifier = nn.Linear(lstm_output_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)

        sequence_output = self.dropout(outputs[0])

        lstm_output, _ = self.lstm(sequence_output)

        logits = self.classifier(lstm_output)

        logits = logits * self.weight

        if labels is not None:
            loss = -self.crf(logits, labels, mask=attention_mask.bool(), reduction='mean')
            return loss
        else:
            predictions = self.crf.decode(logits, mask=attention_mask.bool())
            return predictions


def load_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    ner_model = BertBiLSTMCRF('bert-base-uncased', num_tags=9, lstm_hidden_size=768).to("cuda:0")
    model_path = r'C:\Users\AI\PycharmProjects\backend\app\model\bert-bilstm-crf\best_model.pth'
    ner_model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda:0')))

    ner_model.eval()  # 设置为评估模式
    return ner_model, tokenizer