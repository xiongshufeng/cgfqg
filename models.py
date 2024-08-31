import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchcrf import CRF

from pytorch_transformers import BertPreTrainedModel, BertModel
from pytorchcrf import CRF

from attentions import AdditiveAttention, ScaledDotProductAttention, LabelAttention


class BERT_BiLSTM_CRF(BertPreTrainedModel):

    def __init__(self, config, need_birnn=False, rnn_dim=128):
        super(BERT_BiLSTM_CRF, self).__init__(config)

        self.num_tags = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        out_dim = config.hidden_size
        self.need_birnn = need_birnn

        # 如果为False，则不要BiLSTM层
        if need_birnn:
            self.birnn = nn.LSTM(config.hidden_size, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            out_dim = rnn_dim * 2

        self.hidden2tag = nn.Linear(out_dim, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(self, input_ids, tags, token_type_ids=None, input_mask=None):
        emissions = self.tag_outputs(input_ids, token_type_ids, input_mask)
        loss = -1 * self.crf(emissions, tags, mask=input_mask.byte())

        return loss

    def tag_outputs(self, input_ids, token_type_ids=None, input_mask=None):

        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)

        sequence_output = outputs[0]

        if self.need_birnn:
            sequence_output, _ = self.birnn(sequence_output)

        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)

        return emissions

    def predict(self, input_ids, token_type_ids=None, input_mask=None):
        emissions = self.tag_outputs(input_ids, token_type_ids, input_mask)
        return self.crf.decode(emissions, input_mask.byte())


class BERT_BiLSTM_CRF_MUL(BertPreTrainedModel):

    def __init__(self, config, need_birnn=False, rnn_dim=128, num_labels2=0):
        super(BERT_BiLSTM_CRF_MUL, self).__init__(config)

        self.num_labels2 = num_labels2
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        out_dim = config.hidden_size
        self.need_birnn = need_birnn

        # 如果为False，则不要BiLSTM层
        if need_birnn:
            self.birnn = nn.LSTM(config.hidden_size, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            out_dim = rnn_dim * 2

        # self.attn = nn.MultiheadAttention(out_dim, num_heads=2)
        # self.attn = ScaledDotProductAttention(out_dim)
        self.attn = LabelAttention(out_dim, config.num_labels)
        self.hidden2tag1 = nn.Linear(out_dim, config.num_labels)
        self.hidden2tag2 = nn.Linear(out_dim, num_labels2)
        self.l2tol1 = nn.Linear(num_labels2, config.num_labels)
        self.softmax = nn.Softmax(dim=-1)
        self.w_gates = nn.Parameter(torch.randn(out_dim, 3), requires_grad=True)
        self.crf1 = CRF(config.num_labels, batch_first=True)
        self.crf2 = CRF(num_labels2, batch_first=True)

    def forward(self, input_ids, tags1, tags2, token_type_ids=None, input_mask=None):
        emissions1, emissions2 = self.tag_outputs(input_ids, token_type_ids, input_mask)
        loss1 = -1 * self.crf1(emissions1, tags1, mask=input_mask.byte())
        loss2 = -1 * self.crf2(emissions2, tags2, mask=input_mask.byte())

        return loss1, loss2

    def tag_outputs(self, input_ids, token_type_ids=None, input_mask=None):

        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)

        sequence_output = outputs[0]

        if self.need_birnn:
            sequence_output, (hn, cn) = self.birnn(sequence_output)

        # ### MultiheadAttention ###
        # x = sequence_output
        # x, _ = self.attn(x, x, x)
        # x = self.dropout(x)
        # emissions1 = self.hidden2tag1(x)
        # ###   end   ###

        # ### ScaledDotProductAttention ###
        # x = sequence_output
        # # h_n = torch.cat([s for s in hn], 1)  # [batch, hidden_size]
        # # x, _ = self.attn(h_n, x, x)
        # x, _ = self.attn(x, x, x)
        #
        # x = self.dropout(x)
        # emissions1 = self.hidden2tag1(x)
        # ### end ###

        ### LabelAttention ###
        x = sequence_output
        x = self.dropout(x)
        _, attn = self.attn(x)  # [batch_size, seq_len, num_labels]
        emiss1 = self.hidden2tag1(x) + attn
        ### end ###

        sequence_output = self.dropout(sequence_output)
        # emissions1 = self.hidden2tag1(sequence_output)
        emissions2 = self.hidden2tag2(sequence_output)

        emiss2 = self.l2tol1(emissions2)
        gates_o = self.softmax(x @ self.w_gates)
        experts_o_tensor = torch.stack([emiss1, attn, emiss2], dim=-1)

        gates_o = torch.unsqueeze(gates_o, 2).expand_as(experts_o_tensor)

        tower_input = gates_o * experts_o_tensor
        emissions1 = torch.sum(tower_input, dim=-1)

        return emissions1, emissions2

    def predict(self, input_ids, token_type_ids=None, input_mask=None):
        emissions1, emissions2 = self.tag_outputs(input_ids, token_type_ids, input_mask)
        return self.crf1.decode(emissions1, input_mask.byte())


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


if __name__ == '__main__':
    awl = AutomaticWeightedLoss(2)
    print(awl.parameters())
