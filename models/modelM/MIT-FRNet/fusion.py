import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class GRUencoder(nn.Module):
    def __init__(self, embedding_dim, utterance_dim, num_layers):
        super(GRUencoder, self).__init__()
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=utterance_dim, bidirectional=True, num_layers=num_layers)
    def forward(self, utterance, utterance_lens):
        utterance_embs = utterance.transpose(0,1)
        sorted_utter_length, indices = torch.sort(utterance_lens, descending=True)
        _, indices_unsort = torch.sort(indices)
        s_embs = utterance_embs.index_select(1, indices)
        utterance_packed = pack_padded_sequence(s_embs, sorted_utter_length.cpu())
        utterance_output = self.gru(utterance_packed)[0]
        utterance_output = pad_packed_sequence(utterance_output, total_length=utterance.size(1))[0]
        utterance_output = utterance_output.index_select(1, indices_unsort)
        return utterance_output.transpose(0,1)
class C_GATE(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, drop):
        super(C_GATE, self).__init__()
        self.gru = GRUencoder(embedding_dim, hidden_dim, num_layers)
        self.cnn = nn.Conv1d(in_channels= 2 * hidden_dim, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(hidden_dim * 2 + embedding_dim, hidden_dim)
        self.dropout_in = nn.Dropout(drop)
    def forward(self, utterance, utterance_mask):
        add_zero = torch.zeros(size=[utterance.shape[0], 1], requires_grad=False).type_as(utterance_mask).to(utterance_mask.device)
        utterance_mask = torch.cat((utterance_mask, add_zero), dim=1)
        utterance_lens = torch.argmin(utterance_mask, dim=1)
        transformed_ = self.gru(utterance, utterance_lens) 
        gate = F.sigmoid(self.cnn(transformed_.transpose(1, 2)).transpose(1, 2)) 
        gate_x = torch.tanh(transformed_) * gate 
        utterance_rep = torch.tanh(self.fc(torch.cat([utterance, gate_x], dim=-1))) 
        utterance_rep = torch.max(utterance_rep, dim=1)[0] 
        utterance_rep = self.dropout_in(utterance_rep) 
        return utterance_rep
class GATE_F(nn.Module):
    def __init__(self, args):
        super(GATE_F, self).__init__()
        self.text_encoder = C_GATE(args.fusion_t_in, args.fusion_t_hid, args.fusion_gru_layers, args.fusion_drop)
        self.audio_encoder = C_GATE(args.fusion_a_in, args.fusion_a_hid, args.fusion_gru_layers, args.fusion_drop)
        self.vision_encoder = C_GATE(args.fusion_v_in, args.fusion_v_hid, args.fusion_gru_layers, args.fusion_drop)
        self.classifier = nn.Sequential()
        self.classifier.add_module('linear_trans_norm', nn.BatchNorm1d(args.fusion_t_hid + args.fusion_a_hid + args.fusion_v_hid))
        self.classifier.add_module('linear_trans_hidden', nn.Linear(args.fusion_t_hid + args.fusion_a_hid + args.fusion_v_hid, args.cls_hidden_dim))
        self.classifier.add_module('linear_trans_activation', nn.LeakyReLU())
        self.classifier.add_module('linear_trans_drop', nn.Dropout(args.cls_dropout))
        self.classifier.add_module('linear_trans_final', nn.Linear(args.cls_hidden_dim, 1))
    def forward(self, text_x, audio_x, vision_x):
        text_x, text_mask = text_x
        audio_x, audio_mask = audio_x
        vision_x, vision_mask = vision_x
        text_rep = self.text_encoder(text_x, text_mask)
        audio_rep = self.audio_encoder(audio_x, audio_mask)
        vision_rep = self.vision_encoder(vision_x, vision_mask)
        utterance_rep = torch.cat((text_rep, audio_rep, vision_rep), dim=1)  
        return self.classifier(utterance_rep)
MODULE_MAP = {
    'c_gate': GATE_F,
}
class Fusion(nn.Module):
    def __init__(self, args):
        super(Fusion, self).__init__()
        select_model = MODULE_MAP[args.fusionModule]
        self.Model = select_model(args)
    def forward(self, text_x, audio_x, vision_x):
        return self.Model(text_x, audio_x, vision_x)