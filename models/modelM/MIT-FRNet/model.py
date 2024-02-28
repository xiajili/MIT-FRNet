import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from models.modelM.MIT-FRNet.alignment import Alignment
from models.modelM.MIT-FRNet.generator import Generator
from models.subNets.BertTextEncoder import BertTextEncoder
from models.modelM.MIT-FRNet.fusion import Fusion
from models.modelM.MIT-FRNet.Utils import FocalLoss, make_weights_for_balanced_classes, SMLossAVT

class CMD(nn.Module):
    def __init__(self):
        super(CMD, self).__init__()
    def forward(self, x1, x2, n_moments=3):
        x1 = x1.view(-1, x1.shape[-1])
        x2 = x2.view(-1, x2.shape[-1])
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        b = torch.max(x2, dim=0)[0]
        a = torch.min(x2, dim=0)[0]
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms
    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = (summed+1e-12)**(0.5)
        return sqrt
    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)
class RECLoss(nn.Module):
    def __init__(self, args):
        super(RECLoss, self).__init__()
        self.eps = torch.FloatTensor([1e-4]).to(args.device)
        self.args = args
        if args.recloss_type == 'SmoothL1Loss': self.loss = nn.SmoothL1Loss(reduction='sum')
        elif args.recloss_type == 'MSELoss': self.loss = nn.MSELoss(reduction='sum')
        elif args.recloss_type == 'cmd': self.loss = CMD()
        elif args.recloss_type == 'combine': 
            self.loss = nn.SmoothL1Loss(reduction='sum')
            self.loss_cmd = CMD()
    def forward(self, pred, target, mask):
        mask = mask.unsqueeze(-1).expand(pred.shape[0], pred.shape[1], pred.shape[2])
        loss = self.loss(pred*mask, target*mask) / (torch.sum(mask) + self.eps)
        if self.args.recloss_type == 'combine' and self.args.weight_sim_loss!=0:
            loss += (self.args.weight_sim_loss * self.loss_cmd(pred*mask, target*mask) / (torch.sum(mask) + self.eps))
        return loss
###################################
def mean_temporal(data, aug_dim):
    mean_features = torch.mean(data, dim=aug_dim)
    return mean_features
class FBP(nn.Module):
    def __init__(self, d_emb_1, d_emb_2, fbp_hid, fbp_k, dropout):
        super(FBP, self).__init__()
        self.fusion_1_matrix = nn.Linear(d_emb_1, fbp_hid*fbp_k, bias=False)
        self.fusion_2_matrix = nn.Linear(d_emb_2, fbp_hid*fbp_k, bias=False)
        self.fusion_dropout = nn.Dropout(dropout)
        self.fusion_pooling = nn.AvgPool1d(kernel_size=fbp_k)
        self.fbp_k = fbp_k
    def forward(self, seq1, seq2):
        seq1 = self.fusion_1_matrix(seq1)
        seq2 = self.fusion_2_matrix(seq2)
        fused_feature = torch.mul(seq1, seq2)
        if len(fused_feature.shape) == 2:
            fused_feature = fused_feature.unsqueeze(0)
        fused_feature = self.fusion_dropout(fused_feature)
        fused_feature = self.fusion_pooling(fused_feature).squeeze(0) * self.fbp_k
        fused_feature = F.normalize(fused_feature, dim=-1, p=2)
        return fused_feature
#################################
class MIT-FRNet(nn.Module):
    def __init__(self, args):
        super(MIT-FRNet, self).__init__()
        self.args = args
        self.d_a = args.a_dim
        self.d_v = args.v_dim
        self.d_t = args.t_dim
        self.d_out = args.d_out
        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_bert_finetune)
        self.align_subnet = Alignment(args)
        #########################
        self.fc_a = nn.Linear(self.d_a, self.d_out)
        self.fc_v = nn.Linear(self.d_v, self.d_out)
        self.fc_t = nn.Linear(self.d_t, self.d_out)
        self.fbp = FBP(self.d_k, self.d_k, 32, 2, dropout)   #
        self.fc_gate = nn.Linear(32, 1)   #
        self.gate_activate = nn.Tanh()   #
        self.sm_loss_func = SMLossAVT('gau', 'vec')
        self.w_1 = nn.Linear(d_in, d_hid)  #
        self.w_2 = nn.Linear(d_hid, d_in)  #
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)   #
        self.dropout = nn.Dropout(dropout)   #
        ########################
        if not args.without_generator:
            ############################################################################
            self.fc_a = nn.Linear(self.d_a, self.d_out)   #
            self.fc_v = nn.Linear(self.d_v, self.d_out)   #
            self.fc_t = nn.Linear(self.d_t, self.d_out)   #
            ############################################################################
            self.generator_t = Generator(args, modality='text')
            self.generator_a = Generator(args, modality='audio')
            self.generator_v = Generator(args, modality='vision')
            self.gen_loss = RECLoss(args)
        args.fusion_t_in = args.fusion_a_in = args.fusion_v_in = args.dst_feature_dim_nheads[0] * 3
        self.fusion_subnet = Fusion(args)
    def forward(self, text, audio, vision):
        text, text_m, missing_mask_t = text
        audio, audio_m, audio_mask, missing_mask_a = audio 
        vision, vision_m, vision_mask, missing_mask_v = vision
        text_mask = text[:,1,:]
        text_m = self.text_model(text_m)
        text = self.text_model(text)
        text_h, audio_h, vision_h, text_h_g, audio_h_g, vision_h_g = self.align_subnet(text_m, audio_m, vision_m)
        ########################
        gate_ = self.fbp(q_, k_)   #
        gate_ = self.gate_activate(self.fc_gate(gate_))   #
        gate_sign = gate_ / torch.abs(gate_)   #
        gate_ = (gate_sign + torch.abs(gate_sign)) / 2.0   #
        if len(gate_.shape) == 2:   #
            gate_ = gate_.unsqueeze(-1)   #
        result = result * gate_   #
        result += residual   #
        result = self.layer_norm(result)   #
        result = result.masked_fill(torch.isnan(result), 0.0)   #
        #######################   
        ########################
        residual = result   #
        result = self.w_2(F.relu(self.w_1(result)))   #
        result = self.dropout(result)   #
        result += residual   #
        result = self.layer_norm(result)   #
        result = result.masked_fill(torch.isnan(result), 0.0)   #
        #######################
        if not self.args.without_generator:
            text_ = self.generator_t(text_h_g)
            audio_ = self.generator_a(audio_h_g)
            vision_ = self.generator_v(vision_h_g)
            ############################################################################
            audio_h_g, vision_h_g, text_h_g = audio_h_g.transpose(1, 2), vision_h_g.transpose(1, 2), text_h_g.transpose(1, 2)
            A_F_ = self.fc_a(audio_h_g)
            V_F_ = self.fc_v(vision_h_g)
            T_F_ = self.fc_t(text_h_g)
            A_F, V_F, T_F = mean_temporal(A_F_, 1), mean_temporal(V_F_, 1), mean_temporal(T_F_, 1)   #
            loss_sm = self.sm_loss_func((V_F_, A_F_, T_F_))
            ############################################################################
            text_gen_loss = self.gen_loss(text_, text, text_mask - missing_mask_t)           
            audio_gen_loss = self.gen_loss(audio_, audio, audio_mask - missing_mask_a)
            vision_gen_loss = self.gen_loss(vision_, vision, vision_mask - missing_mask_v)
            prediction = self.fusion_subnet((text_h, text_mask), (audio_h, audio_mask), (vision_h, vision_mask))
            return prediction, self.args.weight_gen_loss[0] * text_gen_loss + self.args.weight_gen_loss[1] * audio_gen_loss + self.args.weight_gen_loss[2] * vision_gen_loss + 0.001 * loss_sm
        else:
            prediction = self.fusion_subnet((text_h, text_mask), (audio_h, audio_mask), (vision_h, vision_mask))
            return prediction, torch.Tensor([0]).to(self.args.device)