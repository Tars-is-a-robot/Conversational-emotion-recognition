import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv
import numpy as np, itertools, random, copy, math
from model_GCN import GCN_2Layers, GCNLayer1, GCNII, TextCNN
from model_mm import MM_GCN, MM_GCN2
from typing import List, Optional
import ipdb
import utils



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 'MELD' 用的损失函数
class FocalLoss(nn.Module):
    def __init__(self, gamma = 2.5, alpha = 1, size_average = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
    
    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        labels_length = logits.size(1)
        seq_length = logits.size(0)

        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([seq_length, labels_length]).to(device).scatter_(1, new_label, 1)
        log_p = F.log_softmax(logits, dim = -1)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt)**self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()
# 先保留

class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1,1)
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target)\
                            /torch.sum(self.weight[target]*mask_.squeeze())
        return loss

 # t 预处理
class Pretreatment(nn.Module):
    def __init__(self, args, residual=False):

        super(Pretreatment, self).__init__()

        self.model = args.text_model #model=2(iemocap)
        self.norm_strategy = args.text_norm #norm=3(iemocap)

        if self.model == 0:
            D_x = 4 * args.D_m
        elif self.model == 1:
            D_x = 2 * args.D_m
        else:
            D_x = args.D_m   #执行D_x=D_m=1024(iemocap)

        self.linear_in = nn.Linear(D_x, args.D_text)  # in 1024 out 100(文本线性层)
        self.residual = residual  # False (iemocap)
        self.r_weights = nn.Parameter(torch.tensor([0.25, 0.25, 0.25, 0.25])) #使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化 猜测应该是这四个数据集在优化参数过程中所占比例

        norm_train = True
        self.norm1a = nn.LayerNorm(args.D_m, elementwise_affine=norm_train) # D_m=1024 归一化（目的是为了把输入转化成均值为0方差为1的数据）最后一个维度 且有学习参数（elementwise_affine=ture）
        self.norm1b = nn.LayerNorm(args.D_m, elementwise_affine=norm_train)
        self.norm1c = nn.LayerNorm(args.D_m, elementwise_affine=norm_train)
        self.norm1d = nn.LayerNorm(args.D_m, elementwise_affine=norm_train)
        """https://blog.csdn.net/weixin_39228381/article/details/107896863?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164827728616780255212164%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=164827728616780255212164&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-8-107896863.142^v5^pc_search_result_cache,143^v6^control&utm_term=nn.BatchNorm1d%EF%BC%88%EF%BC%89&spm=1018.2226.3001.4187
            一种归一化 mlp与cnn上使用表现优异 加在激活值获得后 非线性函数变换前 
            BatchNorm就是在深度神经网络训练过程中使得每一层神经网络的输入保持相同分布的 num_features就是你需要归一化的那一维的维度
            norm_train = True 最后一维归一化 添加可学习仿射变换函数
        """
        self.norm3a = nn.BatchNorm1d(args.D_m, affine=norm_train)
        self.norm3b = nn.BatchNorm1d(args.D_m, affine=norm_train)
        self.norm3c = nn.BatchNorm1d(args.D_m, affine=norm_train)
        self.norm3d = nn.BatchNorm1d(args.D_m, affine=norm_train)


        self.exchange_graph1 = nn.Linear(args.D_s, args.D_audio)  #  进入图卷积的知识维度变换g线性层
        self.exchange_graph2 = nn.Linear(args.D_s, args.D_visual)  # 进入图卷积的知识维度变换p线性层
        self.exchange_graph3 = nn.Linear(args.D_s, args.D_visual)  # 进入图卷积的知识维度变换p线性层
        self.exchange_graph4 = nn.Linear(args.D_s, args.D_visual)  # 进入图卷积的知识维度变换p线性层
        self.exchange_graph5 = nn.Linear(args.D_s, args.D_visual)  # 进入图卷积的知识维度变换p线性层
        self.exchange_graph6 = nn.Linear(args.D_s, args.D_visual)  # 进入图卷积的知识维度变换p线性层
        self.exchange_graph7 = nn.Linear(args.D_s, args.D_visual)  # 进入图卷积的知识维度变换p线性层
        self.exchange_graph8 = nn.Linear(args.D_s, args.D_visual)  # 进入图卷积的知识维度变换p线性层
        self.exchange_graph9 = nn.Linear(args.D_s, args.D_visual)  # 进入图卷积的知识维度变换p线性层

        self.exchange_order1 = nn.Linear(args.D_s, args.order_know_dim)  # 进入顺序变换的知识维度变换r线性层
        self.exchange_order2 = nn.Linear(args.D_s, args.order_know_dim)  #
        self.exchange_order3 = nn.Linear(args.D_s, args.order_know_dim)
        self.exchange_order4 = nn.Linear(args.D_s, args.order_know_dim)

        #utils.initialize_weights(self)


    def forward(self, r1, r2, r3, r4, x1, x2, x3, x4, x5, x6, o1, o2, o3):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        r1, r2, r3, r4, ->>seq*batch*dim  dim=1024 （话语特征）
        x5, x6, x1, o2, o3, ->>seq*batch*dim  dim=768(所使用的5种类型的外部知识)
        qmask  ->>seq*batch*dim  dim=2    umask,label ->>batch*seq   att2=True   (iemocap)

        """
        seq_len, batch, feature_dim = r1.size()
        if self.norm_strategy == 1:
            r1 = self.norm1a(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1,0)
            r2 = self.norm1b(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1,0)
            r3 = self.norm1c(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1,0)
            r4 = self.norm1d(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1,0)
        elif self.norm_strategy == 2:
            norm2 = nn.LayerNorm((seq_len, feature_dim), elementwise_affine=False)
            r1 = norm2(r1.transpose(0, 1)).transpose(0, 1)
            r2 = norm2(r2.transpose(0, 1)).transpose(0, 1)
            r3 = norm2(r3.transpose(0, 1)).transpose(0, 1)
            r4 = norm2(r4.transpose(0, 1)).transpose(0, 1)
        elif self.norm_strategy == 3:
            ##经过归一化 r1, r2, r3, r4回到原形状 seq*batch*dim  (3,iemocap执行)
            r1 = self.norm3a(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1,0)
            r2 = self.norm3b(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1,0)
            r3 = self.norm3c(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1,0)
            r4 = self.norm3d(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1,0)

        if self.model == 0:
            r = torch.cat([r1, r2, r3, r4], axis=-1)
        elif self.model == 1:
            r = torch.cat([r1, r2], axis=-1)
        elif self.model == 2:  ## (2,iemocap执行,r->>seq*batch*dim)
            r = (r1 + r2 + r3 + r4) / 4
        elif self.model == 3:
            r = r1
        elif self.model == 4:
            r = r2
        elif self.model == 5:
            r = r3
        elif self.model == 6:
            r = r4
        elif self.model == 7:
            r = self.r_weights[0] * r1 + self.r_weights[1] * r2 + self.r_weights[2] * r3 + self.r_weights[3] * r4

        r = self.linear_in(r)  ## (iemocap 经过线性层 得r->>seq*batch*100)

        # o1 = torch.cat([o1,o2],dim=-1)
        know_g = self.exchange_graph1(x6) ##
        know_p = self.exchange_graph2(o3)
        know_k1 = self.exchange_graph3(x1)
        know_k2 = self.exchange_graph4(x3)
        know_k3 = self.exchange_graph4(o2)


        k1 = self.exchange_order1(x1)
        k2 = self.exchange_order2(o2)
        k3 = self.exchange_order3(x3)
        k4 = self.exchange_order4(o1)

        return r, know_g, know_p, k1, k2, k3, k4, know_k1, know_k2, know_k3

# t
class DialogueGCNModel(nn.Module):

    def __init__(self, args, D_m, D_g, D_p, D_e, D_h, D_a, graph_hidden_size, n_speakers, max_seq_len,
                 n_classes=7, dynamic_edge_w=False):

        super(DialogueGCNModel, self).__init__()

        listener_state=args.active_listener
        context_attention = args.attention
        dropout_rec = 0.5
        dropout = args.dropout
        nodal_attention = args.nodal_attention
        D_m_v = args.D_visual
        D_m_a = args.D_audio
        modals = args.modals
        Deep_GCN_nlayers = args.Deep_GCN_nlayers
        self.seqlstm = Lstmhidden(args)
        self.args = args
        self.base_model = args.base_model
        self.avec = False
        self.no_cuda = args.no_cuda
        self.graph_type = args.graph_type
        self.multiheads = args.multiheads
        self.graph_construct = args.graph_construct
        self.use_topic = args.use_topic
        self.dropout = dropout
        self.use_GCN = args.use_gcn
        self.use_residue = args.use_residue
        self.dynamic_edge_w = dynamic_edge_w
        self.return_feature = True
        self.modals = [x for x in modals]  # a, v, l
        self.use_speaker = args.use_speaker
        self.use_modal = args.use_modal
        self.att_type = args.mm_fusion_mthd

        if self.att_type == 'gated' or self.att_type == 'concat_subsequently':
            self.multi_modal = True
            self.av_using_lstm = args.av_using_lstm
        else:
            self.multi_modal = False
        self.use_bert_seq = False
        self.dataset = args.Dataset

        """ 三种模态的预处理阶段
            base_model == 'LSTM' 且 multi_modal = Ture 则 进入else中 
            进行三种模态的预处理工作 声音和视频进行线性层 文本进到线性层加lstm
        """


        if self.base_model == 'DialogRNN':
            self.dialog_rnn_f = DialogueRNN(D_m, D_g, D_p, D_e, listener_state, context_attention, D_a, dropout_rec)
            self.dialog_rnn_r = DialogueRNN(D_m, D_g, D_p, D_e, listener_state, context_attention, D_a, dropout_rec)

        elif self.base_model == 'LSTM':
            if not self.multi_modal:
                if len(self.modals) == 3:
                    hidden_ = 250
                elif ''.join(self.modals) == 'al':
                    hidden_ = 150
                elif ''.join(self.modals) == 'vl':
                    hidden_ = 150
                else:
                    hidden_ = 100
                self.linear_ = nn.Linear(D_m, hidden_)
                self.lstm = nn.LSTM(input_size=hidden_, hidden_size=D_e, num_layers=2, bidirectional=True,
                                    dropout=dropout)

             ## 使用下边几个线性层

            else:
                if 'a' in self.modals:
                    hidden_a = args.graph_fusion_dim
                    self.linear_a = nn.Linear(D_m_a, hidden_a)
                    if self.av_using_lstm:
                        self.lstm_a = nn.LSTM(input_size=hidden_a, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)

                if 'v' in self.modals:
                    hidden_v = args.graph_fusion_dim
                    self.linear_v = nn.Linear(D_m_v, hidden_v)
                    if self.av_using_lstm:
                        self.lstm_v = nn.LSTM(input_size=hidden_v, hidden_size=D_e, num_layers=2, bidirectional=True,
                                              dropout=dropout)
                if 'l' in self.modals:
                    hidden_l = args.graph_fusion_dim
                    if self.use_bert_seq:
                        self.txtCNN = TextCNN(input_dim=D_m, emb_size=hidden_l)
                    else:
                        self.linear_l = nn.Linear(D_m, hidden_l)
                    self.lstm_l = nn.LSTM(input_size=hidden_l, hidden_size=D_e, num_layers=2, bidirectional=True,
                                          dropout=dropout)

                # 新添加三个线性层 用来给知识预处理
                self.linear_k1 = nn.Linear(D_m_a, args.graph_fusion_dim)
                self.linear_k2 = nn.Linear(D_m_a, args.graph_fusion_dim)
                self.linear_k3 = nn.Linear(D_m_a, args.graph_fusion_dim)


        elif self.base_model == 'GRU':
            self.gru = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        elif self.base_model == 'None':
            self.base_linear = nn.Linear(D_m, 2*D_e)
        else:
            print ('Base model must be one of DialogRNN/LSTM/GRU')
            raise NotImplementedError


        n_relations = 2 * n_speakers ** 2
        self.window_past = args.windowp
        self.window_future = args.windowf
        self.att_model = MaskedEdgeAttention(2 * D_e, max_seq_len, self.no_cuda)
        self.nodal_attention = nodal_attention

        """ 
            将三种模态的向量怎样进行拼接 且graph_type=='MMGCN' 则self.graph_model = MM_GCN()这个类
            这个类完成了节点向量更新的工作 
        """
        if self.graph_type == 'relation':
            if not self.multi_modal:
                self.graph_net = GraphNetwork(2 * D_e, n_classes, n_relations, max_seq_len, graph_hidden_size, dropout,
                                              self.no_cuda, self.use_GCN)
            else:
                if 'a' in self.modals:
                    self.graph_net_a = GraphNetwork(2 * D_e, n_classes, n_relations, max_seq_len, graph_hidden_size,
                                                    dropout, self.no_cuda, self.use_GCN, self.return_feature)
                if 'v' in self.modals:
                    self.graph_net_v = GraphNetwork(2 * D_e, n_classes, n_relations, max_seq_len, graph_hidden_size,
                                                    dropout, self.no_cuda, self.use_GCN, self.return_feature)
                if 'l' in self.modals:
                    self.graph_net_l = GraphNetwork(2 * D_e, n_classes, n_relations, max_seq_len, graph_hidden_size,
                                                    dropout, self.no_cuda, self.use_GCN, self.return_feature)
            print("construct relation graph")
        elif self.graph_type == 'GCN3':
            if not self.multi_modal:
                self.graph_net = GCN_2Layers(2 * D_e, graph_hidden_size, n_classes, self.dropout, self.use_topic,
                                             self.use_residue)
            else:
                if 'a' in self.modals:
                    self.graph_net_a = GCN_2Layers(2 * D_e, graph_hidden_size, n_classes, self.dropout, self.use_topic,
                                                   self.use_residue, self.return_feature)
                if 'v' in self.modals:
                    self.graph_net_v = GCN_2Layers(2 * D_e, graph_hidden_size, n_classes, self.dropout, self.use_topic,
                                                   self.use_residue, self.return_feature)
                if 'l' in self.modals:
                    self.graph_net_l = GCN_2Layers(2 * D_e, graph_hidden_size, n_classes, self.dropout, self.use_topic,
                                                   self.use_residue, self.return_feature)
            use_topic_str = "using topic" if self.use_topic else "without using topic"
            print("construct " + self.graph_type + " " + use_topic_str)
        elif self.graph_type == 'DeepGCN':
            if not self.multi_modal:
                self.return_feature = False
                self.graph_net = GCNII(nfeat=2 * D_e, nlayers=Deep_GCN_nlayers, nhidden=graph_hidden_size,
                                       nclass=n_classes, dropout=self.dropout, lamda=0.5, alpha=0.1, variant=True,
                                       return_feature=self.return_feature, use_residue=self.use_residue)  # 最后一个参数是调是否利用
            else:
                if 'a' in self.modals:
                    self.graph_net_a = GCNII(nfeat=2 * D_e, nlayers=Deep_GCN_nlayers, nhidden=graph_hidden_size,
                                             nclass=n_classes, dropout=self.dropout, lamda=0.5, alpha=0.1, variant=True,
                                             return_feature=self.return_feature, use_residue=self.use_residue)
                if 'v' in self.modals:
                    self.graph_net_v = GCNII(nfeat=2 * D_e, nlayers=Deep_GCN_nlayers, nhidden=graph_hidden_size,
                                             nclass=n_classes, dropout=self.dropout, lamda=0.5, alpha=0.1, variant=True,
                                             return_feature=self.return_feature, use_residue=self.use_residue)
                if 'l' in self.modals:
                    self.graph_net_l = GCNII(nfeat=2 * D_e, nlayers=Deep_GCN_nlayers, nhidden=graph_hidden_size,
                                             nclass=n_classes, dropout=self.dropout, lamda=0.5, alpha=0.1, variant=True,
                                             return_feature=self.return_feature, use_residue=self.use_residue)
            print("construct " + self.graph_type, "with", Deep_GCN_nlayers, "layers")
        elif self.graph_type == 'MMGCN' or self.graph_type == 'MMGCN2':
            if self.graph_type == 'MMGCN':  ##graph_type = 'MMGCN'
                self.graph_model = MM_GCN(args, nhidden=graph_hidden_size, nclass=n_classes, variant=True,
                                          return_feature=self.return_feature, n_speakers=n_speakers, modals=self.modals)
            else:
                self.graph_model = MM_GCN2(nfeat=2 * D_e, nlayers=64, nhidden=graph_hidden_size, nclass=n_classes,
                                           dropout=self.dropout, lamda=0.5, alpha=0.1, variant=True,
                                           return_feature=self.return_feature, use_residue=self.use_residue,
                                           modals=modals, mm_graph=self.graph_construct)
            print("construct " + self.graph_type)
        elif self.graph_type == 'None':
            if not self.multi_modal:
                self.graph_net = nn.Linear(2 * D_e, n_classes)
            else:
                if 'a' in self.modals:
                    self.graph_net_a = nn.Linear(2 * D_e, graph_hidden_size)
                if 'v' in self.modals:
                    self.graph_net_v = nn.Linear(2 * D_e, graph_hidden_size)
                if 'l' in self.modals:
                    self.graph_net_l = nn.Linear(2 * D_e, graph_hidden_size)
            print("construct Bi-LSTM")
        else:
            print("There are no such kind of graph")

        edge_type_mapping = {}
        for j in range(n_speakers):
            for k in range(n_speakers):
                edge_type_mapping[str(j) + str(k) + '0'] = len(edge_type_mapping)
                edge_type_mapping[str(j) + str(k) + '1'] = len(edge_type_mapping)
        self.edge_type_mapping = edge_type_mapping

        self.crf_classic = CRF(n_classes)  #CRF层

        if self.multi_modal:
            self.gatedatt = MMGatedAttention(2 * D_e + graph_hidden_size, graph_hidden_size, att_type='general')
            self.dropout_ = nn.Dropout(self.dropout)
            if self.att_type == 'concat_subsequently':   ## 最后的线性层分类   拼接的话就在这里更改
                if args.use_know_graph and not args.only_text:
                    self.smax_fc = nn.Linear((graph_hidden_size + args.graph_fusion_dim) * len(self.modals), n_classes)
                else:
                    self.smax_fc = nn.Linear(graph_hidden_size + args.graph_fusion_dim, n_classes)
            elif self.att_type == 'gated':
                if len(self.modals) == 3:
                    self.smax_fc = nn.Linear(100 * len(self.modals), n_classes)
                else:
                    self.smax_fc = nn.Linear(100, n_classes)
            else:
                self.smax_fc = nn.Linear(2 * D_e + graph_hidden_size * len(self.modals), n_classes) #

        if args.use_cnn:
            self.CNN_d1 = nn.Conv1d(in_channels=args.D_text, out_channels=128,
                                    kernel_size=3, dilation=1, padding=1)
            self.CNN_d2 = nn.Conv1d(in_channels=args.D_text, out_channels=128,
                                    kernel_size=3, dilation=2, padding=2)
            self.CNN_d3 = nn.Conv1d(in_channels=args.D_text, out_channels=128,
                                    kernel_size=3, dilation=3, padding=3)
            self.ReLU = nn.ReLU()
            self.learningToRank = nn.Linear(128 * 3, args.graph_fusion_dim)
            self.learningToRank1 = nn.Linear(2 * args.graph_fusion_dim, args.graph_fusion_dim)
        if self.args.use_cnnd:
            self.cnnd = CNNEncoder(100, 128, 128)
            self.cnnd_linear = nn.Linear(2 * args.graph_fusion_dim, args.graph_fusion_dim)
        #utils.initialize_weights(self)

    def _cnn_resnet(self, C):
            # C [100,150,20]

            C1 = self.CNN_d1(C)
            RC1 = C1.transpose(-2, -1)
            RC1 = self.ReLU(RC1)

            C2 = self.CNN_d2(C)
            RC2 = C2.transpose(-2, -1)
            RC2 = self.ReLU(RC2)

            C3 = self.CNN_d3(C)
            RC3 = C3.transpose(-2, -1)
            RC3 = self.ReLU(RC3)  # [100,20,150]

            TRC = torch.cat([RC1, RC2, RC3], dim=-1).transpose(0, 1)
            # print(TRC.size())

            return TRC

    def _reverse_seq(self, X, mask):
        """
        输入是一个batch的序列 X - seq，batch，dim      mask - batch，seq 就是有话语的地方标记为1 没话语的地方标记为0
        输出是 把这个序列反向之后的表示 seq，batch，dim
        """
        X_ = X.transpose(0, 1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)

    def forward(self,label, train, U, qmask, umask, seq_lengths, global_sdj_den, front_sdj_den, back_sdj_den, s_mask, k1_0, k2_0, k3_0, k4_0, U_a=None, U_v=None, know_k1=None, know_k2=None, know_k3=None):

        """
            label batch,seq
            输入 U, tensor(seq,batch,100) 文本
            qmask, tensor(seq,batch,2) 说话人矩阵 独热向量表示
            umask, tensor(batch,seq) 说话人标记向量 有话语的用1 没话语的用0
            seq_lengths, list:batch  表示每段对话所含话语数量
            U_a, tensor(seq,batch,300) 图融合知识1
            U_v, tensor(seq,batch,300) 图融合知识2
            front_sdj_den, back_sdj_den, global_sdj_den  tensor(batch,seq,seq) 应注意的话语
            k1_0, k2_0, k3_0, k4_0       tensor(seq,batch,100)  用于顺序层中的知识向量
            只返回 log_prob tensor(sum_seq,classic) 话语情感分类 其他都是 none
        """

        """ base_model == 'LSTM'
            文本和两个知识向量的预处理  av_using_lstm = False 说明两个知识不进行lstm 只经过全连接 得到
            emotions_a = U_a ->tensor(seq,batch,200)             emotions_v = U_v ->tensor(seq,batch,200)
            进入文本处理 use_bert_seq = False 只将文本经过全连接与lstm 得到
            emotions_l ->tensor(seq,batch,200)                hidden_l ->tuple:2  每一个是tensor(4,batch,100)
        """
        if self.base_model == "DialogRNN":
            if self.avec:
                emotions, _ = self.dialog_rnn_f(U, qmask)
            else:
                emotions_f, alpha_f = self.dialog_rnn_f(U, qmask)
                rev_U = self._reverse_seq(U, umask)
                rev_qmask = self._reverse_seq(qmask, umask)
                emotions_b, alpha_b = self.dialog_rnn_r(rev_U, rev_qmask)
                emotions_b = self._reverse_seq(emotions_b, umask)
                emotions = torch.cat([emotions_f, emotions_b], dim=-1)
        elif self.base_model == 'LSTM':  # base_model = 'LSTM'
            if not self.multi_modal:  # multi_modal = Ture 不执行这一步 进入else中
                U = self.linear_(U)
                emotions, hidden = self.lstm(U)
            else:  # 然后下边三个都执行
                if 'a' in self.modals:
                    U_a = self.linear_a(U_a)
                    if self.av_using_lstm:   # 经过了线性层 是否还要再经过双向lstm
                        emotions_a, hidden_a = self.lstm_a(U_a)
                    else:
                        emotions_a = U_a
                if 'v' in self.modals:
                    U_v = self.linear_v(U_v)
                    if self.av_using_lstm:   # 经过了线性层 是否还要再经过双向lstm
                        emotions_v, hidden_v = self.lstm_v(U_v)
                    else:
                        emotions_v = U_v
                if 'l' in self.modals:
                    if self.use_bert_seq:
                        U_ = U.reshape(-1, U.shape[-2], U.shape[-1])
                        U = self.txtCNN(U_).reshape(U.shape[0], U.shape[1], -1)
                    else:
                        U_l = self.linear_l(U)  # 100维变256  seq,batch,dim

                    emotions_l, hidden_l = self.lstm_l(U_l)  #emotions_l seq batch 256
                    if self.args.use_cnn:  # 使用cnn加宽度
                        R = U.permute(1, 2, 0)
                        # C0 = self.ReLU(self.CNN_c0(R)). permute(2,0,1) #torch.Size([16, 200, 74])
                        # C1 = self.ReLU(self.CNN_c1(R))
                        # C2 = self.ReLU(self.CNN_c2(C1))
                        # C3 = self.ReLU(self.CNN_c3(C2))
                        # TRC = torch.cat([C1, C2,C3], dim=-2).transpose(-1, -2)
                        emotions_l1 = self._cnn_resnet(R)
                        emotions_l1 = self.learningToRank(emotions_l1)
                        # print(emotions_l1.size()) #torch.Size([74, 16, 200])
                        # emotions_l1=C0+emotions_l1
                        emotions_l = emotions_l1
                        # emotions_l = torch.cat([emotions_l, emotions_l1], dim=-1)
                        # emotions_l = self.learningToRank1(emotions_l)
                    if self.args.use_cnnd:
                        emotions_l2 = self.cnnd(U) # 每一段对话成为一个向量 batch*256
                        emotions_l2 = emotions_l2.unsqueeze(0)
                        emotions_l = 0.6*emotions_l+0.4*emotions_l2

                        # emotions_l = torch.cat([emotions_l, emotions_l2.repeat(emotions_l.size()[0], 1, 1)], 2)
                        # emotions_l = self.cnnd_linear(emotions_l)

                U_k1 = self.linear_k1(know_k1)
                U_k2 = self.linear_k2(know_k2)
                U_k3 = self.linear_k3(know_k3)


        elif self.base_model == 'GRU':
            emotions, hidden = self.gru(U)
        elif self.base_model == 'None':
            emotions = self.base_linear(U)


        """ 
            multi_modal = Ture 故不执行 进入else语句 这里边的三个if都成立 即执行后返回的是三种模态的初始图节点向量
            且 graph_type=='MMGCN' 则都执行的是 simple_batch_graphify() 函数  
            这个函数的输入 features_a\v\l ->tensor(seq,batch,200)单独模态的向量表示   seq_lengths ->list:16 表示每段对话的话语数量
            返回 features_a\v\l ->tensor(sum_seq,200) sum_seq 为这个batch中话语的真实数量 相当于把这个batch中的三种模态的向量表示各自拼起来
            每种模态表示从形状 tensor(seq,batch,200) 变成形状 tensor(sum_seq,200)
            其他四个返回都是none
        """
        if not self.multi_modal:  # 不执行
            if self.graph_type == 'relation':
                features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(emotions, qmask,
                                                                                                seq_lengths,
                                                                                                self.window_past,
                                                                                                self.window_future,
                                                                                                self.edge_type_mapping,
                                                                                                self.att_model,
                                                                                                self.no_cuda)
            else:
                features, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(emotions,
                                                                                                       seq_lengths,
                                                                                                       self.no_cuda)
        else:      ## 进到这里 然后由于modals = 'a'，'v'，'l'  因此以下都执行 且graph_type=='MMGCN'
            if 'a' in self.modals:
                if self.graph_type == 'relation':
                    features_a, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(emotions_a, qmask,seq_lengths,self.window_past,self.window_future,
                                                                                                      self.edge_type_mapping,self.att_model,self.no_cuda)
                else:
                    features_a, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(emotions_a,seq_lengths,self.no_cuda)
            if 'v' in self.modals:
                if self.graph_type == 'relation':
                    features_v, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(emotions_v, qmask,seq_lengths,self.window_past,self.window_future,
                                                                                                      self.edge_type_mapping,self.att_model,self.no_cuda)
                else:
                    features_v, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(emotions_v,seq_lengths,self.no_cuda)
            if 'l' in self.modals:
                if self.graph_type == 'relation':
                    features_l, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(emotions_l, qmask,seq_lengths, self.window_past,self.window_future,
                                                                                                      self.edge_type_mapping, self.att_model,self.no_cuda)
                else:
                    features_l, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(emotions_l,seq_lengths,self.no_cuda)
            features_k1, _, _, _, _ = simple_batch_graphify(U_k1, seq_lengths, self.no_cuda)
            features_k2, _, _, _, _ = simple_batch_graphify(U_k2, seq_lengths, self.no_cuda)
            features_k3, _, _, _, _ = simple_batch_graphify(U_k3, seq_lengths, self.no_cuda)

        """
            上边初始图节点向量创建后  且graph_type = 'MMGCN'  图的类型 进入 self.graph_model = MM_GCN() 为 model_mm.py 中的 class MM_GCN(nn.Module)
            MM_GCN() 输入为节点向量 features_a, features_v, features_l ->tensor(sum_seq,200)  sum_seq 为这个batch中话语的真实数量
            seq_lengths ->list:16 表示每段对话的话语数量        qmask ->tensor(seq,batch,2) 说话人矩阵 独热向量表示
            输出为  emotions_feat = features ->tensor(sum_seq,900) 完成节点的更新工作 并将更新后的3种模态的特征拼接起来  不拼接就是 (sum_seq,9=300)
        """
        if self.graph_type == 'relation':
            if not self.multi_modal:
                log_prob = self.graph_net(features, edge_index, edge_norm, edge_type, seq_lengths, umask,
                                          self.nodal_attention, self.avec)
            else:
                if 'a' in self.modals:
                    emotions_a = self.graph_net_a(features_a, edge_index, edge_norm, edge_type, seq_lengths, umask,
                                                  self.nodal_attention, self.avec)
                else:
                    emotions_a = []
                if 'v' in self.modals:
                    emotions_v = self.graph_net_v(features_v, edge_index, edge_norm, edge_type, seq_lengths, umask,
                                                  self.nodal_attention, self.avec)
                else:
                    emotions_v = []
                if 'l' in self.modals:
                    emotions_l = self.graph_net_l(features_l, edge_index, edge_norm, edge_type, seq_lengths, umask,
                                                  self.nodal_attention, self.avec)
                else:
                    emotions_l = []
                if self.att_type == 'concat_subsequently':
                    emotions = []
                    if len(emotions_a) != 0:
                        emotions.append(emotions_a)
                    if len(emotions_v) != 0:
                        emotions.append(emotions_v)
                    if len(emotions_l) != 0:
                        emotions.append(emotions_l)
                    emotions_feat = torch.cat(emotions, dim=-1)
                elif self.att_type == 'gated':
                    emotions_feat = self.gatedatt(emotions_a, emotions_v, emotions_l, self.modals)
                else:
                    print("There is no such attention mechnism")

                emotions_feat = self.dropout_(emotions_feat)
                log_prob = F.log_softmax(self.smax_fc(emotions_feat), 1)
        elif self.graph_type == 'GCN3' or self.graph_type == 'DeepGCN':
            if self.use_topic:
                topicLabel = []
            else:
                topicLabel = []
            if not self.multi_modal:
                log_prob = self.graph_net(features, seq_lengths, qmask)
            else:
                emotions_a = self.graph_net_a(features_a, seq_lengths, qmask) if 'a' in self.modals else []
                emotions_v = self.graph_net_v(features_v, seq_lengths, qmask) if 'v' in self.modals else []
                emotions_l = self.graph_net_l(features_l, seq_lengths, qmask) if 'l' in self.modals else []

                if self.att_type == 'concat_subsequently':
                    emotions = []
                    if len(emotions_a) != 0:
                        emotions.append(emotions_a)
                    if len(emotions_v) != 0:
                        emotions.append(emotions_v)
                    if len(emotions_l) != 0:
                        emotions.append(emotions_l)
                    emotions_feat = torch.cat(emotions, dim=-1)
                elif self.att_type == 'gated':
                    emotions_feat = self.gatedatt(emotions_a, emotions_v, emotions_l, self.modals)
                else:
                    print("There is no such attention mechnism")

                emotions_feat = self.dropout_(emotions_feat)
                emotions_feat = nn.ReLU()(emotions_feat)
                log_prob = F.log_softmax(self.smax_fc(emotions_feat), 1)
            """ graph_type=='MMGCN' 进入下边代码 graph_model（）输入为节点向量 features_a, features_v, features_l 
                tensor(sum_seq,200)    sum_seq 为这个batch中话语的真实数量 seq_lengths ->list:16 表示每段对话的话语数量
                qmask ->tensor(seq,batch,2) 说话人矩阵 独热向量表示
                输出为 features ->tensor(3*sum_seq,900) 将变换后的3种模态的特征拼接起来 赋给 emotions_feat
            """
        elif self.graph_type == 'MMGCN' or self.graph_type == 'MMGCN2':
            emotions_feat = self.graph_model(features_a, features_v, features_l, seq_lengths, qmask, global_sdj_den,front_sdj_den , back_sdj_den, s_mask, k1_0, k2_0, k3_0, k4_0, features_k1,features_k2,features_k3)  # emotions_feat ->tensor(sum_seq,300) 变换后的文本特征
            """
            这一步加上 情感胶囊论文中 最后分类使用的双向lstm
            第一步 将emotions_feat 从 (sum_seq,900) 变成 seq，batch，dim 进入双向lstm
            第二步 对得到的隐藏态进行整合 每句话由前隐藏态和后隐藏态组成
            """
            if self.args.use_lstm_classic:
                emotions_feat = batch_first(emotions_feat,seq_lengths)   ##batch seq dim
                emotions_feat = self.seqlstm(emotions_feat.transpose(0,1),umask)
                emotions_feat = adbatch_first(emotions_feat.transpose(1,0),seq_lengths)

            loss1 = 0
            prob = None
            emotions_feat = self.dropout_(emotions_feat)
            emotions_feat = nn.ReLU()(emotions_feat)
            log_prob = self.smax_fc(emotions_feat)
            """得到预测向量之后 是否要进入crf层 随机场
            """
            if self.args.use_crf_classic:
                sentences_mask = umask.transpose(0, 1)  #话语标记向量 变成seq batch
                sentences_mask = sentences_mask.byte()  #必须进行类型转换
                crf_emissions = batch_first(log_prob, seq_lengths).transpose(0, 1)  ## batch seq dim 再变成 seq,batch,dim
                loss1 = self.crf_classic(crf_emissions, label.transpose(0, 1), mask=sentences_mask)
                prob = self.crf_classic.decode(crf_emissions, mask=sentences_mask)  # 出来的是list(batch)  里边每个元素也都是list
                for i in range(len(prob)):
                    prob[i] = torch.tensor(prob[i])
                prob = torch.cat(prob, dim = 0)  # 预测标签


            log_prob = F.log_softmax(log_prob, 1)  # 进入分类线性层



        elif self.graph_type == 'None':
            if not self.multi_modal:
                h_ = self.graph_net(features)
                log_prob = F.log_softmax(h_, 1)
            else:
                emotions_a = self.graph_net_a(features_a) if 'a' in self.modals else []
                if type(emotions_a) != type([]):
                    emotions_a = torch.cat([emotions_a, features_a], dim=-1)
                emotions_v = self.graph_net_v(features_v) if 'v' in self.modals else []
                if type(emotions_v) != type([]):
                    emotions_v = torch.cat([emotions_v, features_v], dim=-1)
                emotions_l = self.graph_net_l(features_l) if 'l' in self.modals else []
                if type(emotions_l) != type([]):
                    emotions_l = torch.cat([emotions_l, features_l], dim=-1)

                if self.att_type == 'concat_subsequently':
                    emotions = []
                    if len(emotions_a) != 0:
                        emotions.append(emotions_a)
                    if len(emotions_v) != 0:
                        emotions.append(emotions_v)
                    if len(emotions_l) != 0:
                        emotions.append(emotions_l)
                    emotions_feat = torch.cat(emotions, dim=-1)
                elif self.att_type == 'gated':
                    emotions_feat = self.gatedatt(emotions_a, emotions_v, emotions_l, self.modals)
                else:
                    print("There is no such attention mechnism")

                emotions_feat = self.dropout_(emotions_feat)
                log_prob = F.log_softmax(self.smax_fc(emotions_feat), 1)
        else:
            print("There are no such kind of graph")


        return log_prob, edge_index, edge_norm, edge_type, edge_index_lengths, loss1, prob

# t
def simple_batch_graphify(features, lengths, no_cuda):
    """
        输入是 features ->tensor(seq,batch,200)单独模态的向量表示   lengths ->list:16 表示每段对话的话语数量
        返回 node_features = features_a 或 features_v 或 features_l ->tensor(sum_seq,200)
        sum_seq 为这个batch中话语的真实数量    相当于把这个batch中的三种模态的向量表示各自拼起来
        每种模态表示从形状 tensor(seq,batch,200) 变成形状 tensor(sum_seq,200)
    """
    edge_index, edge_norm, edge_type, node_features = [], [], [], []
    batch_size = features.size(1)

    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])

    node_features = torch.cat(node_features, dim=0)

    if not no_cuda:
        node_features = node_features.to(device)

    return node_features, None, None, None, None

def batch_first(feature,lengths):
    """
    feature ,  sum_seq,100
    lenghts   list    batch   每段对话的话语数量
    要返回新的feature  batch，seq，dim
    """
    lengths = torch.tensor(lengths)
    feature_list = []
    start_i = 0
    for i in range(len(lengths)):
        feature_list.append(feature[start_i:start_i+lengths[i],:])
        start_i += lengths[i]
    return pad_sequence(feature_list,batch_first=True)

def adbatch_first(batch_feature,lengths):
    """
    batch_feature,     batch，seq，dim
    lenghts       list    batch   每段对话的话语数量
    要返回     sum_seq,100
    """
    lengths = torch.tensor(lengths)
    feature_sum = []
    for i,fea in enumerate(batch_feature):
        feature_sum.append(fea[:lengths[i],:])

    return torch.cat(feature_sum,dim = 0)

class CNNEncoder(nn.Module):
    """
    卷积提取Encoder
    """
    def __init__(self, in_channels, out_channels, output_dim):
        """
        CNN Encoder:
            - in_channels: 词向量的维度
            - out_channels: 经过CNN编码后的维度
            - output_dim: 经过线性层后Phrase Encoder的最终维度
        input: (batch_size, max_seq_len, embedding_dim)
        output: (batch_size, output_dim)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels=out_channels
        self.output_dim=output_dim
        self.kernel_sizes = [1, 2]
        #self.kernel_num = 128
        self.convs=nn.ModuleList([nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=k) for k in self.kernel_sizes])
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(in_features=self.out_channels*len(self.kernel_sizes), out_features=self.output_dim)

    def forward(self, input):
        data = input
        # seq_len = input.size()[2]
        # 要变成 (batch_size, embedding_dim, max_seq_len)
        x = data.permute(1, 2, 0)
        # (batch_size, kernel_num, max_seq_len-kernel_size+1) * len(kernels)
        x = [F.relu(conv(x)) for conv in self.convs]
        # (batch_size, kernel_num) * len(kernels)
        x = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in x]
        # (batch_size, kernel_num * len(kernels))
        x = torch.cat(x, 1)
        x = self.dropout(x)
        # (batch_size, output_dim)
        # x = self.linear(x)
        out = F.normalize(x, p=2, dim=1)
        return out
# t
class Lstmhidden(nn.Module):
    def __init__(self,args):
        super(Lstmhidden, self).__init__()
        if args.use_know_graph and not args.only_text:
            self.seqlstm1 = nn.LSTMCell(900, 900)
            self.seqlstm2 = nn.LSTMCell(900, 900)
        else:
            self.seqlstm1 = nn.LSTMCell(300,300)
            self.seqlstm2 = nn.LSTMCell(300,300)

    def forward(self,emotions_feat,umask):
        """
        emotions_feat    seq  batch dim
        """
        seq = emotions_feat.size()[0]
        front_ever_hidden_list = []
        front_sum_hidden_list = []
        back_ever_hidden_list = []
        back_sum_hidden_list = []

        re_emotions_feat = reverse_seq(emotions_feat, umask)    #将batch中的序列反向 出来的是反序列的话语表示 seq  batch dim

        for i in range(seq):
            if i == 0:
                front_hidden = torch.zeros_like(emotions_feat[i,:,:])
                temp_front_hidden,temp_front_c = self.seqlstm1(emotions_feat[i,:,:],(front_hidden,front_hidden))  ## 第一句话 默认短期记忆和长期记忆都是0
                back_hidden = torch.zeros_like(re_emotions_feat[i, :, :])
                temp_back_hidden, temp_back_c = self.seqlstm2(re_emotions_feat[i, :, :],(back_hidden,back_hidden))  ## 最后一句话 默认短期记忆和长期记忆都是0

            if i != 0:
                front_hidden = front_hidden + temp_front_hidden
                temp_front_hidden, temp_front_c = self.seqlstm1(emotions_feat[i, :, :],(temp_front_hidden,temp_front_c))
                back_hidden = back_hidden + temp_back_hidden
                temp_back_hidden, temp_back_c = self.seqlstm2(re_emotions_feat[i, :, :],(temp_back_hidden, temp_back_c))


            front_ever_hidden_list.append(temp_front_hidden.unsqueeze(0))
            front_sum_hidden_list.append(front_hidden.unsqueeze(0))
            back_ever_hidden_list.append(temp_back_hidden.unsqueeze(0))
            back_sum_hidden_list.append(back_hidden.unsqueeze(0))

        front_sum_hidden = torch.cat(front_sum_hidden_list,dim = 0)  ##seq batch dim
        back_sum_hidden = torch.cat(back_sum_hidden_list,dim = 0)    ##seq batch dim
        back_sum_hidden = reverse_seq(back_sum_hidden, umask)


        return front_sum_hidden + back_sum_hidden

"""
将一个batch中的序列给反过来 输入是seq，batch，dim 和 batch seq 这个是标记某段对话是否有话语 有的话标志位1 没有就是0
"""

def reverse_seq(X, mask):
    """
    X -> seq_len, batch, dim
    mask -> batch, seq_len
    """
    X_ = X.transpose(0, 1)
    mask_sum = torch.sum(mask, 1).int()

    xfs = []
    for x, c in zip(X_, mask_sum):
        xf = torch.flip(x[:c], [0])
        xfs.append(xf)
    return pad_sequence(xfs)

"""
最后要加入的crf层
"""
class CRF(nn.Module):
    """Conditional random field.   条件随机场。

    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.

    本模块实现了一个条件随机场[LMP01]_。该类的正向计算计算给定标签序列和发射分数张量的对数似然。
    这个类还有' ~CRF.decode '方法，它使用' Viterbi算法' _找到给定一个发射分数张量的最佳标签序列。

    Args:
        num_tags: Number of tags.  标签的数量。
        batch_first: Whether the first dimension corresponds to the size of a minibatch. 第一个维度是否对应于小批量的大小。

    Attributes: 属性
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size  起始转变分数张量的大小
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size      结束转变分数张量的大小
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size       转变分数张量的大小
            ``(num_tags, num_tags)``.


    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.

    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))  # 全0的张量 形状是tensor（num_tags）  在这里num_tags = 7
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.global_transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.

        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.global_transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(self, emissions: torch.Tensor, tags: torch.LongTensor, mask: Optional[torch.ByteTensor] = None,
                reduction: str = 'sum') -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.  计算给定发射分数的标签序列的条件对数似然。

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.

        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
            对数似然。如果reduction是' none '，则为' size ' (batch_size，) ' '，否则为' '()' '。
        """
        self._validate(emissions, tags=tags, mask=mask)  # 判断张量的形状是否符合要求
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)
        """
        最终的形状应是 
        emissions (`~torch.Tensor`): (seq_length, batch_size, num_tags)
        tags (`~torch.LongTensor`):  (seq_length, batch_size)
        mask (`~torch.ByteTensor`):  (seq_length, batch_size)      
        """
        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)  # 在输入为预测标签的情况下 真实标签的得分 应越大越好
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)  # 输入为预测标签的情况下 别的情况的得分 应越小越好
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.float().sum()

    def decode(self, emissions: torch.Tensor, mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.

        Returns:
            List of list containing the best tag sequence for each batch.
        """
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _validate(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.LongTensor] = None,
            mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # speakers : (seq_length, batch_size)
        # last_turns: (seq_length, batch_size) last turn for the current speaker
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        # st_transitions = torch.softmax(self.start_transitions, -1)
        # ed_transitions = torch.softmax(self.end_transitions, -1)
        # transitions = torch.softmax(self.transitions, -1)
        # emissions = torch.softmax(emissions, -1)
        # personal_transitions = torch.softmax(self.personal_transitions, -1)
        st_transitions = self.start_transitions  # 全0张量 tensor（7）   7是类别数
        ed_transitions = self.end_transitions  # 全0张量 tensor（7）   7是类别数
        score = st_transitions[tags[0]]  # 得到所有batch第一句话的分数    形状是 1，batch_size
        score += emissions[0, torch.arange(batch_size), tags[0]]
        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            global_transitions = self.global_transitions[tags[i - 1], tags[i]]
            score += global_transitions * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)  得到的每段对话的最后一句话的位置
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,) 为了得到每段对话中最后一句话的真实标签 tensor（batch）
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += ed_transitions[last_tags]

        return score

    def _compute_normalizer(
            self, emissions: torch.Tensor,
            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)
        batch_size = emissions.size(1)

        st_transitions = self.start_transitions
        ed_transitions = self.end_transitions
        score = st_transitions + emissions[0]
        scores = []
        scores.append(score)
        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            global_transitions = self.global_transitions
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + global_transitions + broadcast_emissions

            next_score = torch.logsumexp(next_score, dim=1)

            score = torch.where(mask[i].unsqueeze(1), next_score, score)

            scores.append(score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += ed_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(
            self, emissions: torch.FloatTensor,
            mask: torch.ByteTensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        st_transitions = self.start_transitions
        ed_transitions = self.end_transitions
        score = st_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        scores = []
        scores.append(score)
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            global_transitions = self.global_transitions

            next_score = broadcast_score + global_transitions + broadcast_emissions

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            scores.append(score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += ed_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list

class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len
        target -> batch*seq_len
        mask -> batch*seq_len
        """
        loss = self.loss(pred*mask, target)/torch.sum(mask)
        return loss


class UnMaskedWeightedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(UnMaskedWeightedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        """
        if type(self.weight)==type(None):
            loss = self.loss(pred, target)
        else:
            loss = self.loss(pred, target)\
                            /torch.sum(self.weight[target])
        return loss


class SimpleAttention(nn.Module):

    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim,1,bias=False)

    def forward(self, M, x=None):
        """
        M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M)
        alpha = F.softmax(scale, dim=0).permute(1,2,0)
        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:]
        return attn_pool, alpha


class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type!='concat' or alpha_dim!=None
        assert att_type!='dot' or mem_dim==cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type=='general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type=='general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
        elif att_type=='concat':
            self.transform = nn.Linear(cand_dim+mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim) cand_dim == mem_dim?
        mask -> (batch, seq_len)
        """
        if type(mask)==type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type=='dot':
            M_ = M.permute(1,2,0)
            x_ = x.unsqueeze(1)
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)
        elif self.att_type=='general':
            M_ = M.permute(1,2,0)
            x_ = self.transform(x).unsqueeze(1)
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)
        elif self.att_type=='general2':
            M_ = M.permute(1,2,0)
            x_ = self.transform(x).unsqueeze(1)
            mask_ = mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2)
            M_ = M_ * mask_
            alpha_ = torch.bmm(x_, M_)*mask.unsqueeze(1)
            alpha_ = torch.tanh(alpha_)
            alpha_ = F.softmax(alpha_, dim=2)
            alpha_masked = alpha_*mask.unsqueeze(1)
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)
            alpha = alpha_masked/alpha_sum
        else:
            M_ = M.transpose(0,1)
            x_ = x.unsqueeze(1).expand(-1,M.size()[0],-1)
            M_x_ = torch.cat([M_,x_],2)
            mx_a = F.tanh(self.transform(M_x_))
            alpha = F.softmax(self.vector_prod(mx_a),1).transpose(1,2)

        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:]
        return attn_pool, alpha


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)
            score = torch.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=0)
        output = torch.bmm(score, kx)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)
        output = self.proj(output)
        output = self.dropout(output)
        return output, score


class DialogueRNNCell(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, listener_state=False,
                            context_attention='simple', D_a=100, dropout=0.5):
        super(DialogueRNNCell, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e

        self.listener_state = listener_state
        self.g_cell = nn.GRUCell(D_m+D_p,D_g)
        self.p_cell = nn.GRUCell(D_m+D_g,D_p)
        self.e_cell = nn.GRUCell(D_p,D_e)
        if listener_state:
            self.l_cell = nn.GRUCell(D_m+D_p,D_p)

        self.dropout = nn.Dropout(dropout)

        if context_attention=='simple':
            self.attention = SimpleAttention(D_g)
        else:
            self.attention = MatchingAttention(D_g, D_m, D_a, context_attention)

    def _select_parties(self, X, indices):
        q0_sel = []
        for idx, j in zip(indices, X):
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel,0)
        return q0_sel

    def forward(self, U, qmask, g_hist, q0, e0):
        """
        U -> batch, D_m
        qmask -> batch, party
        g_hist -> t-1, batch, D_g
        q0 -> batch, party, D_p
        e0 -> batch, self.D_e
        """
        qm_idx = torch.argmax(qmask, 1)
        q0_sel = self._select_parties(q0, qm_idx)

        g_ = self.g_cell(torch.cat([U,q0_sel], dim=1),
                torch.zeros(U.size()[0],self.D_g).type(U.type()) if g_hist.size()[0]==0 else
                g_hist[-1])
        g_ = self.dropout(g_)
        if g_hist.size()[0]==0:
            c_ = torch.zeros(U.size()[0],self.D_g).type(U.type())
            alpha = None
        else:
            c_, alpha = self.attention(g_hist,U)
        U_c_ = torch.cat([U,c_], dim=1).unsqueeze(1).expand(-1,qmask.size()[1],-1)
        qs_ = self.p_cell(U_c_.contiguous().view(-1,self.D_m+self.D_g),
                q0.view(-1, self.D_p)).view(U.size()[0],-1,self.D_p)
        qs_ = self.dropout(qs_)

        if self.listener_state:
            U_ = U.unsqueeze(1).expand(-1,qmask.size()[1],-1).contiguous().view(-1,self.D_m)
            ss_ = self._select_parties(qs_, qm_idx).unsqueeze(1).\
                    expand(-1,qmask.size()[1],-1).contiguous().view(-1,self.D_p)
            U_ss_ = torch.cat([U_,ss_],1)
            ql_ = self.l_cell(U_ss_,q0.view(-1, self.D_p)).view(U.size()[0],-1,self.D_p)
            ql_ = self.dropout(ql_)
        else:
            ql_ = q0
        qmask_ = qmask.unsqueeze(2)
        q_ = ql_*(1-qmask_) + qs_*qmask_
        e0 = torch.zeros(qmask.size()[0], self.D_e).type(U.type()) if e0.size()[0]==0\
                else e0
        e_ = self.e_cell(self._select_parties(q_,qm_idx), e0)
        e_ = self.dropout(e_)
        return g_,q_,e_,alpha


class DialogueRNN(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, listener_state=False,
                            context_attention='simple', D_a=100, dropout=0.5):
        super(DialogueRNN, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.dropout = nn.Dropout(dropout)

        self.dialogue_cell = DialogueRNNCell(D_m, D_g, D_p, D_e,
                            listener_state, context_attention, D_a, dropout)

    def forward(self, U, qmask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        g_hist = torch.zeros(0).type(U.type())
        q_ = torch.zeros(qmask.size()[1], qmask.size()[2],
                                    self.D_p).type(U.type())
        e_ = torch.zeros(0).type(U.type())
        e = e_

        alpha = []
        for u_,qmask_ in zip(U, qmask):
            g_, q_, e_, alpha_ = self.dialogue_cell(u_, qmask_, g_hist, q_, e_)
            g_hist = torch.cat([g_hist, g_.unsqueeze(0)],0)
            e = torch.cat([e, e_.unsqueeze(0)],0)
            if type(alpha_)!=type(None):
                alpha.append(alpha_[:,0,:])

        return e,alpha


class GRUModel(nn.Module):

    def __init__(self, D_m, D_e, D_h, n_classes=7, dropout=0.5):

        super(GRUModel, self).__init__()

        self.n_classes = n_classes
        self.dropout   = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        self.linear = nn.Linear(2*D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

    def forward(self, U, qmask, umask, att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions, hidden = self.gru(U)
        alpha, alpha_f, alpha_b = [], [], []

        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions,t,mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))

        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        return log_prob, alpha, alpha_f, alpha_b, emotions


class LSTMModel(nn.Module):

    def __init__(self, D_m, D_e, D_h, n_classes=7, dropout=0.5):

        super(LSTMModel, self).__init__()

        self.n_classes = n_classes
        self.dropout   = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        self.linear = nn.Linear(2*D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

    def forward(self, U, qmask, umask, att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions, hidden = self.lstm(U)
        alpha, alpha_f, alpha_b = [], [], []

        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions,t,mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))

        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        return log_prob, alpha, alpha_f, alpha_b, emotions


class DialogRNNModel(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, D_h, D_a=100, n_classes=7, listener_state=False,
        context_attention='simple', dropout_rec=0.5, dropout=0.5):

        super(DialogRNNModel, self).__init__()

        self.dropout   = nn.Dropout(dropout)
        self.dropout_rec = nn.Dropout(dropout+0.15)
        self.dialog_rnn_f = DialogueRNN(D_m, D_g, D_p, D_e,listener_state,
                                    context_attention, D_a, dropout_rec)
        self.dialog_rnn_r = DialogueRNN(D_m, D_g, D_p, D_e,listener_state,
                                    context_attention, D_a, dropout_rec)
        self.matchatt = MatchingAttention(2*D_e,2*D_e,att_type='general2')
        self.linear     = nn.Linear(2*D_e, D_h)
        self.smax_fc    = nn.Linear(D_h, n_classes)

    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)
        return pad_sequence(xfs)

    def forward(self, U, qmask, umask,att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions_f, alpha_f = self.dialog_rnn_f(U, qmask)
        emotions_f = self.dropout_rec(emotions_f)
        rev_U = self._reverse_seq(U, umask)
        rev_qmask = self._reverse_seq(qmask, umask)
        emotions_b, alpha_b = self.dialog_rnn_r(rev_U, rev_qmask)
        emotions_b = self._reverse_seq(emotions_b, umask)
        emotions_b = self.dropout_rec(emotions_b)
        emotions = torch.cat([emotions_f,emotions_b],dim=-1)
        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions,t,mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        return log_prob, alpha, alpha_f, alpha_b, emotions


class MaskedEdgeAttention(nn.Module):

    def __init__(self, input_dim, max_seq_len, no_cuda):
        """
        Method to compute the edge weights, as in Equation 1. in the paper.
        attn_type = 'attn1' refers to the equation in the paper.
        For slightly different attention mechanisms refer to attn_type = 'attn2' or attn_type = 'attn3'
        """

        super(MaskedEdgeAttention, self).__init__()

        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.scalar = nn.Linear(self.input_dim, self.max_seq_len, bias=False)
        self.matchatt = MatchingAttention(self.input_dim, self.input_dim, att_type='general2')
        self.simpleatt = SimpleAttention(self.input_dim)
        self.att = Attention(self.input_dim, score_function='mlp')
        self.no_cuda = no_cuda

    def forward(self, M, lengths, edge_ind):
        """
        M -> (seq_len, batch, vector)
        lengths -> length of the sequences in the batch
        edge_idn -> edge_idn是边的index的集合
        """
        attn_type = 'attn1'

        if attn_type == 'attn1':

            scale = self.scalar(M)
            alpha = F.softmax(scale, dim=0).permute(1, 2, 0)
            if not self.no_cuda:
                mask = Variable(torch.ones(alpha.size()) * 1e-10).detach().to(device)
                mask_copy = Variable(torch.zeros(alpha.size())).detach().to(device)

            else:
                mask = Variable(torch.ones(alpha.size()) * 1e-10).detach()
                mask_copy = Variable(torch.zeros(alpha.size())).detach()

            edge_ind_ = []
            for i, j in enumerate(edge_ind):
                for x in j:
                    edge_ind_.append([i, x[0], x[1]])

            edge_ind_ = np.array(edge_ind_).transpose()
            mask[edge_ind_] = 1
            mask_copy[edge_ind_] = 1
            masked_alpha = alpha * mask
            _sums = masked_alpha.sum(-1, keepdim=True)
            scores = masked_alpha.div(_sums) * mask_copy

            return scores

        elif attn_type == 'attn2':
            scores = torch.zeros(M.size(1), self.max_seq_len, self.max_seq_len, requires_grad=True)

            if not self.no_cuda:
                scores = scores.to(device)


            for j in range(M.size(1)):

                ei = np.array(edge_ind[j])

                for node in range(lengths[j]):

                    neighbour = ei[ei[:, 0] == node, 1]

                    M_ = M[neighbour, j, :].unsqueeze(1)
                    t = M[node, j, :].unsqueeze(0)
                    _, alpha_ = self.simpleatt(M_, t)
                    scores[j, node, neighbour] = alpha_

        elif attn_type == 'attn3':
            scores = torch.zeros(M.size(1), self.max_seq_len, self.max_seq_len, requires_grad=True)

            if not self.no_cuda:
                scores = scores.to(device)

            for j in range(M.size(1)):

                ei = np.array(edge_ind[j])

                for node in range(lengths[j]):

                    neighbour = ei[ei[:, 0] == node, 1]

                    M_ = M[neighbour, j, :].unsqueeze(1).transpose(0, 1)
                    t = M[node, j, :].unsqueeze(0).unsqueeze(0).repeat(len(neighbour), 1, 1).transpose(0, 1)
                    _, alpha_ = self.att(M_, t)
                    scores[j, node, neighbour] = alpha_[0, :, 0]

        return scores


def pad(tensor, length, no_cuda):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            if not no_cuda:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).to(device)])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var
    else:
        if length > tensor.size(0):
            if not no_cuda:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).to(device)])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor


def edge_perms(l, window_past, window_future):

    all_perms = set()
    array = np.arange(l)
    for j in range(l):
        perms = set()

        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:
            eff_array = array[:min(l, j+window_future+1)]
        elif window_future == -1:
            eff_array = array[max(0, j-window_past):]
        else:
            eff_array = array[max(0, j-window_past):min(l, j+window_future+1)]

        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)
    

def batch_graphify(features, qmask, lengths, window_past, window_future, edge_type_mapping, att_model, no_cuda):
    
    edge_index, edge_norm, edge_type, node_features = [], [], [], []
    batch_size = features.size(1)
    length_sum = 0
    edge_ind = []
    edge_index_lengths = [] 
    
    for j in range(batch_size):
        edge_ind.append(edge_perms(lengths[j], window_past, window_future))

    scores = att_model(features, lengths, edge_ind)

    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])
  
        perms1 = edge_perms(lengths[j], window_past, window_future)
        perms2 = [(item[0]+length_sum, item[1]+length_sum) for item in perms1]
        length_sum += lengths[j]

        edge_index_lengths.append(len(perms1))
    
        for item1, item2 in zip(perms1, perms2):
            edge_index.append(torch.tensor([item2[0], item2[1]]))
            edge_norm.append(scores[j, item1[0], item1[1]])
            speaker0 = (qmask[item1[0], j, :] == 1).nonzero()[0][0].tolist()
            speaker1 = (qmask[item1[1], j, :] == 1).nonzero()[0][0].tolist()
        
            if item1[0] < item1[1]:
                edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '0'])
            else:
                edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '1'])
    
    node_features = torch.cat(node_features, dim=0) 
    edge_index = torch.stack(edge_index).transpose(0, 1)
    edge_norm = torch.stack(edge_norm)
    edge_type = torch.tensor(edge_type)

    if not no_cuda:
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
        edge_norm = edge_norm.to(device)
        edge_type = edge_type.to(device)
    
    return node_features, edge_index, edge_norm, edge_type, edge_index_lengths 


def attentive_node_features(emotions, seq_lengths, umask, matchatt_layer, no_cuda):
    """
    Method to obtain attentive node features over the graph convoluted features, as in Equation 4, 5, 6. in the paper.
    """
    
    input_conversation_length = torch.tensor(seq_lengths)
    start_zero = input_conversation_length.data.new(1).zero_()
    
    if not no_cuda:
        input_conversation_length = input_conversation_length.to(device)
        start_zero = start_zero.to(device)

    max_len = max(seq_lengths)

    start = torch.cumsum(torch.cat((start_zero, input_conversation_length[:-1])), 0)

    emotions = torch.stack([pad(emotions.narrow(0, s, l), max_len, no_cuda) 
                                for s, l in zip(start.data.tolist(),
                                input_conversation_length.data.tolist())], 0).transpose(0, 1)


    alpha, alpha_f, alpha_b = [], [], []
    att_emotions = []

    for t in emotions:
        att_em, alpha_ = matchatt_layer(emotions, t, mask=umask)
        att_emotions.append(att_em.unsqueeze(0))
        alpha.append(alpha_[:,0,:])

    att_emotions = torch.cat(att_emotions, dim=0)

    return att_emotions


def classify_node_features(emotions, seq_lengths, umask, matchatt_layer, linear_layer, dropout_layer, smax_fc_layer, nodal_attn, avec, no_cuda):

    if nodal_attn:

        emotions = attentive_node_features(emotions, seq_lengths, umask, matchatt_layer, no_cuda)
        hidden = F.relu(linear_layer(emotions))
        hidden = dropout_layer(hidden)
        hidden = smax_fc_layer(hidden)

        if avec:
            return torch.cat([hidden[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])

        log_prob = F.log_softmax(hidden, 2)
        log_prob = torch.cat([log_prob[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])
        return log_prob

    else:

        hidden = F.relu(linear_layer(emotions))
        hidden = dropout_layer(hidden)
        hidden = smax_fc_layer(hidden)

        if avec:
            return hidden

        log_prob = F.log_softmax(hidden, 1)
        return log_prob


class GraphNetwork(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_relations, max_seq_len, hidden_size=64, dropout=0.5, no_cuda=False, use_GCN=False, return_feature=False):
        super(GraphNetwork, self).__init__()

        self.return_feature = return_feature
        self.no_cuda = no_cuda
        self.use_GCN = use_GCN
        self.conv1 = RGCNConv(num_features, hidden_size, num_relations, num_bases=30)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        if not self.return_feature:
            self.matchatt = MatchingAttention(num_features+hidden_size, num_features+hidden_size, att_type='general2')
            self.linear   = nn.Linear(num_features+hidden_size, hidden_size)
            self.dropout = nn.Dropout(dropout)
            self.smax_fc  = nn.Linear(hidden_size, num_classes)
        if self.use_GCN:
            self.conv3 = GCNLayer1(num_features, hidden_size, False) # index
            self.conv4 = GCNLayer1(hidden_size, hidden_size, False)
            self.linear = nn.Linear(num_features+hidden_size*2, hidden_size)
            self.matchatt = MatchingAttention(num_features+hidden_size*2, num_features+hidden_size*2, att_type='general2')

    def forward(self, x, edge_index, edge_norm, edge_type, seq_lengths, umask, nodal_attn, avec):
        if self.use_GCN:
            topicLabel = []
            out1 = self.conv1(x, edge_index, edge_type, edge_norm)
            out1 = self.conv2(out1, edge_index)
            out2 = self.conv3(x, seq_lengths, topicLabel)
            out2 = self.conv4(out2, seq_lengths, topicLabel)
            emotions = torch.cat([x,out1,out2],dim=-1)
            if self.return_feature:
                return emotions
            log_prob = classify_node_features(emotions, seq_lengths, umask, self.matchatt, self.linear, self.dropout, self.smax_fc, nodal_attn, avec, self.no_cuda)
        else:
            out = self.conv1(x, edge_index, edge_type, edge_norm)
            out = self.conv2(out, edge_index)
            emotions = torch.cat([x, out], dim=-1)
            if self.return_feature:
                return emotions
            log_prob = classify_node_features(emotions, seq_lengths, umask, self.matchatt, self.linear, self.dropout, self.smax_fc, nodal_attn, avec, self.no_cuda)
        return log_prob


class MMGatedAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, att_type='general'):
        super(MMGatedAttention, self).__init__()
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        self.dropouta = nn.Dropout(0.5)
        self.dropoutv = nn.Dropout(0.5)
        self.dropoutl = nn.Dropout(0.5)
        if att_type=='av_bg_fusion':
            self.transform_al = nn.Linear(mem_dim*2, cand_dim, bias=True)
            self.scalar_al = nn.Linear(mem_dim, cand_dim)
            self.transform_vl = nn.Linear(mem_dim*2, cand_dim, bias=True)
            self.scalar_vl = nn.Linear(mem_dim, cand_dim)
        elif att_type=='general':
            self.transform_l = nn.Linear(mem_dim, cand_dim, bias=True)
            self.transform_v = nn.Linear(mem_dim, cand_dim, bias=True)
            self.transform_a = nn.Linear(mem_dim, cand_dim, bias=True)
            self.transform_av = nn.Linear(mem_dim*3,1)
            self.transform_al = nn.Linear(mem_dim*3,1)
            self.transform_vl = nn.Linear(mem_dim*3,1)

    def forward(self, a, v, l, modals=None):
        a = self.dropouta(a) if len(a) !=0 else a
        v = self.dropoutv(v) if len(v) !=0 else v
        l = self.dropoutl(l) if len(l) !=0 else l
        if self.att_type == 'av_bg_fusion':
            if 'a' in modals:
                fal = torch.cat([a, l],dim=-1)
                Wa = torch.sigmoid(self.transform_al(fal))
                hma = Wa*(self.scalar_al(a))
            if 'v' in modals:
                fvl = torch.cat([v, l],dim=-1)
                Wv = torch.sigmoid(self.transform_vl(fvl))
                hmv = Wv*(self.scalar_vl(v))
            if len(modals) == 3:
                hmf = torch.cat([l,hma,hmv], dim=-1)
            elif 'a' in modals:
                hmf = torch.cat([l,hma], dim=-1)
            elif 'v' in modals:
                hmf = torch.cat([l,hmv], dim=-1)
            return hmf
        elif self.att_type == 'general':
            ha = torch.tanh(self.transform_a(a)) if 'a' in modals else a
            hv = torch.tanh(self.transform_v(v)) if 'v' in modals else v
            hl = torch.tanh(self.transform_l(l)) if 'l' in modals else l

            if 'a' in modals and 'v' in modals:
                z_av = torch.sigmoid(self.transform_av(torch.cat([a,v,a*v],dim=-1)))
                h_av = z_av*ha + (1-z_av)*hv
                if 'l' not in modals:
                    return h_av
            if 'a' in modals and 'l' in modals:
                z_al = torch.sigmoid(self.transform_al(torch.cat([a,l,a*l],dim=-1)))
                h_al = z_al*ha + (1-z_al)*hl
                if 'v' not in modals:
                    return h_al
            if 'v' in modals and 'l' in modals:
                z_vl = torch.sigmoid(self.transform_vl(torch.cat([v,l,v*l],dim=-1)))
                h_vl = z_vl*hv + (1-z_vl)*hl
                if 'a' not in modals:
                    return h_vl
            return torch.cat([h_av, h_al, h_vl],dim=-1)



class CNNFeatureExtractor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_size, filters, kernel_sizes, dropout):
        super(CNNFeatureExtractor, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embedding_dim, out_channels=filters, kernel_size=K) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * filters, output_size)
        self.feature_dim = output_size

    def init_pretrained_embeddings_from_numpy(self, pretrained_word_vectors):
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        # if is_static:
        self.embedding.weight.requires_grad = False

    def forward(self, x, umask):
        num_utt, batch, num_words = x.size()

        x = x.long()
        x = x.view(-1, num_words)  # (num_utt, batch, num_words) -> (num_utt * batch, num_words)
        emb = self.embedding(x)  # (num_utt * batch, num_words) -> (num_utt * batch, num_words, 300)
        emb = emb.transpose(-2,-1).contiguous()  # (num_utt * batch, num_words, 300)  -> (num_utt * batch, 300, num_words)

        convoluted = [F.relu(conv(emb)) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.size(2)).squeeze() for c in convoluted]
        concated = torch.cat(pooled, 1)
        features = F.relu(self.fc(self.dropout(concated)))  # (num_utt * batch, 150) -> (num_utt * batch, 100)
        features = features.view(num_utt, batch, -1)  # (num_utt * batch, 100) -> (num_utt, batch, 100)
        mask = umask.unsqueeze(-1).float()  # (batch, num_utt) -> (batch, num_utt, 1)
        mask = mask.transpose(0, 1)  # (batch, num_utt, 1) -> (num_utt, batch, 1)
        mask = mask.repeat(1, 1, self.feature_dim)  # (num_utt, batch, 1) -> (num_utt, batch, 100)
        features = (features * mask)  # (num_utt, batch, 100) -> (num_utt, batch, 100)

        return features


class DialogueGCN_DailyModel(nn.Module):
    def __init__(self, base_model, D_m, D_g, D_p, D_e, D_h, D_a, graph_hidden_size, n_speakers, max_seq_len,
                 window_past, window_future,
                 vocab_size, embedding_dim=100,
                 cnn_output_size=100, cnn_filters=50, cnn_kernel_sizes=(3, 4, 5), cnn_dropout=0.5,
                 n_classes=7, listener_state=False, context_attention='simple', dropout_rec=0.5, dropout=0.5,
                 nodal_attention=True, avec=False, no_cuda=False):

        super(DialogueGCN_DailyModel, self).__init__()
        self.cnn_feat_extractor = CNNFeatureExtractor(vocab_size, embedding_dim, cnn_output_size, cnn_filters,
                                                      cnn_kernel_sizes, cnn_dropout)
        self.base_model = base_model
        self.avec = avec
        self.no_cuda = no_cuda

        if self.base_model == 'DialogRNN':
            self.dialog_rnn_f = DialogueRNN(D_m, D_g, D_p, D_e, listener_state, context_attention, D_a, dropout_rec)
            self.dialog_rnn_r = DialogueRNN(D_m, D_g, D_p, D_e, listener_state, context_attention, D_a, dropout_rec)

        elif self.base_model == 'LSTM':
            self.lstm = nn.LSTM(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)

        elif self.base_model == 'GRU':
            self.gru = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)


        elif self.base_model == 'None':
            self.base_linear = nn.Linear(D_m, 2 * D_e)

        else:
            print('Base model must be one of DialogRNN/LSTM/GRU')
            raise NotImplementedError

        n_relations = 2 * n_speakers ** 2
        self.window_past = window_past
        self.window_future = window_future

        self.att_model = MaskedEdgeAttention(2 * D_e, max_seq_len, self.no_cuda)
        self.nodal_attention = nodal_attention

        self.graph_net = GraphNetwork(2 * D_e, n_classes, n_relations, max_seq_len, graph_hidden_size, dropout,
                                      self.no_cuda)

        edge_type_mapping = {}
        for j in range(n_speakers):
            for k in range(n_speakers):
                edge_type_mapping[str(j) + str(k) + '0'] = len(edge_type_mapping)
                edge_type_mapping[str(j) + str(k) + '1'] = len(edge_type_mapping)

        self.edge_type_mapping = edge_type_mapping

    def init_pretrained_embeddings(self, pretrained_word_vectors):
        self.cnn_feat_extractor.init_pretrained_embeddings_from_numpy(pretrained_word_vectors)

    def _reverse_seq(self, X, mask):
        X_ = X.transpose(0, 1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)

    def forward(self, input_seq, qmask, umask, seq_lengths):
        U = self.cnn_feat_extractor(input_seq, umask)

        if self.base_model == "DialogRNN":

            if self.avec:
                emotions, _ = self.dialog_rnn_f(U, qmask)

            else:
                emotions_f, alpha_f = self.dialog_rnn_f(U, qmask)
                rev_U = self._reverse_seq(U, umask)
                rev_qmask = self._reverse_seq(qmask, umask)
                emotions_b, alpha_b = self.dialog_rnn_r(rev_U, rev_qmask)
                emotions_b = self._reverse_seq(emotions_b, umask)
                emotions = torch.cat([emotions_f, emotions_b], dim=-1)

        elif self.base_model == 'LSTM':
            emotions, hidden = self.lstm(U)

        elif self.base_model == 'GRU':
            emotions, hidden = self.gru(U)

        elif self.base_model == 'None':
            emotions = self.base_linear(U)

        features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(emotions, qmask, seq_lengths,
                                                                                        self.window_past,
                                                                                        self.window_future,
                                                                                        self.edge_type_mapping,
                                                                                        self.att_model, self.no_cuda)
        log_prob = self.graph_net(features, edge_index, edge_norm, edge_type, seq_lengths, umask, self.nodal_attention,
                                  self.avec)

        return log_prob, edge_index, edge_norm, edge_type, edge_index_lengths