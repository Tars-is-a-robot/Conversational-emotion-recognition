import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv
from torch.nn.parameter import Parameter
import numpy as np, itertools, random, copy, math
import math
import scipy.sparse as sp
import ipdb
from torch.nn.utils.rnn import pad_sequence


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GCNLayer1(nn.Module):
    def __init__(self, in_feats, out_feats, use_topic=False, new_graph=True):
        super(GCNLayer1, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.use_topic = use_topic
        self.new_graph = new_graph

    def forward(self, inputs, dia_len, topicLabel):
        if self.new_graph:
            pdb.set_trace()
            adj = self.message_passing_directed_speaker(inputs,dia_len,topicLabel)
        else:
            adj = self.message_passing_wo_speaker(inputs,dia_len,topicLabel)
        x = torch.matmul(adj, inputs)
        x = self.linear(x)
        return x

    def cossim(self, x, y):
        a = torch.matmul(x, y)
        b = torch.sqrt(torch.matmul(x, x)) * torch.sqrt(torch.matmul(y, y))
        if b == 0:
            return 0
        else:
            return (a / b)

    def atom_calculate_edge_weight(self, x, y):
        f = self.cossim(x, y)
        if f >1 and f <1.05:
            f = 1
        elif f< -1 and f>-1.05:
            f = -1
        elif f>=1.05 or f<=-1.05:
            print('cos = {}'.format(f))
        return f

    def message_passing_wo_speaker(self, x,dia_len, topicLabel):
        adj = torch.zeros((x.shape[0], x.shape[0]))+torch.eye(x.shape[0])
        start = 0
        
        for i in range(len(dia_len)): #
            for j in range(dia_len[i]-1):
                for pin in range(dia_len[i] - 1-j):
                    xz=start+j
                    yz=xz+pin+1
                    f = self.cossim(x[xz],x[yz])
                    if f > 1 and f < 1.05:
                        f = 1
                    elif f < -1 and f > -1.05:
                        f = -1
                    elif f >= 1.05 or f <= -1.05:
                        print('cos = {}'.format(f))
                    Aij = 1 - math.acos(f) / math.pi
                    adj[xz][yz] = Aij
                    adj[yz][xz] = Aij
            start+=dia_len[i]

        if self.use_topic:
            for (index, topic_l) in enumerate(topicLabel):
                xz = index
                yz = x.shape[0] + topic_l - 7
                f = self.cossim(x[xz],x[yz])
                if f > 1 and f < 1.05:
                    f = 1
                elif f < -1 and f > -1.05:
                    f = -1
                elif f >= 1.05 or f <= -1.05:
                    print('cos = {}'.format(f))
                Aij = 1 - math.acos(f) / math.pi
                adj[xz][yz] = Aij
                adj[yz][xz] = Aij

        d = adj.sum(1)
        D=torch.diag(torch.pow(d,-0.5))
        adj = D.mm(adj).mm(D).to(device)

        return adj

    def message_passing_directed_speaker(self, x, dia_len, qmask):
        total_len = sum(dia_len)
        adj = torch.zeros((total_len, total_len))+torch.eye(total_len)
        start = 0
        use_utterance_edge=False
        for (i, len_) in enumerate(dia_len):
            speaker0 = []
            speaker1 = []
            for (j, speaker) in enumerate(qmask[start:start+len_]):
                if speaker[0] == 1:
                    speaker0.append(j)
                else:
                    speaker1.append(j)
            if use_utterance_edge:
                for j in range(len_-1):
                    f = self.atom_calculate_edge_weight(x[start+j], x[start+j+1])
                    Aij = 1-math.acos(f) / math.pi
                    adj[start+j][start+j+1] = Aij
                    adj[start+j+1][start+j] = Aij
            for k in range(len(speaker0)-1):
                f = self.atom_calculate_edge_weight(x[start+speaker0[k]], x[start+speaker0[k+1]])
                Aij = 1-math.acos(f) / math.pi
                adj[start+speaker0[k]][start+speaker0[k+1]] = Aij
                adj[start+speaker0[k+1]][start+speaker0[k]] = Aij
            for k in range(len(speaker1)-1):
                f = self.atom_calculate_edge_weight(x[start+speaker1[k]], x[start+speaker1[k+1]])
                Aij = 1-math.acos(f) / math.pi
                adj[start+speaker1[k]][start+speaker1[k+1]] = Aij
                adj[start+speaker1[k+1]][start+speaker1[k]] = Aij

            start+=dia_len[i]
        
        return adj


class GCN_2Layers(nn.Module):
    def __init__(self, lstm_hid_size, gcn_hid_dim, num_class, dropout, use_topic=False, use_residue=True, return_feature=False):
        super(GCN_2Layers, self).__init__()

        self.lstm_hid_size = lstm_hid_size
        self.gcn_hid_dim = gcn_hid_dim
        self.num_class = num_class
        self.dropout = dropout
        self.use_topic = use_topic
        self.return_feature = return_feature

        self.gcn1 = GCNLayer1(self.lstm_hid_size, self.gcn_hid_dim, self.use_topic)
        self.use_residue = use_residue
        if self.use_residue:
            self.gcn2 = GCNLayer1(self.gcn_hid_dim, self.gcn_hid_dim, self.use_topic)
            self.linear = nn.Linear(self.lstm_hid_size+self.gcn_hid_dim,self.num_class)
        else:
            self.gcn2 = GCNLayer1(self.gcn_hid_dim, self.num_class, self.use_topic)

    def forward(self, x,dia_len,topicLabel):
        x_graph = self.gcn1(x,dia_len,topicLabel)
        if not self.use_residue:
            x = self.gcn2(x_graph,dia_len,topicLabel)
            if self.return_feature:
                print("Error, you should change the state of use_residue")
        else:
            x_graph = self.gcn2(x_graph,dia_len,topicLabel)
            x = torch.cat([x,x_graph],dim=-1)
            if self.return_feature:
                return x
            x = self.linear(x)
        log_prob = F.log_softmax(x, 1)

        return log_prob

# t
class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))   ##Parameter() 让里边这个矩阵变成可训练的
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
    """输入是
        input  ->tensor(sum_seq,100) 这个batch中三种模态的向量表示
        adj,   ->tensor(sum_seq,sum_seq) 为整个batch的权重矩阵
        h0,    ->tensor(sum_seq,100) 这个batch中三种模态的初始向量表示 （在循环中一直保持不变）
        lamda, -> 0.5
        alpha, -> 0.1
        l      ->循环次数
        输出是 output ->tensor(sum_seq,100) 
    """
    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1) # 让theta = e的lamda/l+1次方
        hi = torch.spmm(adj, input) # 矩阵乘法 hi ->tensor(sum_seq,100)
        if self.variant: #Ture
            support = torch.cat([hi,h0],1) # support ->tensor(sum_seq,200)
            r = (1-alpha)*hi+alpha*h0 # r ->tensor(sum_seq,100)
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r # output ->tensor(sum_seq,100)
        if self.residual: # False
            output = output+input
        return output




"""
    用于在顺序注意过程中 根据话语对知识进行修改后的知识表示
"""
class Knowmix(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dim = args.hidden_dim   ##特征维度 300
        self.know_dim = args.know_dim  ##知识维度 150
        self.dropout = args.dropout_k
        self.know_mode_gate = args.know_mode_gate
        self.hr_liner = nn.Linear(self.dim + self.know_dim * 4,self.dim)  ##最后再变回输入时形状
        if self.know_mode_gate == 'extend' or self.know_mode_gate == 'dual':  ##哪种全局门方式
            self.init_text_projection = nn.Linear(self.dim, self.dim)  ##一个线性层 变文本向量维度
            self.init_weight_func = nn.Sequential(
                nn.Linear(self.dim + self.know_dim, self.know_dim),  ## 300 到 150
                nn.ELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.know_dim, 1)
            )  ##几个神经网络的组合
        elif self.know_mode_gate == 'grucell':  ##如果用gru单元来进行知识选择
            grus_kn1 = []
            for _ in range(args.gnn_layers):
                grus_kn1 += [nn.GRUCell(self.dim, self.know_dim)]
            self.grus_kn1 = nn.ModuleList(grus_kn1)

            grus_kn2 = []
            for _ in range(args.gnn_layers):
                grus_kn2 += [nn.GRUCell(self.dim, self.know_dim)]
            self.grus_kn2 = nn.ModuleList(grus_kn2)

            grus_kn3 = []
            for _ in range(args.gnn_layers):
                grus_kn3 += [nn.GRUCell(self.dim, self.know_dim)]
            self.grus_kn3 = nn.ModuleList(grus_kn3)

            grus_kn4 = []
            for _ in range(args.gnn_layers):
                grus_kn4 += [nn.GRUCell(self.dim, self.know_dim)]
            self.grus_kn4 = nn.ModuleList(grus_kn4)



    def forward(self,htext,hk1,hk2,hk3,hk4,l):
        """
            htext,            batch,dim
            hk1,nk2,hk3,hk4   batch,dim     这里是旧知识
        """
        global_kn = [hk1,hk2,hk3,hk4]  # 4种旧知识
        if self.know_mode_gate == 'extend' or self.know_mode_gate == 'dual':
            know_weights = []  ##知识权重门列表
            global_knr = []       # 存放修改后的知识表示
            for i,temp_k in enumerate(global_kn):
                if self.know_mode_gate == 'extend' or self.know_mode_gate == 'dual':  ##文本门的计算
                    htext = self.init_text_projection(htext)  # 把文本维度变300
                    logit = self.init_weight_func(torch.cat([htext, temp_k], dim=-1))
                    # logit = torch.sigmoid(logit)
                    know_weights.append(logit)
            know_weights = torch.cat(know_weights,dim = 1)
            know_weights = F.softmax(know_weights,dim = 1)  ## 得到知识权重

            for i, know_ in enumerate(global_kn):
                global_knr.append(know_weights[:, i].unsqueeze(1) * know_)   ##把知识与权重相乘后放到list中 每个元素都是batch，dim（300）
            global_knr1 = global_knr[0]
            global_knr2 = global_knr[1]
            global_knr3 = global_knr[2]
            global_knr4 = global_knr[3]
            global_knr = torch.cat(global_knr, dim=1)  # 150*4 修改后的4个知识 batch，dim
            return global_knr1, global_knr2, global_knr3, global_knr4, global_knr

        elif self.know_mode_gate == 'grucell':
            global_knr1 = self.grus_kn1[l](htext, hk1)
            global_knr2 = self.grus_kn2[l](htext, hk2)
            global_knr3 = self.grus_kn3[l](htext, hk3)
            global_knr4 = self.grus_kn4[l](htext, hk4)
            global_knr = torch.cat([global_knr1, global_knr2, global_knr3, global_knr4], dim=1)
            return global_knr1, global_knr2, global_knr3, global_knr4, global_knr  # 更新后的4个知识表示 batch dim（150）    global_knr 把更新后的知识拼起来
        else:
            return hk1,hk2,hk3,hk4,torch.cat(global_kn,dim=-1)


# t
class GCNII_lyc(nn.Module):
    def __init__(self, args, nhidden, nclass, variant, return_feature, new_graph=False):
        super(GCNII_lyc, self).__init__()
        self.args = args
        self.return_feature = return_feature
        self.use_residue = args.use_residue
        self.new_graph = new_graph
        self.convs = nn.ModuleList()
        for _ in range(args.Deep_GCN_nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant))

        self.ordert = Ordernet(args)  ##添加的顺序层

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(args.graph_fusion_dim, nhidden))
        if not return_feature:
            self.fcs.append(nn.Linear(args.graph_fusion_dim + nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = args.dropout
        self.alpha = args.alpha
        self.lamda = args.lamda

    def cossim(self, x, y):
        a = torch.matmul(x, y)
        b = torch.sqrt(torch.matmul(x, x)) * torch.sqrt(torch.matmul(y, y))
        if b == 0:
            return 0
        else:
            return (a / b)

    def forward(self, x, dia_len, topicLabel, front_sdj_den , back_sdj_den, global_sdj_den, s_mask, k1_0, k2_0, k3_0, k4_0, adj = None):
        """输入 x,       ->tensor(3*sum_seq,200) 这个batch中三种模态的向量表示    或者 ->tensor(sum_seq,200)   单独文本的表示
            dia_len,     list 16
            topicLabel,  ->tensor(sum_seq,2) 说话人矩阵 独热向量表示
            adj ,        ->tensor(sum_seq,sum_seq) 为整个batch的权重矩阵
            输出
            layer_inner  ->tensor(3*sum_seq,300) 这个batch中三种模态的向量表示 维度变了
            各层图卷积的计算过程
        """
        if adj is None:
            if self.new_graph:
                adj = self.message_passing_relation_graph(x, dia_len)
            else:
                adj = self.message_passing_wo_speaker(x, dia_len, topicLabel)
        else:
            adj = adj  # 让注意矩阵变成文本注意矩阵   根据进入的features来选择使用哪个权重矩阵  是将adj 用adj定义还是adj_l定义

        sum_seq = sum(dia_len)

        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)  # 使用时一定要加training=self.training   这时进入图卷积之前 x 的维度还是200
        layer_inner = self.act_fn(self.fcs[0](x))  # 线性层 layer_inner ->tensor(sum_seq,100) 这个batch中三种模态的向量表示  x维度从200变到100
        _layers.append(layer_inner)  # 把这个张量放到list里

        """ 一共要进行64层循环 其con是 上边的类 class GraphConvolution(nn.Module)
            它的输入是 
            layer_inner  ->tensor(sum_seq,100) 这个batch中三种模态的向量表示 每一层都更新
            adj,         ->tensor(sum_seq,sum_seq) 为整个batch的权重矩阵
            _layers[0],  ->tensor(sum_seq,100) 这个batch中三种模态的向量表示 （在循环中一直保持不变）
            self.lamda,  -> 0.5
            self.alpha,  -> 0.1
            i+1         ->当前循环次数 图的层数
            输出是 layer_inner ->tensor(sum_seq,300) 这个batch中文本的向量表示 或者  tensor(3*sum_seq,300)   文本加知识表示
        """
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1))

            if self.args.use_order:  ##如果要是用顺序注意 进入
                if (i + 1) % 16 == 0 or (i+1)==self.args.Deep_GCN_nlayers : # 什么时候插入顺序注意
                    if self.args.use_know_graph: # 如果知识参与了图卷积
                        """
                        在这里 如果上一步使用的是知识加文本的方式进行图融合 那么进到顺序注意的应当是 layer_inner 的文本部分 即最后的一部分切片
                        并在最后替换掉原始切片 进入到图融合网络
                        """
                        text_layer_inner = layer_inner[sum_seq*2:,:]
                        text_layer_inner = self.ordert(text_layer_inner, dia_len, front_sdj_den, back_sdj_den, global_sdj_den, s_mask, k1_0, k2_0, k3_0, k4_0)

                        know1_layer_inner = layer_inner[sum_seq:sum_seq * 2, :]
                        # # know1_layer_inner = self.ordert(know1_layer_inner, dia_len, front_sdj_den, back_sdj_den,global_sdj_den, s_mask, k1_0, k2_0, k3_0, k4_0)
                        know2_layer_inner = layer_inner[:sum_seq, :]
                        # know2_layer_inner = self.ordert(know2_layer_inner, dia_len, front_sdj_den, back_sdj_den,global_sdj_den, s_mask, k1_0, k2_0, k3_0, k4_0)

                        layer_inner = torch.cat([know2_layer_inner,know1_layer_inner,text_layer_inner],dim = 0)
                    else:
                        layer_inner = self.ordert(layer_inner, dia_len,front_sdj_den, back_sdj_den, global_sdj_den, s_mask, k1_0, k2_0, k3_0, k4_0)

        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)  # layer_inner ->tensor(sum_seq,100) 这个batch中文本的向量表示

        if self.use_residue:  # Ture 进入
            layer_inner = torch.cat([x, layer_inner], dim=-1)  # layer_inner ->tensor(sum_seq,300) 图卷积后得到的向量和最初始的向量拼接起来
        if not self.return_feature:  # Ture 不执行
            layer_inner = self.fcs[-1](layer_inner)
            layer_inner = F.log_softmax(layer_inner, dim=1)
        return layer_inner

    def message_passing_wo_speaker(self, x, dia_len, topicLabel):
        adj = torch.zeros((x.shape[0], x.shape[0]))
        start = 0
        for i in range(len(dia_len)):
            sub_adj = torch.zeros((dia_len[i], dia_len[i]))
            temp = x[start:start + dia_len[i]]
            vec_length = torch.sqrt(torch.sum(temp.mul(temp), dim=1))
            norm_temp = (temp.permute(1, 0) / vec_length)
            cos_sim_matrix = torch.sum(torch.matmul(norm_temp.unsqueeze(2), norm_temp.unsqueeze(1)), dim=0)
            cos_sim_matrix = cos_sim_matrix * 0.99999
            sim_matrix = torch.acos(cos_sim_matrix)

            d = sim_matrix.sum(1)
            D = torch.diag(torch.pow(d, -0.5))

            sub_adj[:dia_len[i], :dia_len[i]] = D.mm(sim_matrix).mm(D)
            adj[start:start + dia_len[i], start:start + dia_len[i]] = sub_adj
            start += dia_len[i]

        adj = adj.to(device)

        return adj

    def atom_calculate_edge_weight(self, x, y):
        f = self.cossim(x, y)
        if f > 1 and f < 1.05:
            f = 1
        elif f < -1 and f > -1.05:
            f = -1
        elif f >= 1.05 or f <= -1.05:
            print('cos = {}'.format(f))
        return f

    def message_passing_directed_speaker(self, x, dia_len, qmask):
        total_len = sum(dia_len)
        adj = torch.zeros((total_len, total_len)) + torch.eye(total_len)
        start = 0
        use_utterance_edge = False
        for (i, len_) in enumerate(dia_len):
            speaker0 = []
            speaker1 = []
            for (j, speaker) in enumerate(qmask[i][0:len_]):
                if speaker[0] == 1:
                    speaker0.append(j)
                else:
                    speaker1.append(j)
            if use_utterance_edge:
                for j in range(len_ - 1):
                    f = self.atom_calculate_edge_weight(x[start + j], x[start + j + 1])
                    Aij = 1 - math.acos(f) / math.pi
                    adj[start + j][start + j + 1] = Aij
                    adj[start + j + 1][start + j] = Aij
            for k in range(len(speaker0) - 1):
                f = self.atom_calculate_edge_weight(x[start + speaker0[k]], x[start + speaker0[k + 1]])
                Aij = 1 - math.acos(f) / math.pi
                adj[start + speaker0[k]][start + speaker0[k + 1]] = Aij
                adj[start + speaker0[k + 1]][start + speaker0[k]] = Aij
            for k in range(len(speaker1) - 1):
                f = self.atom_calculate_edge_weight(x[start + speaker1[k]], x[start + speaker1[k + 1]])
                Aij = 1 - math.acos(f) / math.pi
                adj[start + speaker1[k]][start + speaker1[k + 1]] = Aij
                adj[start + speaker1[k + 1]][start + speaker1[k]] = Aij

            start += dia_len[i]

        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D).to(device)

        return adj.to(device)

    def message_passing_relation_graph(self, x, dia_len):
        total_len = sum(dia_len)
        adj = torch.zeros((total_len, total_len)) + torch.eye(total_len)
        window_size = 10
        start = 0
        for (i, len_) in enumerate(dia_len):
            edge_set = []
            for k in range(len_):
                left = max(0, k - window_size)
                right = min(len_ - 1, k + window_size)
                edge_set = edge_set + [str(i) + '_' + str(j) for i in range(left, right) for j in
                                       range(i + 1, right + 1)]
            edge_set = [[start + int(str_.split('_')[0]), start + int(str_.split('_')[1])] for str_ in
                        list(set(edge_set))]
            for left, right in edge_set:
                f = self.atom_calculate_edge_weight(x[left], x[right])
                Aij = 1 - math.acos(f) / math.pi
                adj[left][right] = Aij
                adj[right][left] = Aij
            start += dia_len[i]

        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D).to(device)

        return adj.to(device)




#t   插入的话语顺序注意
class Ordernet(nn.Module):
    def __init__(self,args):
        super(Ordernet, self).__init__()
        self.lstm1 = nn.LSTM(args.hidden_dim,args.hidden_dim)
        self.lstm2 = nn.LSTM(args.hidden_dim,args.hidden_dim)
        self.fc1 = nn.Linear(args.hidden_dim,args.hidden_dim)
        self.args = args
        # self.know_text = Knowmix(args)  # 知识选择类

        # self.know_text = Knowmix(args)
        # self.args = args
        # # gcn layer
        # self.dropout = nn.Dropout(args.dropout)
        # self.gnn_layers = args.gnn_layers
        # if not args.no_rel_attn:
        #     self.rel_attn = True
        # else:
        #     self.rel_attn = False

        """ 
        attn_type 表示顺序上采用什么注意方式 一共三种方式
        """
        if self.args.attn_type == 'linear':
            gats = []
            for _ in range(args.gnn_layers):
                gats += [GatLinear(args.hidden_dim) if args.no_rel_attn else GatLinear_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)
        elif self.args.attn_type == 'mgcn':
            gats = []
            for _ in range(args.gnn_layers):
                gats += [GatDot(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)
        elif self.args.attn_type == 'rgcn':
            gats = []
            for _ in range(args.gnn_layers):
                gats += [GAT_dialoggcn_v1(args)]
            self.gather = nn.ModuleList(gats)
        """
        这几个GRU层按需使用        
        """
        grus_c1 = []
        for _ in range(args.gnn_layers):
            grus_c1 += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_c1 = nn.ModuleList(grus_c1)
        grus_p1 = []
        for _ in range(args.gnn_layers):
            grus_p1 += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_p1 = nn.ModuleList(grus_p1)  ## 前向注意的两个gru
        #
        grus_c2 = []
        for _ in range(args.gnn_layers):
            grus_c2 += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_c2 = nn.ModuleList(grus_c2)  ##

        grus_p2 = []
        for _ in range(args.gnn_layers):
            grus_p2 += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_p2 = nn.ModuleList(grus_p2)  ##  后向注意的两个gru
        #
        # grus_c3 = []
        # for _ in range(args.gnn_layers):
        #     grus_c3 += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        # self.grus_c3 = nn.ModuleList(grus_c3)  ## 远程gru-c
        #
        # grus_p3 = []
        # for _ in range(args.gnn_layers):
        #     grus_p3 += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        # self.grus_p3 = nn.ModuleList(grus_p3)  ## 远程gru-p
        #
        fcs = []
        for _ in range(args.gnn_layers):
            fcs += [nn.Linear(args.hidden_dim * 2, args.hidden_dim)]
        self.fcs = nn.ModuleList(fcs)

        # """
        # 这个是分类前的线性层 在这里先不使用
        # """
        # in_dim = args.hidden_dim * (args.gnn_layers + 1) + 1024 + 600  # + args.emb_dim
        # output mlp layers
        # layers = [nn.Linear(in_dim, args.hidden_dim), nn.ReLU()]
        # for _ in range(args.mlp_layers - 1):
        #     layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        # layers += [self.dropout]
        # layers += [nn.Linear(args.hidden_dim, num_class)]
        # self.out_mlp = nn.Sequential(*layers)
        # self.norm_kn_text = nn.BatchNorm1d(args.hidden_dim * (args.gnn_layers + 1) + 600)
        # self.again_text = MMgcn1(600, 300)

    def forward(self,feature,lengths,front_sdj_den = None, back_sdj_den= None, global_sdj_den= None,s_mask= None,k1_0= None, k2_0= None, k3_0= None, k4_0= None):
        """
        feature,  3*sum_seq,100
        lenghts,  batch    每段对话中具体有多少句话
        global_sdj_den,   全局注意话语
        s_mask,     batch,seq,seq
        k1_0, k2_0, k3_0, k4_0  seq,batch,dim
        实现话语间的顺序注意
        第一步 将GCNII_lyc传过来feature（实际应为 sum_seq,100） 变成batch,seq,dim 的形式
        第二步就可以直接使用有向无环图中的方法进行融合
        第三步再将feature变回去 送到GCNII_lyc

        """
        """
        预处理的部分 将知识变成batch，seq，dim的形式 
        把lenghts变成张量形式 方便后边使用
        将feature（sum_seq）变成batch，seq，dim形式
        """
        k1_0 = k1_0.transpose(0, 1)
        k2_0 = k2_0.transpose(0, 1)
        k3_0 = k3_0.transpose(0, 1)
        k4_0 = k4_0.transpose(0, 1)
        lengths = torch.tensor(lengths)
        features = self.batch_first(feature,lengths)     ##将feature变成batch seq dim

        num_utter = features.size()[1]  ##int  seq  得到seq后就是要
        H0 = F.relu(self.fc1(features))  ##也可不加 在这里进来的维度是100
        H = [H0]   # 将传过来的原始特征 存起来
        HK1 = [k1_0]
        HK2 = [k2_0]
        HK3 = [k3_0]
        HK4 = [k4_0]  ## 四种知识初始表示与文本特征传过来的初始表示

        """  gnn_layers = 1 gcn层数  C：batch,1,dim(100)    M:batch,dim(100)   P:batch,1,dim(100)
             至H1 = C+P 这一步 只在计算图神经网路的第一个节点 下一层循环是在计算剩余节点
             self.args.attn_type == 'rgcn' 节点间注意方式   经过计算后 H1不断拼接 变成batch,seq,dim(100)
             添加新batch   Hr 表示本层图中第i句话的表示
        """

        for l in range(self.args.gnn_layers):
            for i in range(0, num_utter):
                """
                根据attn_type的不同  分为几种图计算方式

                """
                if i == 0:
                    H1 = None
                Hr, Hr1, H_front, H_back, front_sdjr_den, back_sdjr_den,front_s_maskr, back_s_maskr,\
                front_k1, front_k2, front_k3, front_k4,back_k1, back_k2, back_k3, back_k4 = self.batch_function(self.args.batch_integration, lengths, H, l, front_sdj_den,back_sdj_den, s_mask, i, HK1, HK2, HK3, HK4, H1 )      #进行batch的整合


                if self.args.attn_type == 'rgcn':
                    front_M,back_M = self.gather[l](i,num_utter, Hr, Hr1, Hr1, front_sdjr_den, front_s_maskr,front_k1, front_k2, front_k3, front_k4,
                                           H_back,H_back,back_sdjr_den,back_s_maskr,back_k1, back_k2, back_k3, back_k4)  ## 得到近程，中程，远程的话语向量 batch，dim（300）
                elif self.args.attn_type == 'mgcn':
                    front_M, back_M = self.gather[l](i, num_utter, Hr, H_front, H_front, front_sdjr_den, front_s_maskr,
                                                     front_k1, front_k2, front_k3, front_k4,
                                                     H_back, H_back, back_sdjr_den, back_s_maskr, back_k1, back_k2,
                                                     back_k3, back_k4)
                else:
                    _, M = self.gather[l](H[l][:, i, :], H1, H1, s_mask[:, i, :i])
                """
                如果用了batch重整合 那就batch_integration 为Ture 没有用整合 就直接节进到else 选项
                """
                if self.args.batch_integration:
                    C = self.grus_c[l](Hr, M)
                    P = self.grus_p[l](M, Hr)
                    H_temp = C + P  # 新batch,dim
                    if H_temp.size()[0] < features.size()[0]:  ##要让新batch与原batch对应起来
                        temp = torch.zeros(1, H_temp.size()[1])
                        for j in range(lengths.size()[0]):
                            if i > lengths[j] - 1:
                                H_temp = torch.cat((H_temp[:j, :], temp, H_temp[j:, :]), dim=0)
                    H1 = torch.cat((H1, H_temp.unsqueeze(1)), dim=1)
                else:
                    C1 = self.grus_c1[l](Hr, front_M)
                    P1 = self.grus_p1[l](front_M, Hr)
                    H_front = C1 + P1  ## 前向注意计算  形状都是batch，dim

                    C2 = self.grus_c2[l](Hr, back_M)
                    P2 = self.grus_p2[l](back_M, Hr)
                    H_back = C2 + P2

                    H_temp = 0.6*H_front + 0.4*H_back
                    """
                    得到的本层本句话的信息 进行处理
                    """
                    # temp_k1, temp_k2, temp_k3, temp_k4, global_knr = self.know_text(H_temp, old_k1, old_k2, old_k3, old_k4,l)  ## 根据话语得到修改后的知识表示  batch,150
                    if i == 0:
                        H1 = H_temp.unsqueeze(1)
                    else:
                        H1 = torch.cat((H1, H_temp.unsqueeze(1)), dim=1)  ## H1 batch,i,dim      H_temp   batch,1,dim
                    # k1_i = torch.cat((k1_i, temp_k1.unsqueeze(1)), dim=1)
                    # k2_i = torch.cat((k2_i, temp_k2.unsqueeze(1)), dim=1)
                    # k3_i = torch.cat((k3_i, temp_k3.unsqueeze(1)), dim=1)
                    # k4_i = torch.cat((k4_i, temp_k4.unsqueeze(1)), dim=1)
            H.append(H1)   ## 包含了经过顺序建模后的每层的特征
            # HK1.append(k1_i)
            # HK2.append(k2_i)
            # HK3.append(k3_i)
            # HK4.append(k4_i)












        features = H[self.args.gnn_layers]


        features = self.adbatch_first(features,lengths)

        return features

    """
    用于实现的开始 将图卷积后的传过来的文本张量转为顺序注意时需要的张量形状
    """
    def batch_first(self,feature,lengths):
        """
        feature ,  sum_seq,100
        lenghts   list    batch   每段对话的话语数量
        要返回新的feature  batch，seq，dim
        """
        feature_list = []
        start_i = 0
        for i in range(len(lengths)):
            feature_list.append(feature[start_i:start_i+lengths[i],:])
            start_i += lengths[i]
        return pad_sequence(feature_list,batch_first=True)

    """
        用于实现的结束 将顺序注意完成之后的张量形状转为下一步图卷积需要的张量形状
    """
    def adbatch_first(self,batch_feature,lengths):
        """
        batch_feature,     batch，seq，dim
        lenghts       list    batch   每段对话的话语数量
        要返回     sum_seq,100
        """
        feature_sum = []
        for i,fea in enumerate(batch_feature):
            feature_sum.append(fea[:lengths[i],:])

        return torch.cat(feature_sum,dim = 0)
    """
    用于在顺序注意过程中的batch重整 可要可不要 提升不大
    """
    def batch_function(self,batch_integration, lengths, H, l, front_sdj_den, back_sdj_den, s_mask, i, HK1, HK2, HK3, HK4, H1):
        """
        batch的重整合 相当于就是一次只取出某个batch的一句话来训练 取出所有batch的第i句话
        返回的是这句话的本层表示 这句话本层之前的话语 这句话上层之后的话语 这句话前要注意的句子 这句话后要注意的句子
        这句话前他说的话 这句话后他说的话 这句话前的那些话语的知识 这句话后的那些话语的知识
        """

        ma = torch.max(lengths)
        mi = torch.min(lengths)
        if i == ma-1:  ## i 如果是最后一句话
            return H[l][:, i, :], H1,H[l][:, : i, :], None ,front_sdj_den[:, i, :i], None, s_mask[:, i, :i], None,\
                   HK1[l][:, :i, :], HK1[l][:, :i, :], HK1[l][:, :i,:], HK1[l][:, :i, :],None,None,None,None

        if batch_integration and i > mi - 1 and i != ma-1:
            Hr = []
            Hr1 = []
            # adjr = []
            # adjr1 = []
            # adjr2 = []
            # adjr3 = []
            front_sdjr_den = []
            s_maskr = []
            tk1 = []
            tk2 = []
            tk3 = []
            tk4 = []

            for A in range(lengths.size()[0]):  # A指一共几段话
                if i < lengths[A]:
                    Hr.append(H[l][A, i, :].unsqueeze(0))
                    Hr1.append(H1[A, :, :].unsqueeze(0))
                    # adjr.append(adj[A, i, :i].unsqueeze(0))
                    # adjr1.append(adj1[A, i, :i].unsqueeze(0))
                    # adjr2.append(adj2[A, i, :i].unsqueeze(0))
                    # adjr3.append(adj3[A, i, :i].unsqueeze(0))
                    front_sdjr_den.append(front_sdj_den[A, i, :i].unsqueeze(0))
                    s_maskr.append(s_mask[A, i, :i].unsqueeze(0))
                    tk1.append(HK1[l][A, :i, :].unsqueeze(0))
                    tk2.append(HK2[l][A, :i, :].unsqueeze(0))
                    tk3.append(HK3[l][A, :i, :].unsqueeze(0))
                    tk4.append(HK4[l][A, :i, :].unsqueeze(0))

            Hr = torch.cat(Hr, dim=0)
            Hr1 = torch.cat(Hr1, dim=0)
            # adjr = torch.cat(adjr, dim=0)
            # adjr1 = torch.cat(adjr1, dim=0)
            # adjr2 = torch.cat(adjr2, dim=0)
            # adjr3 = torch.cat(adjr3, dim=0)
            front_sdjr_den = torch.cat(front_sdjr_den, dim=0)
            s_maskr = torch.cat(s_maskr, dim=0)
            tk1 = torch.cat(tk1, dim=0)
            tk2 = torch.cat(tk2, dim=0)
            tk3 = torch.cat(tk3, dim=0)
            tk4 = torch.cat(tk4, dim=0)
            return Hr, Hr1, front_sdjr_den, s_maskr, tk1, tk2, tk3, tk4

        else:
            return H[l][:, i, :], H1, H[l][:, : i, :],H[l][:, i+1 : , :],front_sdj_den[:, i, :i],back_sdj_den[:,i,i+1:], s_mask[:, i, :i], s_mask[:, i, i+1:],\
                   HK1[l][:, :i, :], HK1[l][:, :i, :], HK1[l][:, :i,:], HK1[l][:, :i, :],\
                   HK1[l][:, i+1:, :], HK1[l][:, i+1:, :], HK1[l][:, i+1:,:], HK1[l][:, i+1:, :]


"""
在顺序注意过程中 根据注意方式 即attn_type 的不同 几种注意计算的类 
目前只使用第一个 即GAT_dialoggcn_v1

"""
class GAT_dialoggcn_v1(nn.Module):
    '''
    use linear to avoid OOM
    H_i = alpha_ij(W_rH_j)
    alpha_ij = attention(H_i, H_j)
    '''
    def __init__(self, args):
        super().__init__()
        hidden_size = args.hidden_dim
        know_dim = args.order_know_dim
        self.front_linear_den = nn.Linear(hidden_size * 2, 1)
        self.back_linear_den = nn.Linear(hidden_size * 2, 1)
        self.linear2 = nn.Linear(hidden_size * 2, 1)
        self.linear3 = nn.Linear(hidden_size * 2, 1)
        self.front_k_liner = nn.Linear(hidden_size + know_dim, hidden_size)
        self.back_k_liner = nn.Linear(hidden_size + know_dim, hidden_size)

        self.Wr0 = nn.Linear(hidden_size, hidden_size, bias = False)
        self.Wr1 = nn.Linear(hidden_size, hidden_size, bias = False)

        self.grus_den1 = nn.GRU(hidden_size,hidden_size)


    def forward(self, i,num_utter,Q, K, V, front_sdj_den,front_s_mask,front_k1, front_k2, front_k3, front_k4,back_K,back_V,back_sdj_den,back_s_mask,back_k1, back_k2, back_k3, back_k4):
        '''
        imformation gatherer with linear attention
        :param Q: (B, D) # query utterance
        :param K: (B, N, D) # context
        :param V: (B, N, D) # context
        :param adj: (B,  N) # the adj matrix of the i th node
        :param s_mask: (B,  N) #
        :return:
        '''
        if i != 0:   ##只要不是第一句话 就要做前向操作
            """前向操作
            """
            B = K.size()[0]
            N = K.size()[1]
            front_K = torch.cat([K,front_k1],dim = -1) #将意图知识和前向文本拼接起来
            front_K = self.front_k_liner(front_K)     ##经过一个线性层
            front_V = front_K
            front_Q = Q.unsqueeze(1).expand(-1, N, -1) # (B, N, D)   本句话    不变
            front_X = torch.cat((front_Q,front_K), dim = 2) # (B, N, 2D)  拼接起来

            """
            分别使用了密度矩阵  
            """
            front_alpha_den = self.front_linear_den(front_X).permute(0, 2, 1)  # (B, 1, N)  本句话与所有之前话语的得分
            # alpha = F.leaky_relu(alpha)
            front_sdj_den = front_sdj_den.unsqueeze(1)  # (B, 1, N)
            front_alpha_den = mask_logic(front_alpha_den, front_sdj_den)  # (B, 1, N) 只有要关注的话语才显示分数
            front_weight_den = F.softmax(front_alpha_den, dim=2)  # (B, 1, N)  近程要关注的softmax得分

            front_V0 = self.Wr0(front_V) # (B, N, D)
            front_V1 = self.Wr1(front_V) # (B, N, D)
            front_s_mask = front_s_mask.unsqueeze(2).float()   # (B, N, 1)
            front_V = front_V0 * front_s_mask + front_V1 * (1 - front_s_mask)

            front_sum_den = torch.bmm(front_weight_den, front_V).squeeze(1)  # (B, D)   远程话语集合
        if i != num_utter-1: ## 只要不是最后一句 就要进行后向操作
            B = back_K.size()[0]
            N = back_K.size()[1]

            back_K = torch.cat([back_K, back_k2], dim=-1)  # 将意图知识和前向文本拼接起来
            back_K = self.back_k_liner(back_K)  ##经过一个线性层
            back_V = back_K
            back_Q = Q.unsqueeze(1).expand(-1, N, -1)  # (B, N, D)   本句话    不变
            back_X = torch.cat((back_Q, back_K), dim=2)  # (B, N, 2D)  拼接起来

            """
            分别使用了密度矩阵  
            """
            back_alpha_den = self.back_linear_den(back_X).permute(0, 2, 1)  # (B, 1, N)  本句话与所有之前话语的得分
            # alpha = F.leaky_relu(alpha)
            back_sdj_den = back_sdj_den.unsqueeze(1)  # (B, 1, N)
            back_alpha_den = mask_logic(back_alpha_den, back_sdj_den)  # (B, 1, N) 只有要关注的话语才显示分数
            back_weight_den = F.softmax(back_alpha_den, dim=2)  # (B, 1, N)  近程要关注的softmax得分

            back_V0 = self.Wr0(back_V)  # (B, N, D)
            back_V1 = self.Wr1(back_V)  # (B, N, D)
            back_s_mask = back_s_mask.unsqueeze(2).float()  # (B, N, 1)
            back_V = back_V0 * back_s_mask + back_V1 * (1 - back_s_mask)

            back_sum_den = torch.bmm(back_weight_den, back_V).squeeze(1)  # (B, D)   远程话语集合




        if i == 0:
            front_sum_den = torch.zeros_like(Q)    ## 如果是第一句话 那前向融合就是和输入的本句话一样形状的全0张量
        if i == num_utter-1:
            back_sum_den = torch.zeros_like(Q)

        return front_sum_den, back_sum_den

class GatDot(nn.Module):
    def __init__(self, hidden_size):

        super().__init__()
        self.hidden_size = hidden_size
        self.front_linear_den = nn.Linear(hidden_size * 2 + 150, 1)
        self.back_linear_den = nn.Linear(hidden_size * 2 + 150, 1)
        self.linear2 = nn.Linear(hidden_size * 2, 1)
        self.linear3 = nn.Linear(hidden_size * 2, 1)
        self.front_k_liner = nn.Linear(hidden_size + 150, hidden_size)
        self.back_k_liner = nn.Linear(hidden_size + 150, hidden_size)

        self.Wr0 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wr1 = nn.Linear(hidden_size, hidden_size, bias=False)

        self.grus_den1 = nn.GRU(hidden_size, hidden_size)

    def forward(self, i,num_utter,Q, K, V, front_sdj_den,front_s_mask,front_k1, front_k2, front_k3, front_k4,back_K,back_V,back_sdj_den,back_s_mask,back_k1, back_k2, back_k3, back_k4):

        if i != 0:  ##只要不是第一句话 就要做前向操作
            """前向操作
            """
            B = K.size()[0]
            N = K.size()[1]
            front_K = K
            # front_K = self.front_k_liner(front_K)  ##经过一个线性层
            front_V = front_K
            front_Q = Q.unsqueeze(1).expand(-1, N, -1)  # (B, N, D)   本句话    不变

            front_X = torch.cat((front_K, front_k1, front_Q), dim=2)  # (B, N, 2D+150)  拼接起来

            """
            分别使用了密度矩阵  
            """
            front_alpha_den = self.front_linear_den(front_X).permute(0, 2, 1)  # (B, 1, N)  本句话与所有之前话语的得分
            # alpha = F.leaky_relu(alpha)
            front_sdj_den = front_sdj_den.unsqueeze(1)  # (B, 1, N)
            front_alpha_den = mask_logic(front_alpha_den, front_sdj_den)  # (B, 1, N) 只有要关注的话语才显示分数
            front_weight_den = F.softmax(front_alpha_den, dim=2)  # (B, 1, N)  近程要关注的softmax得分

            front_V0 = self.Wr0(front_V)  # (B, N, D)
            front_V1 = self.Wr1(front_V)  # (B, N, D)
            front_s_mask = front_s_mask.unsqueeze(2).float()  # (B, N, 1)
            front_V = front_V0 * front_s_mask + front_V1 * (1 - front_s_mask)

            front_sum_den = torch.bmm(front_weight_den, front_V).squeeze(1)  # (B, D)   远程话语集合
        if i != num_utter - 1:  ## 只要不是最后一句 就要进行后向操作
            B = back_K.size()[0]
            N = back_K.size()[1]

            back_K = back_K
            # back_K = self.back_k_liner(back_K)  ##经过一个线性层
            back_V = back_K
            back_Q = Q.unsqueeze(1).expand(-1, N, -1)  # (B, N, D)   本句话    不变

            back_X = torch.cat((back_Q, back_k2, back_K), dim=2)  # (B, N, 2D)  拼接起来

            """
            分别使用了密度矩阵  
            """
            back_alpha_den = self.back_linear_den(back_X).permute(0, 2, 1)  # (B, 1, N)  本句话与所有之前话语的得分
            # alpha = F.leaky_relu(alpha)
            back_sdj_den = back_sdj_den.unsqueeze(1)  # (B, 1, N)
            back_alpha_den = mask_logic(back_alpha_den, back_sdj_den)  # (B, 1, N) 只有要关注的话语才显示分数
            back_weight_den = F.softmax(back_alpha_den, dim=2)  # (B, 1, N)  近程要关注的softmax得分

            back_V0 = self.Wr0(back_V)  # (B, N, D)
            back_V1 = self.Wr1(back_V)  # (B, N, D)
            back_s_mask = back_s_mask.unsqueeze(2).float()  # (B, N, 1)
            back_V = back_V0 * back_s_mask + back_V1 * (1 - back_s_mask)

            back_sum_den = torch.bmm(back_weight_den, back_V).squeeze(1)  # (B, D)   远程话语集合

        if i == 0:
            front_sum_den = torch.zeros_like(Q)  ## 如果是第一句话 那前向融合就是和输入的本句话一样形状的全0张量
        if i == num_utter - 1:
            back_sum_den = torch.zeros_like(Q)

        return front_sum_den, back_sum_den



class GatDot_rel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.rel_emb = nn.Embedding(2, hidden_size)

    def forward(self, Q, K, V, adj, s_mask):
        '''
        imformation gatherer with dot product attention
        :param Q: (B, D) # query utterance
        :param K: (B, N, D) # context
        :param V: (B, N, D) # context
        :param adj: (B,  N) # the adj matrix of the i th node
        :param s_mask: (B,  N) #  relation mask
        :return:
        '''
        N = K.size()[1]

        rel_emb = self.rel_emb(s_mask)
        Q = self.linear1(Q).unsqueeze(2) # (B,D,1)
        K = self.linear2(K) # (B, N, D)
        y = self.linear3(rel_emb) # (B, N, 1)

        alpha = (torch.bmm(K, Q) + y).permute(0, 2, 1)  # (B, 1, N)

        adj = adj.unsqueeze(1)
        alpha = mask_logic(alpha, adj)  # (B, 1, N)

        attn_weight = F.softmax(alpha, dim=2)  # (B, 1, N)

        attn_sum = torch.bmm(attn_weight, V).squeeze(1)  # (B,  D)

        return attn_weight, attn_sum
class GatLinear(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size * 2, 1)


    def forward(self, Q, K, V, adj):
        '''
        imformation gatherer with linear attention
        :param Q: (B, D) # query utterance
        :param K: (B, N, D) # context
        :param V: (B, N, D) # context
        :param adj: (B,  N) # the adj matrix of the i th node
        :return:
        '''
        N = K.size()[1]
        # print('Q',Q.size())
        Q = Q.unsqueeze(1).expand(-1, N, -1) # (B, N, D)
        # print('K',K.size())
        X = torch.cat((Q,K), dim = 2) # (B, N, 2D)
        # print('X',X.size())
        alpha = self.linear(X).permute(0,2,1) #(B, 1, N)
        # print('alpha',alpha.size())
        # print(alpha)
        adj = adj.unsqueeze(1)
        alpha = mask_logic(alpha, adj) # (B, 1, N)
        # print('alpha after mask',alpha.size())
        # print(alpha)

        attn_weight = F.softmax(alpha, dim = 2) # (B, 1, N)
        # print('attn_weight',attn_weight.size())
        # print(attn_weight)

        attn_sum = torch.bmm(attn_weight, V).squeeze(1) # (B, D)
        # print('attn_sum',attn_sum.size())

        return attn_weight, attn_sum
class GatLinear_rel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size * 3, 1)
        self.rel_emb = nn.Embedding(2, hidden_size)


    def forward(self, Q, K, V, adj, s_mask):
        '''
        imformation gatherer with linear attention
        :param Q: (B, D) # query utterance
        :param K: (B, N, D) # context
        :param V: (B, N, D) # context
        :param adj: (B,  N) # the adj matrix of the i th node
        :param s_mask: (B,  N) #
        :return:
        '''
        rel_emb = self.rel_emb(s_mask) # (B, N, D)
        N = K.size()[1]
        # print('Q',Q.size())
        Q = Q.unsqueeze(1).expand(-1, N, -1) # (B, N, D)
        # print('K',K.size())
        # print('rel_emb', rel_emb.size())
        X = torch.cat((Q,K, rel_emb), dim = 2) # (B, N, 2D)?   (B, N, 3D)
        # print('X',X.size())
        alpha = self.linear(X).permute(0,2,1) #(B, 1, N)
        # print('alpha',alpha.size())
        # print(alpha)
        adj = adj.unsqueeze(1)
        alpha = mask_logic(alpha, adj) # (B, 1, N)
        # print('alpha after mask',alpha.size())
        # print(alpha)

        attn_weight = F.softmax(alpha, dim = 2) # (B, 1, N)
        # print('attn_weight',attn_weight.size())
        # print(attn_weight)

        attn_sum = torch.bmm(attn_weight, V).squeeze(1) # (B, D)
        # print('attn_sum',attn_sum.size())

        return attn_weight, attn_sum

def mask_logic(alpha, adj):
    '''
    performing mask logic with adj
    :param alpha:
    :param adj:
    :return:
    '''
    return alpha - (1 - adj) * 1e30




































class TextCNN(nn.Module):
    def __init__(self, input_dim, emb_size=128, in_channels=1, out_channels=128, kernel_heights=[3, 4, 5], dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], input_dim), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], input_dim), stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], input_dim), stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)
        self.embd = nn.Sequential(
            nn.Linear(len(kernel_heights) * out_channels, emb_size),
            nn.ReLU(inplace=True),
        )

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)  # conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))  # activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(
            2)  # maxpool_out.size() = (batch_size, out_channels)
        return max_out

    def forward(self, frame_x):
        batch_size, seq_len, feat_dim = frame_x.size()  # dia_len, utt_len, batch_size, feat_dim
        frame_x = frame_x.view(batch_size, 1, seq_len, feat_dim)
        max_out1 = self.conv_block(frame_x, self.conv1)
        max_out2 = self.conv_block(frame_x, self.conv2)
        max_out3 = self.conv_block(frame_x, self.conv3)
        all_out = torch.cat((max_out1, max_out2, max_out3), 1)
        fc_in = self.dropout(all_out)
        embd = self.embd(fc_in)
        return embd

class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, return_feature, use_residue,
                 new_graph=False):
        super(GCNII, self).__init__()
        self.return_feature = return_feature
        self.use_residue = use_residue
        self.new_graph = new_graph
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        if not return_feature:
            self.fcs.append(nn.Linear(nfeat + nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def cossim(self, x, y):
        a = torch.matmul(x, y)
        b = torch.sqrt(torch.matmul(x, x)) * torch.sqrt(torch.matmul(y, y))
        if b == 0:
            return 0
        else:
            return (a / b)

    def forward(self, x, dia_len, topicLabel):
        if self.new_graph:
            adj = self.message_passing_directed_speaker(x, dia_len, topicLabel)
        else:
            adj = self.create_big_adj(x, dia_len)
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        if self.use_residue:
            layer_inner = torch.cat([x, layer_inner], dim=-1)
        if not self.return_feature:
            layer_inner = self.fcs[-1](layer_inner)
            layer_inner = F.log_softmax(layer_inner, dim=1)
        return layer_inner

    def create_big_adj(self, x, dia_len):
        adj = torch.zeros((x.shape[0], x.shape[0]))
        start = 0
        for i in range(len(dia_len)):
            sub_adj = torch.zeros((dia_len[i], dia_len[i]))
            temp = x[start:start + dia_len[i]]
            temp_len = torch.sqrt(torch.bmm(temp.unsqueeze(1), temp.unsqueeze(2)).squeeze(-1).squeeze(-1))
            temp_len_matrix = temp_len.unsqueeze(1) * temp_len.unsqueeze(0)
            cos_sim_matrix = torch.matmul(temp, temp.permute(1, 0)) / temp_len_matrix
            sim_matrix = torch.acos(cos_sim_matrix * 0.99999)
            sim_matrix = 1 - sim_matrix / math.pi

            sub_adj[:dia_len[i], :dia_len[i]] = sim_matrix

            m_start = start
            n_start = start
            adj[m_start:m_start + dia_len[i], n_start:n_start + dia_len[i]] = sub_adj

            start += dia_len[i]
        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D).to(device)

        return adj

    def message_passing_wo_speaker(self, x, dia_len, topicLabel):
        adj = torch.zeros((x.shape[0], x.shape[0])) + torch.eye(x.shape[0])
        start = 0
        for i in range(len(dia_len)):  #
            for j in range(dia_len[i] - 1):
                for pin in range(dia_len[i] - 1 - j):
                    xz = start + j
                    yz = xz + pin + 1
                    f = self.cossim(x[xz], x[yz])
                    if f > 1 and f < 1.05:
                        f = 1
                    elif f < -1 and f > -1.05:
                        f = -1
                    elif f >= 1.05 or f <= -1.05:
                        print('cos = {}'.format(f))
                    Aij = 1 - math.acos(f) / math.pi
                    adj[xz][yz] = Aij
                    adj[yz][xz] = Aij
            start += dia_len[i]

        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D).to(device)

        return adj

    def atom_calculate_edge_weight(self, x, y):
        f = self.cossim(x, y)
        if f > 1 and f < 1.05:
            f = 1
        elif f < -1 and f > -1.05:
            f = -1
        elif f >= 1.05 or f <= -1.05:
            print('cos = {}'.format(f))
        return f

    def message_passing_directed_speaker(self, x, dia_len, qmask):
        total_len = sum(dia_len)
        adj = torch.zeros((total_len, total_len)) + torch.eye(total_len)
        start = 0
        use_utterance_edge = False
        for (i, len_) in enumerate(dia_len):
            speaker0 = []
            speaker1 = []
            for (j, speaker) in enumerate(qmask[i][0:len_]):
                if speaker[0] == 1:
                    speaker0.append(j)
                else:
                    speaker1.append(j)
            if use_utterance_edge:
                for j in range(len_ - 1):
                    f = self.atom_calculate_edge_weight(x[start + j], x[start + j + 1])
                    Aij = 1 - math.acos(f) / math.pi
                    adj[start + j][start + j + 1] = Aij
                    adj[start + j + 1][start + j] = Aij
            for k in range(len(speaker0) - 1):
                f = self.atom_calculate_edge_weight(x[start + speaker0[k]], x[start + speaker0[k + 1]])
                Aij = 1 - math.acos(f) / math.pi
                adj[start + speaker0[k]][start + speaker0[k + 1]] = Aij
                adj[start + speaker0[k + 1]][start + speaker0[k]] = Aij
            for k in range(len(speaker1) - 1):
                f = self.atom_calculate_edge_weight(x[start + speaker1[k]], x[start + speaker1[k + 1]])
                Aij = 1 - math.acos(f) / math.pi
                adj[start + speaker1[k]][start + speaker1[k + 1]] = Aij
                adj[start + speaker1[k + 1]][start + speaker1[k]] = Aij

            start += dia_len[i]

        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D).to(device)

        return adj.to(device)

    def message_passing_relation_graph(self, x, dia_len):
        total_len = sum(dia_len)
        adj = torch.zeros((total_len, total_len)) + torch.eye(total_len)
        window_size = 10
        start = 0
        for (i, len_) in enumerate(dia_len):
            edge_set = []
            for k in range(len_):
                left = max(0, k - window_size)
                right = min(len_ - 1, k + window_size)
                edge_set = edge_set + [str(i) + '_' + str(j) for i in range(left, right) for j in
                                       range(i + 1, right + 1)]
            edge_set = [[start + int(str_.split('_')[0]), start + int(str_.split('_')[1])] for str_ in
                        list(set(edge_set))]
            for left, right in edge_set:
                f = self.atom_calculate_edge_weight(x[left], x[right])
                Aij = 1 - math.acos(f) / math.pi
                adj[left][right] = Aij
                adj[right][left] = Aij
            start += dia_len[i]

        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D).to(device)

        return adj.to(device)
