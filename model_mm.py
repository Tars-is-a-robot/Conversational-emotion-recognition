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
from model_GCN import GCNII_lyc
import ipdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output


# t
class MM_GCN(nn.Module):
    def __init__(self, args, nhidden, nclass, variant, return_feature, new_graph='full',n_speakers=2, modals=['a','v','l']):
        super(MM_GCN, self).__init__()
        self.args = args
        self.return_feature = return_feature
        self.use_residue = args.use_residue
        self.new_graph = new_graph

        # 图的节点更新运算网络
        self.graph_net = GCNII_lyc(args, nhidden=nhidden, nclass=nclass, variant=variant, return_feature=return_feature)
        self.a_fc = nn.Linear(args.graph_fusion_dim, args.graph_fusion_dim)
        self.v_fc = nn.Linear(args.graph_fusion_dim, args.graph_fusion_dim)
        self.l_fc = nn.Linear(args.graph_fusion_dim, args.graph_fusion_dim)
        if self.use_residue:
            self.feature_fc = nn.Linear(args.graph_fusion_dim*3+nhidden*3, nhidden)
        else:
            self.feature_fc = nn.Linear(nhidden * 3, nhidden)
        self.final_fc = nn.Linear(nhidden, nclass)
        self.act_fn = nn.ReLU()
        self.dropout = args.dropout
        self.modals = modals
        self.modal_embeddings = nn.Embedding(3, args.graph_fusion_dim)
        self.speaker_embeddings = nn.Embedding(n_speakers, args.graph_fusion_dim)
        self.a_spk_embs = nn.Embedding(n_speakers, args.graph_fusion_dim)
        self.v_spk_embs = nn.Embedding(n_speakers, args.graph_fusion_dim)
        self.l_spk_embs = nn.Embedding(n_speakers, args.graph_fusion_dim)
        self.use_speaker = args.use_speaker
        self.use_modal = args.use_modal

    def forward(self, a, v, l, dia_len, qmask, global_sdj_den, front_sdj_den, back_sdj_den, s_mask, k1_0, k2_0, k3_0, k4_0, features_k1,features_k2,features_k3):
        """
            输入为节点向量
            文本 l ->tensor(sum_seq,200)  sum_seq 为这个batch中话语的真实数量
            知识 a, v, k1_0, k2_0, k3_0, k4_0, features_k1,features_k2,features_k3 ->tensor(sum_seq,200)  sum_seq 为这个batch中话语的真实数量
            dia_len ->list:16 表示每段对话的话语数量
            qmask ->tensor(seq,batch,2) 说话人矩阵 独热向量表示
            global_sdj_den, front_sdj_den, back_sdj_den   batch,seq,seq   应注意的话语
            s_mask batch,seq,seq  判断两句话是否是同一个人说的 是用1 不是用0
            输出为  features ->tensor(3*sum_seq,900) 将更新后的3种模态的特征拼接起来
        """
        qmask = torch.cat([qmask[:x, i, :] for i,x in enumerate(dia_len)],dim=0) # qmask ->tensor(sum_seq,2) 说话人矩阵 独热向量表示

        spk_idx = torch.argmax(qmask, dim=-1) # spk_idx ->tensor(sum_seq,) 标出说话人是0还是1
        spk_emb_vector = self.speaker_embeddings(spk_idx) #把说话人向量化 spk_emb_vector ->tensor(sum_seq,200)
        if self.use_speaker:  #use_speaker = False 是否在文本向量中加入说话人向量
            if 'l' in self.modals:
                l += spk_emb_vector ## l 与 spk_emb_vector对应相加 l ->tensor(sum_seq,200) 在文本表示中加入了说话人向量表示
        if self.use_modal:  #use_modal = False 是否加入知识编码
            emb_idx = torch.LongTensor([0, 1, 2]).to(device)
            emb_vector = self.modal_embeddings(emb_idx)
            if 'a' in self.modals:
                a += emb_vector[0].reshape(1, -1).expand(a.shape[0], a.shape[1])
            if 'v' in self.modals:
                v += emb_vector[1].reshape(1, -1).expand(v.shape[0], v.shape[1])
            if 'l' in self.modals:
                l += emb_vector[2].reshape(1, -1).expand(l.shape[0], l.shape[1])

        """ self.args.use_huge_adj 是否使用更多的知识参与卷积
            create_big_adj() 输入 
            a, v, l, ->tensor(sum_seq,200) 三种模态向量表示 sum_seq 为这个batch中话语的真实数量
            dia_len, ->list:16 表示每段对话的话语数量  self.modals = ['a', 'v', 'l'] list
            global_sdj_den   batch,seq,seq   应注意的话语
            输出 adj ->tensor(sum_seq,sum_seq) 为整个batch的权重矩阵  
            加 sdj_l 单独拿出文本间的注意矩阵
            adj 文本加知识的全局图权重    adj_l 单独文本图权重矩阵
        """
        if self.args.use_huge_adj:
            all_length = l.shape[0] if len(l) != 0 else a.shape[0] if len(a) != 0 else v.shape[0]
            adj_huge, adj_know_and_text = self.create_huge_adj(a, v, l, dia_len, self.modals, global_sdj_den, s_mask, features_k1, features_k2, features_k3)
            features = torch.cat([a, v, l, features_k1, features_k2, features_k3], dim=0).to(device)  # 文本和知识都进去 features ->tensor(6*sum_seq,200)
            features = self.graph_net(features, dia_len, qmask, front_sdj_den, back_sdj_den, global_sdj_den, s_mask, k1_0, k2_0, k3_0, k4_0, adj_know_and_text)  # (sum_seq,300) 这个batch中文本的向量表示 维度也变了 并成为这个类的输出
            if self.args.only_text:  # 最后的特征只是文本向量
                features = features[all_length * 2:all_length * 3]
            else:
                features = torch.cat([features[:all_length], features[all_length:all_length * 2],
                                      features[all_length * 2:all_length * 3]], dim=-1)


        else:
            adj, adj_l, adj_know_and_text, adj_know_no_text, adj_know_in_text = self.create_big_adj(a, v, l, dia_len, self.modals, global_sdj_den, s_mask)
            """ 把features放进一个类self.graph_net(features, None, qmask, adj) 为 model_GCN.py中的class GCNII_lyc(nn.Module)
                输入为 features, ->tensor(sum_seq,200) 这个batch中三种模态的向量表示
                qmask, ->tensor(sum_seq,2) 说话人矩阵 独热向量表示
                adj, ->tensor(sum_seq,sum_seq) 为整个batch的权重矩阵
                输出 features ->tensor(sum_seq,300) 这个batch中三种模态的向量表示 维度也变了 并成为这个类的输出
            """
            if self.args.use_know_graph:   ## 是否将知识参与图卷积
                features = torch.cat([a, v, l], dim=0).to(device)  # 文本和知识都进去 features ->tensor(3*sum_seq,200)
                if self.args.text_and_know == 'all':
                    features = self.graph_net(features, dia_len, qmask, front_sdj_den, back_sdj_den, global_sdj_den, s_mask,
                                          k1_0, k2_0, k3_0, k4_0, adj)  # (sum_seq,300) 这个batch中文本的向量表示 维度也变了 并成为这个类的输出
                elif self.args.text_and_know == 'tandk':
                    features = self.graph_net(features, dia_len, qmask, front_sdj_den, back_sdj_den, global_sdj_den, s_mask,
                                              k1_0, k2_0, k3_0, k4_0, adj_know_and_text)
                elif self.args.text_and_know == 'tnok':
                    features = self.graph_net(features, dia_len, qmask, front_sdj_den, back_sdj_den, global_sdj_den, s_mask,
                                              k1_0, k2_0, k3_0, k4_0, adj_know_no_text)
                elif self.args.text_and_know == 'tink':
                    features = self.graph_net(features, dia_len, qmask, front_sdj_den, back_sdj_den, global_sdj_den, s_mask,
                                              k1_0, k2_0, k3_0, k4_0, adj_know_in_text)

                """
                因为知识也参与了图卷积 这是将文本和知识拼接起来 
                用于最终分类 如果只用文本 那就不相加了 要使用的话 就是将维度从300 拼接到了900
                """
                all_length = l.shape[0] if len(l) != 0 else a.shape[0] if len(a) != 0 else v.shape[0]  # all_length （int） sum_seq batch中话语总长度
                if self.args.only_text:  # 最后的特征只是文本向量
                    features = features[all_length * 2:all_length * 3]
                else:  # 进入 features ->tensor(sum_seq,900) 将3种模态的特征拼接起来
                    features = torch.cat([features[:all_length], features[all_length:all_length * 2], features[all_length * 2:all_length * 3]], dim=-1)

            else:
                features = l      # features ->tensor(3*sum_seq,200)  只将文本当做feature ->tensor(sum_seq,200)
                features = self.graph_net(features, dia_len, qmask, front_sdj_den, back_sdj_den, global_sdj_den, s_mask, k1_0, k2_0, k3_0, k4_0, adj_l)  # (sum_seq,300) 这个batch中文本的向量表示 维度也变了 并成为这个类的输出

        if self.return_feature:  # Ture
            return features     # 维度是300或900

        else:
            return F.softmax(self.final_fc(features), dim=-1)

    def create_big_adj(self, a, v, l, dia_len, modals, global_sdj_den, s_mask):
        """ 输入
            a, v, l, ->tensor(sum_seq,200)  三种模态向量表示  sum_seq 为这个batch中话语的真实数量
            dia_len, ->list:16 表示每段对话的话语数量  self.modals = ['a', 'v', 'l'] list:3
            global_sdj_den   batch,seq,seq   应注意的话语
            输出 adj ->tensor(sum_seq,sum_seq) 为整个batch的权重矩阵
        """

        modal_num = len(modals) # 3
        all_length = l.shape[0] if len(l)!=0 else a.shape[0] if len(a) != 0 else v.shape[0] # 这个batch中话语总个数
        adj = torch.zeros((modal_num*all_length, modal_num*all_length)).to(device) # 建立图矩阵 adj ->tensor(3*话语总数,3*话语总数) 全0
        if len(modals) == 3: #执行 features list:3  每个元素是tensor(sum_seq,200)
            features = [a, v, l]
        elif 'a' in modals and 'v' in modals:
            features = [a, v]
        elif 'a' in modals and 'l' in modals:
            features = [a, l]
        elif 'v' in modals and 'l' in modals:
            features = [v, l]
        else:
            return NotImplementedError
        start = 0
        """
        完成这个循环后 得到这个batch的权重矩阵 
        """
        for i in range(len(dia_len)): # batch中有几段对话
            sub_adjs = []
            """ 这个循环中，x是a，v，l中的一个张量 tensor(sum_seq,200) 但进入循环后会切割出来本段对话需要的向量 得到本段对话三种模态各自的权重矩阵
                然后进入m、n的循环 得到本段对话中不同模态间的权重矩阵 不同模态间权重矩阵只是一个对角阵 即只在对角线上有值 循环完成 就得到了本段对话一
                个完整的权重矩阵 然后继续下一个i循环 即第i段对话 整个循环进行完成 会得到完整batch的三种模态的权重矩阵
                这个循环 是要实现一段对话的三个模态各自模态内图的构建  第i段对话 
            """
            for j, x in enumerate(features):
                if j < 0:
                    sub_adj = torch.zeros((dia_len[i], dia_len[i])) + torch.eye(dia_len[i])
                else:
                    sub_adj = torch.zeros((dia_len[i], dia_len[i]))  # dia_len[i]第i段对话的会话数量
                    temp = x[start:start + dia_len[i]] # 来确定第i段对话在x中所处范围 并取出 temp ->tensor(seq,200)
                    """ temp.mul(temp) : torch.mul() 输入是两个张量 对应位置相乘 返回一个张量 这个表示temp自己对应元素相乘
                        相当于每个元素平方 然后torch.sum() 在维度1上元素相加 返回tensor(seq) torch.sqrt()逐元素计算张量的平方根
                    """
                    vec_length = torch.sqrt(torch.sum(temp.mul(temp), dim=1)) # vec_length ->tensor(seq,)
                    norm_temp = (temp.permute(1, 0) / vec_length) # norm_temp->tensor(200,seq) 变换后的节点向量
                    cos_sim_matrix = torch.sum(torch.matmul(norm_temp.unsqueeze(2), norm_temp.unsqueeze(1)), dim=0)  # seq, seq
                    cos_sim_matrix = cos_sim_matrix * 0.99999 #cos_sim_matrix ->tensor(seq,seq)
                    sim_matrix = torch.acos(cos_sim_matrix)/np.pi # 该段对话最终的权重矩阵 sim_matrix ->tensor(seq,seq)

                    if self.args.use_density:   # 是否只注意密度矩阵注意的话语
                        if j==2:
                            sub_adj[:dia_len[i], :dia_len[i]] = sim_matrix * global_sdj_den[i,:dia_len[i],:dia_len[i]]# sub_adj ->tensor(seq,seq)
                        else:
                            if self.args.know_fusion_att == 'all':
                                sub_adj[:dia_len[i], :dia_len[i]] = sim_matrix
                            elif self.args.know_fusion_att == 'selfk':
                                sub_adj[:dia_len[i], :dia_len[i]] = sim_matrix * s_mask[i, :dia_len[i], :dia_len[i]]
                            elif self.args.know_fusion_att == 'denk':
                                sub_adj[:dia_len[i], :dia_len[i]] = sim_matrix * global_sdj_den[i, :dia_len[i], :dia_len[i]]
                            else:
                                sub_adj[:dia_len[i], :dia_len[i]] = sim_matrix * global_sdj_den[i, :dia_len[i], :dia_len[i]] * s_mask[i, :dia_len[i], :dia_len[i]]


                    else:
                        sub_adj[:dia_len[i], :dia_len[i]] = sim_matrix
                sub_adjs.append(sub_adj) ##每完成一段对话中的一个模态权重矩阵 都放到sub_adjs这个list中 最后这段对话三种模态循环完成 list:3
            """ np.diag_indices(dia_len[i]) 输入是一个int 代表长度 输出为在对角线上的索引 即
                (array([0, 1, 2, 3, ...dia_len[i]-1]), array([0, 1, 2, 3, ...dia_len[i]-1]))
                然后再 np.array 得到 [[0 1 2 3 ...dia_len[i]-1]
                                     [0 1 2 3 ...dia_len[i]-1]]
            """
            dia_idx = np.array(np.diag_indices(dia_len[i])) #ndarry（2，seq）
            for m in range(modal_num):
                for n in range(modal_num):
                    m_start = start + all_length*m
                    n_start = start + all_length*n
                    if m == n:
                        adj[m_start:m_start+dia_len[i], n_start:n_start+dia_len[i]] = sub_adjs[m]
                    else:
                        modal1 = features[m][start:start+dia_len[i]] #modal1，modal2 ->tensor(seq,200)
                        modal2 = features[n][start:start+dia_len[i]] #得到这段对话中两种不同模态的权重矩阵
                        normed_modal1 = modal1.permute(1, 0) / torch.sqrt(torch.sum(modal1.mul(modal1), dim=1)) #dim, length
                        normed_modal2 = modal2.permute(1, 0) / torch.sqrt(torch.sum(modal2.mul(modal2), dim=1)) #dim, length
                        dia_cos_sim = torch.sum(normed_modal1.mul(normed_modal2).permute(1, 0), dim=1) #tensor(seq)
                        dia_cos_sim = dia_cos_sim * 0.99999
                        dia_sim = 1 - torch.acos(dia_cos_sim)/np.pi
                        idx =dia_idx.copy()
                        idx[0] += m_start
                        idx[1] += n_start
                        adj[idx] = dia_sim  ##将同一话语不同模态间的得分放到整体权重矩阵中

            start += dia_len[i]  ##第i段话从哪里开始

        d = adj.sum(1) #tensor(sum_seq) 在维度1 上加和
        D = torch.diag(torch.pow(d, -0.5)) # 建立一个tensor ->tensor(sum_seq,sum_seq) 对角线上值是torch.pow(d, -0.5) 其他全是0
        adj = D.mm(adj).mm(D) # mm 矩阵乘法   得到的是一个归一化的矩阵表示

        """
        新的权重矩阵 知识内部不融合 
        分为只让文本与相对应的知识图卷积中进行融合
        除了文本与相对应的知识在图卷积中进行融合 相对应的不同知识和文本也融合
        """
        adj_l_no_k = adj
        adj_l_nok = torch.zeros(all_length, all_length)  # 全0
        adj_l_nok1 = torch.zeros(all_length, all_length) + torch.eye(all_length) #只在对角线有元素
        adj_l_no_k[:all_length, :all_length] = adj_l_nok1*adj_l_no_k[:all_length, :all_length]
        adj_l_no_k[all_length:all_length*2, all_length:all_length*2] = adj_l_no_k[all_length:all_length*2, all_length:all_length*2] * adj_l_nok1
        adj_know_and_text = adj_l_no_k  ##除了文本与相对应的知识在图卷积中进行融合 相对应的不同知识和文本也融合  就是让同一知识内部不再融合

        adj_l_no_k[ : all_length, all_length : 2 * all_length] = adj_l_nok
        adj_l_no_k[ : all_length, 2 * all_length :] =  adj_l_nok
        adj_l_no_k[all_length : 2 * all_length,  : all_length] = adj_l_nok
        adj_l_no_k[all_length : 2 * all_length, 2 * all_length :] = adj_l_nok
        adj_know_no_text = adj_l_no_k #只让文本与相对应的知识图卷积中进行融合

        adj_know_text = adj
        adj_know_text[: all_length, all_length: 2 * all_length] = adj_l_nok
        adj_know_text[: all_length, 2 * all_length:] = adj_l_nok
        adj_know_text[all_length: 2 * all_length, : all_length] = adj_l_nok
        adj_know_text[all_length: 2 * all_length, 2 * all_length:] = adj_l_nok  ## 不同否认知识间不融合
        adj_know_in_text = adj_know_text

        d = adj_know_and_text.sum(1)  # tensor(sum_seq) 在维度1 上加和
        D = torch.diag(torch.pow(d, -0.5))  # 建立一个tensor ->tensor(sum_seq,sum_seq) 对角线上值是torch.pow(d, -0.5) 其他全是0
        adj_know_and_text = D.mm(adj_know_and_text).mm(D)  # mm 矩阵乘法   得到的是一个归一化的矩阵表示

        d = adj_know_no_text.sum(1)  # tensor(sum_seq) 在维度1 上加和
        D = torch.diag(torch.pow(d, -0.5))  # 建立一个tensor ->tensor(sum_seq,sum_seq) 对角线上值是torch.pow(d, -0.5) 其他全是0
        adj_know_no_text = D.mm(adj_know_no_text).mm(D)  # mm 矩阵乘法   得到的是一个归一化的矩阵表示

        adj_l = adj[all_length * 2:, all_length * 2:]
        d_l = adj_l.sum(1)  # tensor(sum_seq) 在维度1 上加和
        D_l = torch.diag(torch.pow(d_l, -0.5))  # 建立一个tensor ->tensor(sum_seq,sum_seq) 对角线上值是torch.pow(d, -0.5) 其他全是0
        adj_l = D_l.mm(adj_l).mm(D_l)  # mm 矩阵乘法   得到的是一个归一化的矩阵表示
        ## adj是根据图构建规则构建的全局图矩阵   adj_l是根据图构建规则构建的单文本权重矩阵
        return adj, adj_l, adj_know_and_text, adj_know_no_text, adj_know_in_text

    def create_huge_adj(self, a, v, l, dia_len, modals, global_sdj_den, s_mask, k1, k2, k3):
        """
        多使用几个外部知识参与图卷积 再将外部知识与文本的图卷积分来 最后将两个不同的图卷积进行加权和
        输入
            a, v, l, ->tensor(sum_seq,200)  三种模态向量表示  sum_seq 为这个batch中话语的真实数量
            dia_len, ->list:16 表示每段对话的话语数量  self.modals = ['a', 'v', 'l'] list:3
            global_sdj_den   batch,seq,seq   应注意的话语
            输出 adj ->tensor(sum_seq,sum_seq) 为整个batch的权重矩阵
        """

        modal_num = self.args.know_fusion_num + 1   # 几种知识参与图卷积
        all_length = l.shape[0] if len(l)!=0 else a.shape[0] if len(a) != 0 else v.shape[0] # 这个batch中话语总个数
        adj = torch.zeros((modal_num*all_length, modal_num*all_length)).to(device) # 建立图矩阵 adj ->tensor(6*话语总数,6*话语总数) 全0
        #执行 features list:3  每个元素是tensor(sum_seq,200)
        features = [a, v, l, k1, k2, k3]

        start = 0
        """
        完成这个循环后 得到这个batch的权重矩阵 
        """
        for i in range(len(dia_len)): # batch中有几段对话
            sub_adjs = []
            """ 这个循环中，x是a，v，l中的一个张量 tensor(sum_seq,200) 但进入循环后会切割出来本段对话需要的向量 得到本段对话三种模态各自的权重矩阵
                然后进入m、n的循环 得到本段对话中不同模态间的权重矩阵 不同模态间权重矩阵只是一个对角阵 即只在对角线上有值 循环完成 就得到了本段对话一
                个完整的权重矩阵 然后继续下一个i循环 即第i段对话 整个循环进行完成 会得到完整batch的三种模态的权重矩阵
                这个循环 是要实现一段对话的三个模态各自模态内图的构建  第i段对话 
            """
            for j, x in enumerate(features):
                if j < 0:
                    sub_adj = torch.zeros((dia_len[i], dia_len[i])) + torch.eye(dia_len[i])
                else:
                    sub_adj = torch.zeros((dia_len[i], dia_len[i]))  # dia_len[i]第i段对话的会话数量
                    temp = x[start:start + dia_len[i]] # 来确定第i段对话在x中所处范围 并取出 temp ->tensor(seq,200)
                    """ temp.mul(temp) : torch.mul() 输入是两个张量 对应位置相乘 返回一个张量 这个表示temp自己对应元素相乘
                        相当于每个元素平方 然后torch.sum() 在维度1上元素相加 返回tensor(seq) torch.sqrt()逐元素计算张量的平方根
                    """
                    vec_length = torch.sqrt(torch.sum(temp.mul(temp), dim=1)) # vec_length ->tensor(seq,)
                    norm_temp = (temp.permute(1, 0) / vec_length) # norm_temp->tensor(200,seq) 变换后的节点向量
                    cos_sim_matrix = torch.sum(torch.matmul(norm_temp.unsqueeze(2), norm_temp.unsqueeze(1)), dim=0)  # seq, seq
                    cos_sim_matrix = cos_sim_matrix * 0.99999 #cos_sim_matrix ->tensor(seq,seq)
                    sim_matrix = 1 - torch.acos(cos_sim_matrix)/np.pi # 该段对话最终的权重矩阵 sim_matrix ->tensor(seq,seq)
                    if self.args.use_density:   # 是否只注意密度矩阵注意的话语
                        if j==2:
                            sub_adj[:dia_len[i], :dia_len[i]] = sim_matrix * global_sdj_den[i,:dia_len[i],:dia_len[i]]# sub_adj ->tensor(seq,seq)
                        else:
                            if self.args.know_fusion_att == 'all':
                                sub_adj[:dia_len[i], :dia_len[i]] = sim_matrix
                            elif self.args.know_fusion_att == 'selfk':
                                sub_adj[:dia_len[i], :dia_len[i]] = sim_matrix * s_mask[i, :dia_len[i], :dia_len[i]]
                            elif self.args.know_fusion_att == 'denk':
                                sub_adj[:dia_len[i], :dia_len[i]] = sim_matrix * global_sdj_den[i, :dia_len[i], :dia_len[i]]
                            else:
                                sub_adj[:dia_len[i], :dia_len[i]] = sim_matrix * global_sdj_den[i, :dia_len[i], :dia_len[i]] * s_mask[i, :dia_len[i], :dia_len[i]]


                    else:
                        sub_adj[:dia_len[i], :dia_len[i]] = sim_matrix
                sub_adjs.append(sub_adj) ##每完成一段对话中的一个模态权重矩阵 都放到sub_adjs这个list中 最后这段对话三种模态循环完成 list:6
            """ np.diag_indices(dia_len[i]) 输入是一个int 代表长度 输出为在对角线上的索引 即
                (array([0, 1, 2, 3, ...dia_len[i]-1]), array([0, 1, 2, 3, ...dia_len[i]-1]))
                然后再 np.array 得到 [[0 1 2 3 ...dia_len[i]-1]
                                     [0 1 2 3 ...dia_len[i]-1]]
            """
            dia_idx = np.array(np.diag_indices(dia_len[i])) #ndarry（2，seq）
            for m in range(modal_num):
                for n in range(modal_num):
                    m_start = start + all_length*m
                    n_start = start + all_length*n
                    if m == n:
                        adj[m_start:m_start+dia_len[i], n_start:n_start+dia_len[i]] = sub_adjs[m]
                    else:
                        modal1 = features[m][start:start+dia_len[i]] #modal1，modal2 ->tensor(seq,200)
                        modal2 = features[n][start:start+dia_len[i]] #得到这段对话中两种不同模态的权重矩阵
                        normed_modal1 = modal1.permute(1, 0) / torch.sqrt(torch.sum(modal1.mul(modal1), dim=1)) #dim, length
                        normed_modal2 = modal2.permute(1, 0) / torch.sqrt(torch.sum(modal2.mul(modal2), dim=1)) #dim, length
                        dia_cos_sim = torch.sum(normed_modal1.mul(normed_modal2).permute(1, 0), dim=1) #tensor(seq)
                        dia_cos_sim = dia_cos_sim * 0.99999
                        dia_sim = 1 - torch.acos(dia_cos_sim)/np.pi
                        idx =dia_idx.copy()
                        idx[0] += m_start
                        idx[1] += n_start
                        adj[idx] = dia_sim  ##将同一话语不同模态间的得分放到整体权重矩阵中

            start += dia_len[i]  ##第i段话从哪里开始

        d = adj.sum(1) #tensor(sum_seq) 在维度1 上加和
        D = torch.diag(torch.pow(d, -0.5)) # 建立一个tensor ->tensor(sum_seq,sum_seq) 对角线上值是torch.pow(d, -0.5) 其他全是0
        adj = D.mm(adj).mm(D) # mm 矩阵乘法   得到的是一个归一化的矩阵表示

        """
        新的权重矩阵 知识内部不融合 
        分为只让文本与相对应的知识图卷积中进行融合
        除了文本与相对应的知识在图卷积中进行融合 相对应的不同知识和文本也融合
        """
        adj_l_no_k = adj
        adj_l_nok = torch.zeros(all_length, all_length)  # 全0
        adj_l_nok1 = torch.zeros(all_length, all_length) + torch.eye(all_length) #只在对角线有元素
        adj_l_no_k[:all_length, :all_length] = adj_l_nok1 * adj_l_no_k[:all_length, :all_length]
        adj_l_no_k[all_length:all_length*2, all_length:all_length*2] = adj_l_no_k[all_length:all_length*2, all_length:all_length*2] * adj_l_nok1
        adj_l_no_k[all_length*3:all_length*4, all_length*3:all_length*4] = adj_l_no_k[all_length*3:all_length*4, all_length*3:all_length*4] * adj_l_nok1
        adj_l_no_k[all_length*4:all_length*5, all_length*4:all_length*5] = adj_l_no_k[all_length*4:all_length*5, all_length*4:all_length*5] * adj_l_nok1
        adj_l_no_k[all_length*5:all_length*6, all_length*5:all_length*6] = adj_l_no_k[all_length*5:all_length*6, all_length*5:all_length*6] * adj_l_nok1

        adj_know_and_text = adj_l_no_k  ##除了文本与相对应的知识在图卷积中进行融合 相对应的不同知识和文本也融合
        #
        # adj_l_no_k[ : all_length, all_length : 2 * all_length] = adj_l_nok
        # adj_l_no_k[ : all_length, 2 * all_length :] =  adj_l_nok
        # adj_l_no_k[all_length : 2 * all_length, 2 * all_length :] = adj_l_nok
        # adj_l_no_k[all_length : 2 * all_length, 2 * all_length :] = adj_l_nok
        # adj_know_no_text = adj_l_no_k #只让文本与相对应的知识图卷积中进行融合
        #
        # d = adj_know_and_text.sum(1)  # tensor(sum_seq) 在维度1 上加和
        # D = torch.diag(torch.pow(d, -0.5))  # 建立一个tensor ->tensor(sum_seq,sum_seq) 对角线上值是torch.pow(d, -0.5) 其他全是0
        # adj_know_and_text = D.mm(adj_know_and_text).mm(D)  # mm 矩阵乘法   得到的是一个归一化的矩阵表示
        #
        # d = adj_know_no_text.sum(1)  # tensor(sum_seq) 在维度1 上加和
        # D = torch.diag(torch.pow(d, -0.5))  # 建立一个tensor ->tensor(sum_seq,sum_seq) 对角线上值是torch.pow(d, -0.5) 其他全是0
        # adj_know_no_text = D.mm(adj_know_no_text).mm(D)  # mm 矩阵乘法   得到的是一个归一化的矩阵表示
        #
        #
        # adj_l = adj[all_length * 2:, all_length * 2:]
        # d_l = adj_l.sum(1)  # tensor(sum_seq) 在维度1 上加和
        # D_l = torch.diag(torch.pow(d_l, -0.5))  # 建立一个tensor ->tensor(sum_seq,sum_seq) 对角线上值是torch.pow(d, -0.5) 其他全是0
        # adj_l = D_l.mm(adj_l).mm(D_l)  # mm 矩阵乘法   得到的是一个归一化的矩阵表示
        # ## adj是根据图构建规则构建的全局图矩阵   adj_l是根据图构建规则构建的单文本权重矩阵
        return adj, adj_know_and_text#, adj_l, adj_know_and_text, adj_know_no_text


class MM_GCN2(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha, variant, return_feature, use_residue, new_graph=False, modals='avl',mm_graph='single'):
        super(MM_GCN2, self).__init__()
        self.return_feature = return_feature
        self.use_residue = use_residue
        self.new_graph = new_graph
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        if not return_feature:
            self.fcs.append(nn.Linear(nfeat+nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.mm_graph = mm_graph
        self.modals = modals
        if self.modals == 'al':
            self.fcs.append(nn.Linear(nfeat, nhidden))
        else:
            self.fcs.append(nn.Linear(nfeat, nhidden))
            self.fcs.append(nn.Linear(nfeat, nhidden))

    def cossim(self, x, y):
        a = torch.matmul(x, y)
        b = torch.sqrt(torch.matmul(x, x)) * torch.sqrt(torch.matmul(y, y))
        if b == 0:
            return 0
        else:
            return (a / b)

    def forward(self, a, v, l, dia_len, topicLabel):

        a_ = F.dropout(a, self.dropout, training=self.training)
        
        if self.modals == 'al':
            a_ = F.dropout(a, self.dropout, training=self.training)
            a_ = self.act_fn(self.fcs[0](a_))
            l_ = F.dropout(l, self.dropout, training=self.training)
            l_ = self.act_fn(self.fcs[1](l_))
            x = torch.cat([a_,l_],dim=0)
        else:
            a_ = F.dropout(a, self.dropout, training=self.training)
            a_ = self.act_fn(self.fcs[0](a_))
            l_ = F.dropout(l, self.dropout, training=self.training)
            l_ = self.act_fn(self.fcs[1](l_))
            v_ = F.dropout(v, self.dropout, training=self.training)
            v_ = self.act_fn(self.fcs[2](v_))
            x = torch.cat([a_,v_,l_],dim=0)
        if self.new_graph:
            adj = self.message_passing_relation_graph(x, dia_len)
        else:
            adj = self.create_big_adj(a,v,l,dia_len)
        _layers = []
        layer_inner = x
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        if self.modals == 'al':
            bias_ = layer_inner.shape[0]//2
            layer_inner = torch.cat([layer_inner[:bias_],layer_inner[bias_:]],dim=-1)
        else:
            bias_ = layer_inner.shape[0]//3
            layer_inner = torch.cat([layer_inner[:bias_],layer_inner[bias_:2*bias_],layer_inner[bias_*2:]],dim=-1)
        if self.use_residue:
            layer_inner = torch.cat([l, layer_inner], dim=-1)
        if not self.return_feature:
            layer_inner = self.fcs[-1](layer_inner)
            layer_inner = F.log_softmax(layer_inner, dim=1)
        return layer_inner

    def create_big_adj(self, a, v, l, dia_len):
        adj = torch.zeros((3*l.shape[0], 3*l.shape[0]))
        all_length = l.shape[0]
        features = [a, v, l]
        start = 0
        for i in range(len(dia_len)):
            sub_adjs = []
            for x in features:
                sub_adj = torch.zeros((dia_len[i], dia_len[i]))
                temp = x[start:start + dia_len[i]]
                vec_length = torch.sqrt(torch.sum(temp.mul(temp), dim=1))
                norm_temp = (temp.permute(1, 0) / vec_length)
                cos_sim_matrix = torch.sum(torch.matmul(norm_temp.unsqueeze(2), norm_temp.unsqueeze(1)), dim=0)
                cos_sim_matrix = cos_sim_matrix * 0.99999
                sim_matrix = torch.acos(cos_sim_matrix)

                sub_adj[:dia_len[i], :dia_len[i]] = sim_matrix
                sub_adjs.append(sub_adj)
            dia_idx = np.array(np.diag_indices(dia_len[i]))
            for m in range(3):
                for n in range(3):
                    m_start = start + all_length*m
                    n_start = start + all_length*n
                    if m == n:
                        adj[m_start:m_start+dia_len[i], n_start:n_start+dia_len[i]] = sub_adjs[m]
                    else:
                        idx =dia_idx.copy()
                        idx[0] += m_start
                        idx[1] += n_start
                        adj[idx] = 0.99999

            start += dia_len[i]
        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D).to(device)

        return adj

    def message_passing_wo_speaker(self, x,dia_len, topicLabel):

        if self.modals != 'al':
            adj = torch.zeros((x.shape[0], x.shape[0]))+torch.eye(x.shape[0])
            modal_index_bias = int(x.shape[0]//3)
            start = 0
            for i in range(len(dia_len)):
                for j in range(dia_len[i]-1):
                    for pin in range(dia_len[i] - 1-j):
                        xz=start+j
                        yz=xz+pin+1
                        f = self.atom_calculate_edge_weight(x[xz], x[yz])
                        Aij = 1 - math.acos(f) / math.pi
                        adj[xz][yz] = Aij
                        adj[yz][xz] = Aij
                        
                        xz = modal_index_bias+start+j
                        yz = xz+pin+1
                        f = self.atom_calculate_edge_weight(x[xz], x[yz])
                        Aij = 1 - math.acos(f) / math.pi
                        adj[xz][yz] = Aij
                        adj[yz][xz] = Aij

                        xz = modal_index_bias*2+start+j
                        yz = xz+pin+1
                        f = self.atom_calculate_edge_weight(x[xz], x[yz])
                        Aij = 1 - math.acos(f) / math.pi
                        adj[xz][yz] = Aij
                        adj[yz][xz] = Aij

                start+=dia_len[i]

            if self.mm_graph == 'single':
                for i in range(sum(dia_len)):
                    xz = i
                    yz = modal_index_bias+i
                    f = self.atom_calculate_edge_weight(x[xz], x[yz])
                    Aij = 1 - math.acos(f) / math.pi
                    adj[xz][yz] = Aij
                    adj[yz][xz] = Aij

                    xz = i 
                    yz = modal_index_bias*2+i
                    f = self.atom_calculate_edge_weight(x[xz], x[yz])
                    Aij = 1 - math.acos(f) / math.pi
                    adj[xz][yz] = Aij
                    adj[yz][xz] = Aij   

                    xz = modal_index_bias+i 
                    yz = modal_index_bias*2+i
                    f = self.atom_calculate_edge_weight(x[xz], x[yz])
                    Aij = 1 - math.acos(f) / math.pi
                    adj[xz][yz] = Aij
                    adj[yz][xz] = Aij
            elif self.mm_graph == 'window':
                window_size = 10
                start = 0
                for i in range(len(dia_len)):
                    for j in range(dia_len[i]):
                        xz = start + j
                        left = max(j-window_size,0)
                        right = min(j+window_size,dia_len[i])
                        for pin in range(left, right):
                            yz = modal_index_bias + start + pin
                            f = self.atom_calculate_edge_weight(x[xz], x[yz])
                            Aij = 1 - math.acos(f) / math.pi
                            adj[xz][yz] = Aij
                            adj[yz][xz] = Aij
                        
                        xz = start + j
                        for pin in range(left, right):
                            yz = modal_index_bias*2 + start + pin
                            f = self.atom_calculate_edge_weight(x[xz], x[yz])
                            Aij = 1 - math.acos(f) / math.pi
                            adj[xz][yz] = Aij
                            adj[yz][xz] = Aij

                        xz = modal_index_bias + start +j
                        for pin in range(left, right):
                            yz = modal_index_bias*2 + start + pin
                            f = self.atom_calculate_edge_weight(x[xz], x[yz])
                            Aij = 1 - math.acos(f) / math.pi
                            adj[xz][yz] = Aij
                            adj[yz][xz] = Aij
                    start += dia_len[i]
            elif self.mm_graph == 'fc':
                start = 0
                for i in range(len(dia_len)):
                    for j in range(dia_len[i]):
                        for pin in range(j, dia_len[i]):
                            xz=start+j
                            yz=modal_index_bias+pin
                            f = self.atom_calculate_edge_weight(x[xz], x[yz])
                            Aij = 1 - math.acos(f) / math.pi
                            adj[xz][yz] = Aij
                            adj[yz][xz] = Aij

                            xz=start+j
                            yz=modal_index_bias*2+pin
                            f = self.atom_calculate_edge_weight(x[xz], x[yz])
                            Aij = 1 - math.acos(f) / math.pi
                            adj[xz][yz] = Aij
                            adj[yz][xz] = Aij

                            xz=modal_index_bias+start+j
                            yz=modal_index_bias*2+pin
                            f = self.atom_calculate_edge_weight(x[xz], x[yz])
                            Aij = 1 - math.acos(f) / math.pi
                            adj[xz][yz] = Aij
                            adj[yz][xz] = Aij
            else:
                print('mm_graph set fault, chech self.mm_graph of this class, the value should in single, window or fc')
                print('However the value is', self.mm_graph)
        else:
            adj = torch.zeros((x.shape[0], x.shape[0]))+torch.eye(x.shape[0])
            modal_index_bias = int(x.shape[0]/3)

            start = 0
            for i in range(len(dia_len)):
                for j in range(dia_len[i]-1):
                    for pin in range(dia_len[i] - 1-j):
                        xz=start+j
                        yz=xz+pin+1
                        f = self.atom_calculate_edge_weight(x[xz], x[yz])
                        Aij = 1 - math.acos(f) / math.pi
                        adj[xz][yz] = Aij
                        adj[yz][xz] = Aij
                        
                        xz = modal_index_bias+start+j
                        yz = xz+pin+1
                        f = self.atom_calculate_edge_weight(x[xz], x[yz])
                        Aij = 1 - math.acos(f) / math.pi
                        adj[xz][yz] = Aij
                        adj[yz][xz] = Aij
                start+=dia_len[i]

            if self.mm_graph == 'single':
                for i in range(sum(dia_len)):
                    xz = i
                    yz = modal_index_bias+i
                    f = self.atom_calculate_edge_weight(x[xz], x[yz])
                    Aij = 1 - math.acos(f) / math.pi
                    adj[xz][yz] = Aij
                    adj[yz][xz] = Aij

            elif self.mm_graph == 'window':
                window_size = 10
                start = 0
                for i in range(len(dia_len)):
                    for j in range(dia_len[i]):
                        xz = start + j
                        left = max(j-window_size,0)
                        right = min(j+window_size,dia_len[i])
                        for pin in range(left, right):
                            yz = modal_index_bias + start + pin
                            f = self.atom_calculate_edge_weight(x[xz], x[yz])
                            Aij = 1 - math.acos(f) / math.pi
                            adj[xz][yz] = Aij
                            adj[yz][xz] = Aij
                    start += dia_len[i]
            elif self.mm_graph == 'fc':
                start = 0
                for i in range(len(dia_len)):
                    for j in range(dia_len[i]):
                        for pin in range(j, dia_len[i]):
                            xz=start+j
                            yz=modal_index_bias+pin
                            f = self.atom_calculate_edge_weight(x[xz], x[yz])
                            Aij = 1 - math.acos(f) / math.pi
                            adj[xz][yz] = Aij
                            adj[yz][xz] = Aij
            else:
                print('mm_graph set fault, chech self.mm_graph of this class, the value should in single, window or fc')
                print('However the value is', self.mm_graph)

        d = adj.sum(1)
        D=torch.diag(torch.pow(d,-0.5))
        adj = D.mm(adj).mm(D).to(device)

        return adj

    def atom_calculate_edge_weight(self, x, y):
        f = self.cossim(x, y)
        if f >1 and f <1.05:
            f = 1
        elif f< -1 and f>-1.05:
            f = -1
        elif f>=1.05 or f<=-1.05:
            print('cos = {}'.format(f))
        return f

    def message_passing_directed_speaker(self, x, dia_len, qmask):
        total_len = sum(dia_len)
        adj = torch.zeros((total_len, total_len))+torch.eye(total_len)
        start = 0
        use_utterance_edge=False
        for (i, len_) in enumerate(dia_len):
            speaker0 = []
            speaker1 = []
            for (j, speaker) in enumerate(qmask[i][0:len_]):
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

        d = adj.sum(1)
        D=torch.diag(torch.pow(d,-0.5))
        adj = D.mm(adj).mm(D).to(device)
        
        return adj.to(device)

    def message_passing_relation_graph(self, x, dia_len):
        total_len = sum(dia_len)
        adj = torch.zeros((total_len, total_len))+torch.eye(total_len)
        window_size = 10
        start = 0
        for (i, len_) in enumerate(dia_len):
            edge_set = []
            for k in range(len_):
                left = max(0, k-window_size)
                right = min(len_-1, k+window_size)
                edge_set = edge_set + [str(i)+'_'+str(j) for i in range(left, right) for j in range(i+1, right+1)]
            edge_set = [[start+int(str_.split('_')[0]),start+int(str_.split('_')[1])] for str_ in list(set(edge_set))]
            for left, right in edge_set:
                f = self.atom_calculate_edge_weight(x[left], x[right])
                Aij = 1-math.acos(f) / math.pi
                adj[left][right] = Aij
                adj[right][left] = Aij
            start+=dia_len[i]

        d = adj.sum(1)
        D=torch.diag(torch.pow(d,-0.5))
        adj = D.mm(adj).mm(D).to(device)
        
        return adj.to(device)
