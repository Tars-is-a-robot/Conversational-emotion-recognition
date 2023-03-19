import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import json
from model import MaskedNLLLoss
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data.sampler import SubsetRandomSampler

def get_train_valid_sampler(trainset, valid=0.1, dataset='IEMOCAP'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

class IEMOCAPDataset(Dataset):

    def __init__(self, split):
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.speakers, self.labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open('iemocap/iemocap_features_roberta.pkl', 'rb'),
                          encoding='latin1')  # 除了最后三个是list类型 其他都是dict类型

        self.xIntent, self.xAttr, self.xNeed, self.xWant, self.xEffect, self.xReact, self.oWant, self.oEffect, self.oReact \
            = pickle.load(open('iemocap/iemocap_features_comet.pkl', 'rb'), encoding='latin1')  # 都是dict类型

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]), \
               torch.FloatTensor(self.roberta2[vid]), \
               torch.FloatTensor(self.roberta3[vid]), \
               torch.FloatTensor(self.roberta4[vid]), \
               torch.FloatTensor(self.xIntent[vid]), \
               torch.FloatTensor(self.xAttr[vid]), \
               torch.FloatTensor(self.xNeed[vid]), \
               torch.FloatTensor(self.xWant[vid]), \
               torch.FloatTensor(self.xEffect[vid]), \
               torch.FloatTensor(self.xReact[vid]), \
               torch.FloatTensor(self.oWant[vid]), \
               torch.FloatTensor(self.oEffect[vid]), \
               torch.FloatTensor(self.oReact[vid]), \
               torch.FloatTensor([[1, 0] if x == 'M' else [0, 1] for x in self.speakers[vid]]), \
               torch.FloatTensor([1] * len(self.labels[vid])), \
               torch.LongTensor(self.labels[vid]), \
               torch.FloatTensor([1 if x == 'M' else 0 for x in self.speakers[vid]]).int().view(-1).tolist(), \
               vid

    def __len__(self):
        return self.len

    # def get_adj_v1(self, speakers, max_dialog_len,lengths):
    #     '''
    #     get adj matrix
    #     :param speakers:  (B, N)  这是list 每个元素也都是list
    #     :param max_dialog_len:
    #     :return:
    #         adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
    #     '''
    #     adj = []
    #     for m,speaker in enumerate(speakers):
    #         a = torch.zeros(max_dialog_len, max_dialog_len)
    #         for x in range(lengths[m]):
    #             for y in range(x):
    #                 a[x,y] = 3
    #         for i,s in enumerate(speaker):
    #             cnt = 0
    #             for j in range(i - 1, -1, -1):
    #                 a[i,j] = cnt +1
    #                 if speaker[j] == s:
    #                     cnt += 1
    #                     if cnt==3:
    #                         break
    #         adj.append(a)
    #     return torch.stack(adj)
    #
    # def get_adj3(self,adj):
    #     adj3 = []
    #     B = adj.size()[0]
    #     S = adj.size()[1]
    #     for i in range(B):
    #         temp = adj[i,:,:]
    #         a = torch.zeros_like(adj[i, :, :])
    #         for x in range(temp.size()[0]):
    #             for y in range(x):
    #                 if temp[x, y] == 3.:
    #                     a[x, y] = 1
    #
    #         adj3.append(a)
    #     return torch.stack(adj3)
    #
    # def get_adj2(self,adj):
    #     adj2 = []
    #     B = adj.size()[0]
    #     S = adj.size()[1]
    #     for i in range(B):
    #         temp = adj[i,:,:]
    #         a = torch.zeros_like(adj[i, :, :])
    #         for x in range(temp.size()[0]):
    #             for y in range(x):
    #                 if temp[x, y] == 2.:
    #                     a[x, y] = 1
    #
    #         adj2.append(a)
    #     return torch.stack(adj2)
    #
    # def get_adj1(self,adj):
    #     adj1 = []
    #     B = adj.size()[0]
    #     S = adj.size()[1]
    #     for i in range(B):
    #         temp = adj[i,:,:]
    #         a = torch.zeros_like(adj[i, :, :])
    #         for x in range(temp.size()[0]):
    #             for y in range(x):
    #                 if temp[x, y] == 1.:
    #                     a[x, y] = 1
    #
    #         adj1.append(a)
    #     return torch.stack(adj1)

    # def get_adj_number(self, speakers, max_dialog_len,lengths):
    #     '''
    #     输入是这段对话的说话人 以及这个batch中对话最多的话语数量      根据出现的说话人数量来进行注意矩阵的输出
    #     get adj matrix
    #     :param speakers:  (B, N)
    #     :param max_dialog_len:
    #     :return:
    #         adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
    #     '''
    #     adj = []   ##空列表
    #     for m,speaker in enumerate(speakers):    ##拿出一个batch来 形成话语密度注意矩阵
    #         a = torch.zeros(max_dialog_len, max_dialog_len)  # 全0
    #         for i in range(lengths[m]):  ## i指这个batch中的第几句话  speaker.size()[0] 可以改小 用lengths
    #             num_list = [0] * 2  ## 出现的说话人数量列表  2 这里可以在别的数据集中改大 说话人数量
    #             s_list =[]      ## 出现的说话人
    #             max_s = 0       ## 出现的说话人标记
    #             for m,s in enumerate(speaker[:i]):   ## 这句话之前的话语    循环完之后会得到 在第i句话时 出现的说话人数量列表和出现的说话人列表
    #                 if max_s < s:
    #                     max_s = s      ## 标记最大说话人标号
    #                 if s in s_list:  # 如果这句话的说话人本来就在说话人列表中 那么这句话的说话人密度要加1
    #                     num_list[s] = num_list[s] + 1
    #                 else:           # 如果这句话的说话人不在说话人列表中 就把他加到说话人列表中 并且这句话的说话人数量要加1
    #                     s_list.append(s)
    #                     num_list[s] = num_list[s] + 1
    #                     """
    #                     循环完成 得到 第i句话时的说话人标记的最大值 出现的说话人密度列表 出现的说话人列表
    #                     """
    #             s_tens = torch.Tensor(s_list).int()  ##出现的说话人索引
    #             num_tens = torch.Tensor(num_list)  ##所有说话人在此时的出现次数
    #             base_min = torch.min(torch.index_select(num_tens, dim=0, index=s_tens)).item()  ## 得到暂时出现的说话人 出现次数最少的那个说话人的次数
    #             new_num_tens = torch.zeros(max_s + 1)     ## 一个新的全部为0的说话人次数张量
    #
    #             for j in range(i - 1, -1, -1):
    #                 a[i, j] = 1              ## 第i段对话与它前边的那个第j段对话要注意
    #                 new_num_tens[speaker[j]] = new_num_tens[speaker[j]] + 1     ## 将第j句话的说话人 放进到新的存放说话人出现次数的张量中
    #                 if torch.min(torch.index_select(new_num_tens, dim=0, index=s_tens)) < base_min:  ##现在出现的说话人次数太少 那就继续循环 否则 就跳出来
    #                     pass
    #                 else:
    #                     break
    #         adj.append(a)
    #     return torch.stack(adj)

    def get_adj_density(self, speakers, max_dialog_len, lengths):
        '''
        输入是这段对话的说话人 以及这个batch中对话最多的话语数量      根据出现的说话人密度来进行注意矩阵的输出
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        front_adj = []  ##前向空列表
        back_adj = []  ##后向空列表
        global_adj = []  ##全局空列表
        for m, speaker in enumerate(speakers):  ##拿出一个batch来 形成话语密度注意矩阵
            speaker = torch.Tensor(speaker).int()  ##将元素是整数型的
            front_a = torch.zeros(max_dialog_len, max_dialog_len)  # 全0前向注意矩阵
            back_a = torch.zeros(max_dialog_len, max_dialog_len)  # 全0后向注意矩阵
            global_a = torch.zeros(max_dialog_len, max_dialog_len)  # 全0全局注意矩阵

            for i in range(0, lengths[m]):  ## i指这个batch中的第几句话  第一句话和最后一句话要使用特殊方式 第一句不算前向 最后一句不算后向

                num_front_list = [0] * 2  ## 出现的前向说话人数量列表  2 这里可以在别的数据集中改大 说话人数量  num_list
                s_front_list = []  ## 出现的前向说话人  s_list
                max_s_front = 0  ## 出现的前向说话人标记  max_s

                """
                在这句话之前的话语进行循环 循环完成 得到 第i句话时的说话人标记的最大值 出现的说话人密度列表 出现的说话人列表
                """
                if i != 0:  ##i是0 就不要进行前向操作了
                    for _, s in enumerate(speaker[:i]):  ## 这句话之前的话语    循环完之后会得到 在第i句话时 出现的说话人数量列表和出现的说话人列表
                        if max_s_front < s:
                            max_s_front = s  ## 标记最大说话人标号
                        if s in s_front_list:  # 如果这句话的说话人本来就在说话人列表中 那么这句话的说话人密度要加1
                            num_front_list[s] = num_front_list[s] + 1
                        else:  # 如果这句话的说话人不在说话人列表中 就把他加到说话人列表中 并且这句话的说话人数量要加1
                            s_front_list.append(s)
                            num_front_list[s] = num_front_list[s] + 1
                    """
                    将前向循环得到的list 变成张量 方便后边操作
                    """
                    s_front_tens = torch.Tensor(s_front_list).int()  ##出现的前向说话人
                    num_front_tens = torch.Tensor(num_front_list)  ##所有出现的前向说话人在此时的出现次数
                    base_front_min = (torch.min(torch.index_select(num_front_tens, dim=0,
                                                                   index=s_front_tens)).item()) / i  ## 得到暂时出现的前向说话人 出现次数最少的那个说话人的次数 再得到此时的密度
                    new_front_num_tens = torch.zeros(max_s_front + 1)  ## 一个新的全部为0的说话人次数张量
                    """
                    得到数据后 前向密度说话人注意的计算                
                    """
                    for j in range(i - 1, -1, -1):
                        front_a[i, j] = 1  ## 第i段对话与它前边的那个第j段对话要注意
                        new_front_num_tens[speaker[j]] = new_front_num_tens[
                                                             speaker[j]] + 1  ## 将第j句话的说话人 放进到新的存放说话人出现次数的张量中
                        if (torch.min(torch.index_select(new_front_num_tens, dim=0, index=s_front_tens))) / (
                                i - j) < base_front_min:  ##现在出现的说话人次数太少 那就继续循环 否则 就跳出来  (i-j) 为现在话语数量
                            pass
                        else:
                            break

                num_back_list = [0] * 2  ## 出现的后向说话人数量列表  2 这里可以在别的数据集中改大 说话人数量
                s_back_list = []  ## 出现的后向说话人
                max_s_back = 0  ## 出现的后向说话人标记

                if i != lengths[m] - 1:  ##i是最后一句 就不要进行后向操作了
                    for _, s in enumerate(speaker[i + 1:]):  ## 这句话之后的话语    循环完之后会得到 在第i句话时 后边出现的说话人数量列表和出现的说话人列表
                        if max_s_back < s:
                            max_s_back = s  ## 标记最大说话人标号
                        if s in s_back_list:  # 如果这句话的说话人本来就在说话人列表中 那么这句话的说话人数量要加1
                            num_back_list[s] = num_back_list[s] + 1
                        else:  # 如果这句话的说话人不在说话人列表中 就把他加到说话人列表中 并且这句话的说话人数量要加1
                            s_back_list.append(s)
                            num_back_list[s] = num_back_list[s] + 1
                    """
                    将后向循环得到的list 变成张量 方便后边操作
                    """
                    s_back_tens = torch.Tensor(s_back_list).int()  ##出现的后向说话人
                    num_back_tens = torch.Tensor(num_back_list)  ##所有出现的后向说话人在此时的出现次数
                    base_back_min = (torch.min(torch.index_select(num_back_tens, dim=0, index=s_back_tens)).item()) / (
                                lengths[m] - i)  ## 得到出现的后向说话人 出现次数最少的那个说话人的次数 再得到此时的密度
                    new_back_num_tens = torch.zeros(max_s_back + 1)  ## 一个新的全部为0的说话人次数张量
                    """
                    得到数据后 后向密度说话人注意的计算                
                    """
                    for j in range(i + 1, lengths[m]):
                        back_a[i, j] = 1  ## 第i段对话与它前边的那个第j段对话要注意
                        new_back_num_tens[speaker[j]] = new_back_num_tens[
                                                            speaker[j]] + 1  ## 将第j句话的说话人 放进到新的存放说话人出现次数的张量中
                        if (torch.min(torch.index_select(new_back_num_tens, dim=0, index=s_back_tens))) / (
                                j - i) < base_back_min:  ##现在出现的说话人次数太少 那就继续循环 否则 就跳出来  (i-j) 为现在话语数量
                            pass
                        else:
                            break

            """
            每当把一个batch循环完成 就把这个batch形成的注意矩阵放到列表中 下一个batch时再将注意矩阵归零        
            """
            global_a = front_a + back_a + torch.eye(max_dialog_len) ##形成全局注意矩阵
            front_adj.append(front_a)
            back_adj.append(back_a)
            global_adj.append(global_a)



        return torch.stack(front_adj), torch.stack(back_adj),torch.stack(global_adj)

    def get_s_mask(self, speakers, max_dialog_len):
        '''
        :param speakers:
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:]
         means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         表示节点i的前辈的说话人信息，其中1表示同一说话人，0表示不同说话人
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        s_mask = []
        s_mask_onehot = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype=torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 2)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i, j] = 1
                        s_onehot[i, j, 1] = 1
                    else:
                        s_onehot[i, j, 0] = 1

            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)

    def collate_fn(self, data):  # 如何将多个样本数据拼接成一个batch
        max_dialog_len = max([len(d[16]) for d in data])     ## 这个得到的是每个batch中对话话语最多的是多少 一个整数

        r1 = pad_sequence([d[0] for d in data])  # (B, N, D)
        r2 = pad_sequence([d[1] for d in data])
        r3 = pad_sequence([d[2] for d in data])
        r4 = pad_sequence([d[3] for d in data])

        xi = pad_sequence([d[4] for d in data])
        xa = pad_sequence([d[5] for d in data])
        xn = pad_sequence([d[6] for d in data])
        xw = pad_sequence([d[7] for d in data])
        xe = pad_sequence([d[8] for d in data])
        xr = pad_sequence([d[9] for d in data])
        ow = pad_sequence([d[10] for d in data])
        oe = pad_sequence([d[11] for d in data])
        oR = pad_sequence([d[12] for d in data])

        """以上这些都是张量 seq，batch，dim r1--r4 dim=1024
            剩下九个知识的  dim=768
            以下 qmask 是  seq,batch,party(2)     标记每句话是谁说的（用独热向量来标记）
            umask      是 batch，seq     这一段对话有话语的地方就用1来标记 没话语就用0标记
            labels  每句话的情感标签 batch，seq
            lengths 张量 batch   里边元素是每段对话的话语数量
            s_mask, s_mask_onehot    s_mask 是 batch，seq，seq  记录的是每句话的说话人 如果是同一人 就用1 不是就用0   s_mask_onehot batch，seq，seq，2  没有使用 就是将对话人是否是同一人标记成独热的
            speakers  batch，seq   里边标记了每句话的说话人编号 
            front_sdj_den, back_sdj_den       batch，seq，seq  得到的前向和后向要注意的句子
            vid  是每段对话的标记 一个字符串
        """

        qmask = pad_sequence([d[13] for d in data])  #
        umask = pad_sequence([d[14] for d in data],batch_first=True)  #
        labels = pad_sequence([d[15] for d in data], batch_first=True, padding_value=-1)  # batch,seq

        lengths = torch.LongTensor([d[14].size(0) for d in data])  ##每段对话的真实长度
        # adj = self.get_adj_v1([d[16] for d in data], max_dialog_len,lengths)  ##把说话人矩阵和对话长度放进去
        s_mask, s_mask_onehot = self.get_s_mask([d[16] for d in data], max_dialog_len)  #
        speakers = pad_sequence([torch.LongTensor(d[16]) for d in data], batch_first=True ,padding_value=-1)  ##说话人矩阵
        front_sdj_den, back_sdj_den, global_sdj_den= self.get_adj_density([d[16] for d in data], max_dialog_len,lengths)  ##把说话人矩阵和对话长度放进去
        vid = [d[17] for d in data]
        # adj1 = self.get_adj1(adj)
        # adj2 = self.get_adj2(adj)
        # adj3 = self.get_adj3(adj)
        # adj_num = self.get_adj_number([d[16] for d in data], max_dialog_len,lengths)  ##把说话人矩阵和对话长度放进去
        return r1,r2,r3,r4,xi,xa,xn,xw,xe,xr,ow,oe,oR,labels, front_sdj_den, back_sdj_den,global_sdj_den,s_mask,s_mask_onehot,lengths,speakers,qmask,umask,vid

def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset('train')
    validset = IEMOCAPDataset('valid')
    testset = IEMOCAPDataset('test')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)


    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader



class MELDDataset(Dataset):

    def __init__(self, split, classify='emotion'):
        '''
        label index mapping =
        '''
        self.speakers, self.emotion_labels, self.sentiment_labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open('meld/meld_features_roberta.pkl', 'rb'), encoding='latin1')

        self.xIntent, self.xAttr, self.xNeed, self.xWant, self.xEffect, self.xReact, self.oWant, self.oEffect, self.oReact \
            = pickle.load(open('meld/meld_features_comet.pkl', 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]

        if classify == 'emotion':
            self.labels = self.emotion_labels
        else:
            self.labels = self.sentiment_labels

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]), \
               torch.FloatTensor(self.roberta2[vid]), \
               torch.FloatTensor(self.roberta3[vid]), \
               torch.FloatTensor(self.roberta4[vid]), \
               torch.FloatTensor(self.xIntent[vid]), \
               torch.FloatTensor(self.xAttr[vid]), \
               torch.FloatTensor(self.xNeed[vid]), \
               torch.FloatTensor(self.xWant[vid]), \
               torch.FloatTensor(self.xEffect[vid]), \
               torch.FloatTensor(self.xReact[vid]), \
               torch.FloatTensor(self.oWant[vid]), \
               torch.FloatTensor(self.oEffect[vid]), \
               torch.FloatTensor(self.oReact[vid]), \
               torch.FloatTensor(self.speakers[vid]), \
               torch.FloatTensor([1] * len(self.labels[vid])), \
               torch.LongTensor(self.labels[vid]), \
               torch.argmax(torch.IntTensor(self.speakers[vid]), 1).tolist(), \
               vid

    def __len__(self):
        return self.len

    def get_adj_density(self, speakers, max_dialog_len, lengths):
        '''
        输入是这段对话的说话人 以及这个batch中对话最多的话语数量      根据出现的说话人密度来进行注意矩阵的输出
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        front_adj = []  ##前向空列表
        back_adj = []  ##后向空列表
        global_adj = []  ##全局空列表
        for m, speaker in enumerate(speakers):  ##拿出一个batch来 形成话语密度注意矩阵
            speaker = torch.Tensor(speaker).int()  ##将元素是整数型的
            front_a = torch.zeros(max_dialog_len, max_dialog_len)  # 全0前向注意矩阵
            back_a = torch.zeros(max_dialog_len, max_dialog_len)  # 全0后向注意矩阵
            global_a = torch.zeros(max_dialog_len, max_dialog_len)  # 全0全局注意矩阵

            for i in range(0, lengths[m]):  ## i指这个batch中的第几句话  第一句话和最后一句话要使用特殊方式 第一句不算前向 最后一句不算后向

                num_front_list = [0] * 9  ## 出现的前向说话人数量列表  2 这里可以在别的数据集中改大 说话人数量  num_list
                s_front_list = []  ## 出现的前向说话人  s_list
                max_s_front = 0  ## 出现的前向说话人标记  max_s

                """
                在这句话之前的话语进行循环 循环完成 得到 第i句话时的说话人标记的最大值 出现的说话人密度列表 出现的说话人列表
                """
                if i != 0:  ##i是0 就不要进行前向操作了
                    for _, s in enumerate(speaker[:i]):  ## 这句话之前的话语    循环完之后会得到 在第i句话时 出现的说话人数量列表和出现的说话人列表
                        if max_s_front < s:
                            max_s_front = s  ## 标记最大说话人标号
                        if s in s_front_list:  # 如果这句话的说话人本来就在说话人列表中 那么这句话的说话人密度要加1
                            num_front_list[s] = num_front_list[s] + 1
                        else:  # 如果这句话的说话人不在说话人列表中 就把他加到说话人列表中 并且这句话的说话人数量要加1
                            s_front_list.append(s)
                            num_front_list[s] = num_front_list[s] + 1
                    """
                    将前向循环得到的list 变成张量 方便后边操作
                    """
                    s_front_tens = torch.Tensor(s_front_list).int()  ##出现的前向说话人
                    num_front_tens = torch.Tensor(num_front_list)  ##所有出现的前向说话人在此时的出现次数
                    base_front_min = (torch.min(torch.index_select(num_front_tens, dim=0,
                                                                   index=s_front_tens)).item()) / i  ## 得到暂时出现的前向说话人 出现次数最少的那个说话人的次数 再得到此时的密度
                    new_front_num_tens = torch.zeros(max_s_front + 1)  ## 一个新的全部为0的说话人次数张量
                    """
                    得到数据后 前向密度说话人注意的计算                
                    """
                    for j in range(i - 1, -1, -1):
                        front_a[i, j] = 1  ## 第i段对话与它前边的那个第j段对话要注意
                        new_front_num_tens[speaker[j]] = new_front_num_tens[
                                                             speaker[j]] + 1  ## 将第j句话的说话人 放进到新的存放说话人出现次数的张量中
                        if (torch.min(torch.index_select(new_front_num_tens, dim=0, index=s_front_tens))) / (
                                i - j) < base_front_min:  ##现在出现的说话人次数太少 那就继续循环 否则 就跳出来  (i-j) 为现在话语数量
                            pass
                        else:
                            break

                num_back_list = [0] * 9  ## 出现的后向说话人数量列表  2 这里可以在别的数据集中改大 说话人数量
                s_back_list = []  ## 出现的后向说话人
                max_s_back = 0  ## 出现的后向说话人标记

                if i != lengths[m] - 1:  ##i是最后一句 就不要进行后向操作了
                    for _, s in enumerate(speaker[i + 1:]):  ## 这句话之后的话语    循环完之后会得到 在第i句话时 后边出现的说话人数量列表和出现的说话人列表
                        if max_s_back < s:
                            max_s_back = s  ## 标记最大说话人标号
                        if s in s_back_list:  # 如果这句话的说话人本来就在说话人列表中 那么这句话的说话人数量要加1
                            num_back_list[s] = num_back_list[s] + 1
                        else:  # 如果这句话的说话人不在说话人列表中 就把他加到说话人列表中 并且这句话的说话人数量要加1
                            s_back_list.append(s)
                            num_back_list[s] = num_back_list[s] + 1
                    """
                    将后向循环得到的list 变成张量 方便后边操作
                    """
                    s_back_tens = torch.Tensor(s_back_list).int()  ##出现的后向说话人
                    num_back_tens = torch.Tensor(num_back_list)  ##所有出现的后向说话人在此时的出现次数
                    base_back_min = (torch.min(torch.index_select(num_back_tens, dim=0, index=s_back_tens)).item()) / (
                            lengths[m] - i)  ## 得到出现的后向说话人 出现次数最少的那个说话人的次数 再得到此时的密度
                    new_back_num_tens = torch.zeros(max_s_back + 1)  ## 一个新的全部为0的说话人次数张量
                    """
                    得到数据后 后向密度说话人注意的计算                
                    """
                    for j in range(i + 1, lengths[m]):
                        back_a[i, j] = 1  ## 第i段对话与它前边的那个第j段对话要注意
                        new_back_num_tens[speaker[j]] = new_back_num_tens[
                                                            speaker[j]] + 1  ## 将第j句话的说话人 放进到新的存放说话人出现次数的张量中
                        if (torch.min(torch.index_select(new_back_num_tens, dim=0, index=s_back_tens))) / (
                                j - i) < base_back_min:  ##现在出现的说话人次数太少 那就继续循环 否则 就跳出来  (i-j) 为现在话语数量
                            pass
                        else:
                            break

            """
            每当把一个batch循环完成 就把这个batch形成的注意矩阵放到列表中 下一个batch时再将注意矩阵归零        
            """
            global_a = front_a + back_a + torch.eye(max_dialog_len)  ##形成全局注意矩阵
            front_adj.append(front_a)
            back_adj.append(back_a)
            global_adj.append(global_a)

        return torch.stack(front_adj), torch.stack(back_adj), torch.stack(global_adj)

    def get_s_mask(self, speakers, max_dialog_len):
        '''
        :param speakers:
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:]
         means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         表示节点i的前辈的说话人信息，其中1表示同一说话人，0表示不同说话人
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        s_mask = []
        s_mask_onehot = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype=torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 2)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i, j] = 1
                        s_onehot[i, j, 1] = 1
                    else:
                        s_onehot[i, j, 0] = 1

            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)

    def collate_fn(self, data):  # 如何将多个样本数据拼接成一个batch
        max_dialog_len = max([len(d[16]) for d in data])  ## 这个得到的是每个batch中对话话语最多的是多少 一个整数

        r1 = pad_sequence([d[0] for d in data])  # (B, N, D)
        r2 = pad_sequence([d[1] for d in data])
        r3 = pad_sequence([d[2] for d in data])
        r4 = pad_sequence([d[3] for d in data])

        xi = pad_sequence([d[4] for d in data])
        xa = pad_sequence([d[5] for d in data])
        xn = pad_sequence([d[6] for d in data])
        xw = pad_sequence([d[7] for d in data])
        xe = pad_sequence([d[8] for d in data])
        xr = pad_sequence([d[9] for d in data])
        ow = pad_sequence([d[10] for d in data])
        oe = pad_sequence([d[11] for d in data])
        oR = pad_sequence([d[12] for d in data])

        """以上这些都是张量 seq，batch，dim r1--r4 dim=1024
            剩下九个知识的  dim=768
            以下 qmask 是  seq,batch,party(2)     标记每句话是谁说的（用独热向量来标记）
            umask      是 batch，seq     这一段对话有话语的地方就用1来标记 没话语就用0标记
            labels  每句话的情感标签 batch，seq
            lengths 张量 batch   里边元素是每段对话的话语数量
            s_mask, s_mask_onehot    s_mask 是 batch，seq，seq  记录的是每句话的说话人 如果是同一人 就用1 不是就用0   s_mask_onehot batch，seq，seq，2  没有使用 就是将对话人是否是同一人标记成独热的
            speakers  batch，seq   里边标记了每句话的说话人编号 
            front_sdj_den, back_sdj_den       batch，seq，seq  得到的前向和后向要注意的句子
            vid  是每段对话的标记 一个字符串
        """

        qmask = pad_sequence([d[13] for d in data])  #
        umask = pad_sequence([d[14] for d in data], batch_first=True)  #
        labels = pad_sequence([d[15] for d in data], batch_first=True, padding_value=-1)  # batch,seq

        lengths = torch.LongTensor([d[14].size(0) for d in data])  ##每段对话的真实长度
        # adj = self.get_adj_v1([d[16] for d in data], max_dialog_len,lengths)  ##把说话人矩阵和对话长度放进去
        s_mask, s_mask_onehot = self.get_s_mask([d[16] for d in data], max_dialog_len)  #
        speakers = pad_sequence([torch.LongTensor(d[16]) for d in data], batch_first=True, padding_value=-1)  ##说话人矩阵
        front_sdj_den, back_sdj_den, global_sdj_den = self.get_adj_density([d[16] for d in data], max_dialog_len, lengths)  ##把说话人矩阵和对话长度放进去
        vid = [d[17] for d in data]
        # adj1 = self.get_adj1(adj)
        # adj2 = self.get_adj2(adj)
        # adj3 = self.get_adj3(adj)
        # adj_num = self.get_adj_number([d[16] for d in data], max_dialog_len,lengths)  ##把说话人矩阵和对话长度放进去
        return r1, r2, r3, r4, xi, xa, xn, xw, xe, xr, ow, oe, oR, labels, front_sdj_den, back_sdj_den, global_sdj_den, s_mask, s_mask_onehot, lengths, speakers, qmask, umask, vid

def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset(split = 'train')
    validset = MELDDataset(split = 'valid')
    testset = MELDDataset(split = 'test')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)


    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader



class EmoryNLPDataset(Dataset):

    def __init__(self, split, classify='emotion'):

        '''
        label index mapping =  {'Joyful': 0, 'Mad': 1, 'Peaceful': 2, 'Neutral': 3, 'Sad': 4, 'Powerful': 5, 'Scared': 6}
        '''

        self.speakers, self.emotion_labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainId, self.testId, self.validId \
            = pickle.load(open('emorynlp/emorynlp_features_roberta.pkl', 'rb'), encoding='latin1')

        sentiment_labels = {}
        for item in self.emotion_labels:
            array = []
            # 0 negative, 1 neutral, 2 positive
            for e in self.emotion_labels[item]:
                if e in [1, 4, 6]:
                    array.append(0)
                elif e == 3:
                    array.append(1)
                elif e in [0, 2, 5]:
                    array.append(2)
            sentiment_labels[item] = array

        self.xIntent, self.xAttr, self.xNeed, self.xWant, self.xEffect, self.xReact, self.oWant, self.oEffect, self.oReact \
            = pickle.load(open('emorynlp/emorynlp_features_comet.pkl', 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]

        if classify == 'emotion':
            self.labels = self.emotion_labels
        elif classify == 'sentiment':
            self.labels = sentiment_labels

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]), \
               torch.FloatTensor(self.roberta2[vid]), \
               torch.FloatTensor(self.roberta3[vid]), \
               torch.FloatTensor(self.roberta4[vid]), \
               torch.FloatTensor(self.xIntent[vid]), \
               torch.FloatTensor(self.xAttr[vid]), \
               torch.FloatTensor(self.xNeed[vid]), \
               torch.FloatTensor(self.xWant[vid]), \
               torch.FloatTensor(self.xEffect[vid]), \
               torch.FloatTensor(self.xReact[vid]), \
               torch.FloatTensor(self.oWant[vid]), \
               torch.FloatTensor(self.oEffect[vid]), \
               torch.FloatTensor(self.oReact[vid]), \
               torch.FloatTensor(self.speakers[vid]), \
               torch.FloatTensor([1] * len(self.labels[vid])), \
               torch.LongTensor(self.labels[vid]), \
               torch.argmax(torch.IntTensor(self.speakers[vid]), 1).tolist(), \
               vid

    def __len__(self):
        return self.len

    def get_adj_density(self, speakers, max_dialog_len, lengths):
        '''
        输入是这段对话的说话人 以及这个batch中对话最多的话语数量      根据出现的说话人密度来进行注意矩阵的输出
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        front_adj = []  ##前向空列表
        back_adj = []  ##后向空列表
        global_adj = []  ##全局空列表
        for m, speaker in enumerate(speakers):  ##拿出一个batch来 形成话语密度注意矩阵
            speaker = torch.Tensor(speaker).int()  ##将元素是整数型的
            front_a = torch.zeros(max_dialog_len, max_dialog_len)  # 全0前向注意矩阵
            back_a = torch.zeros(max_dialog_len, max_dialog_len)  # 全0后向注意矩阵
            global_a = torch.zeros(max_dialog_len, max_dialog_len)  # 全0全局注意矩阵

            for i in range(0, lengths[m]):  ## i指这个batch中的第几句话  第一句话和最后一句话要使用特殊方式 第一句不算前向 最后一句不算后向

                num_front_list = [0] * 9  ## 出现的前向说话人数量列表  2 这里可以在别的数据集中改大 说话人数量  num_list
                s_front_list = []  ## 出现的前向说话人  s_list
                max_s_front = 0  ## 出现的前向说话人标记  max_s

                """
                在这句话之前的话语进行循环 循环完成 得到 第i句话时的说话人标记的最大值 出现的说话人密度列表 出现的说话人列表
                """
                if i != 0:  ##i是0 就不要进行前向操作了
                    for _, s in enumerate(speaker[:i]):  ## 这句话之前的话语    循环完之后会得到 在第i句话时 出现的说话人数量列表和出现的说话人列表
                        if max_s_front < s:
                            max_s_front = s  ## 标记最大说话人标号
                        if s in s_front_list:  # 如果这句话的说话人本来就在说话人列表中 那么这句话的说话人密度要加1
                            num_front_list[s] = num_front_list[s] + 1
                        else:  # 如果这句话的说话人不在说话人列表中 就把他加到说话人列表中 并且这句话的说话人数量要加1
                            s_front_list.append(s)
                            num_front_list[s] = num_front_list[s] + 1
                    """
                    将前向循环得到的list 变成张量 方便后边操作
                    """
                    s_front_tens = torch.Tensor(s_front_list).int()  ##出现的前向说话人
                    num_front_tens = torch.Tensor(num_front_list)  ##所有出现的前向说话人在此时的出现次数
                    base_front_min = (torch.min(torch.index_select(num_front_tens, dim=0,
                                                                   index=s_front_tens)).item()) / i  ## 得到暂时出现的前向说话人 出现次数最少的那个说话人的次数 再得到此时的密度
                    new_front_num_tens = torch.zeros(max_s_front + 1)  ## 一个新的全部为0的说话人次数张量
                    """
                    得到数据后 前向密度说话人注意的计算                
                    """
                    for j in range(i - 1, -1, -1):
                        front_a[i, j] = 1  ## 第i段对话与它前边的那个第j段对话要注意
                        new_front_num_tens[speaker[j]] = new_front_num_tens[
                                                             speaker[j]] + 1  ## 将第j句话的说话人 放进到新的存放说话人出现次数的张量中
                        if (torch.min(torch.index_select(new_front_num_tens, dim=0, index=s_front_tens))) / (
                                i - j) < base_front_min:  ##现在出现的说话人次数太少 那就继续循环 否则 就跳出来  (i-j) 为现在话语数量
                            pass
                        else:
                            break

                num_back_list = [0] * 9  ## 出现的后向说话人数量列表  2 这里可以在别的数据集中改大 说话人数量
                s_back_list = []  ## 出现的后向说话人
                max_s_back = 0  ## 出现的后向说话人标记

                if i != lengths[m] - 1:  ##i是最后一句 就不要进行后向操作了
                    for _, s in enumerate(speaker[i + 1:]):  ## 这句话之后的话语    循环完之后会得到 在第i句话时 后边出现的说话人数量列表和出现的说话人列表
                        if max_s_back < s:
                            max_s_back = s  ## 标记最大说话人标号
                        if s in s_back_list:  # 如果这句话的说话人本来就在说话人列表中 那么这句话的说话人数量要加1
                            num_back_list[s] = num_back_list[s] + 1
                        else:  # 如果这句话的说话人不在说话人列表中 就把他加到说话人列表中 并且这句话的说话人数量要加1
                            s_back_list.append(s)
                            num_back_list[s] = num_back_list[s] + 1
                    """
                    将后向循环得到的list 变成张量 方便后边操作
                    """
                    s_back_tens = torch.Tensor(s_back_list).int()  ##出现的后向说话人
                    num_back_tens = torch.Tensor(num_back_list)  ##所有出现的后向说话人在此时的出现次数
                    base_back_min = (torch.min(torch.index_select(num_back_tens, dim=0, index=s_back_tens)).item()) / (
                            lengths[m] - i)  ## 得到出现的后向说话人 出现次数最少的那个说话人的次数 再得到此时的密度
                    new_back_num_tens = torch.zeros(max_s_back + 1)  ## 一个新的全部为0的说话人次数张量
                    """
                    得到数据后 后向密度说话人注意的计算                
                    """
                    for j in range(i + 1, lengths[m]):
                        back_a[i, j] = 1  ## 第i段对话与它前边的那个第j段对话要注意
                        new_back_num_tens[speaker[j]] = new_back_num_tens[
                                                            speaker[j]] + 1  ## 将第j句话的说话人 放进到新的存放说话人出现次数的张量中
                        if (torch.min(torch.index_select(new_back_num_tens, dim=0, index=s_back_tens))) / (
                                j - i) < base_back_min:  ##现在出现的说话人次数太少 那就继续循环 否则 就跳出来  (i-j) 为现在话语数量
                            pass
                        else:
                            break

            """
            每当把一个batch循环完成 就把这个batch形成的注意矩阵放到列表中 下一个batch时再将注意矩阵归零        
            """
            global_a = front_a + back_a + torch.eye(max_dialog_len)  ##形成全局注意矩阵
            front_adj.append(front_a)
            back_adj.append(back_a)
            global_adj.append(global_a)

        return torch.stack(front_adj), torch.stack(back_adj), torch.stack(global_adj)

    def get_s_mask(self, speakers, max_dialog_len):
        '''
        :param speakers:
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:]
         means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         表示节点i的前辈的说话人信息，其中1表示同一说话人，0表示不同说话人
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        s_mask = []
        s_mask_onehot = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype=torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 2)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i, j] = 1
                        s_onehot[i, j, 1] = 1
                    else:
                        s_onehot[i, j, 0] = 1

            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)
    def collate_fn(self, data):  # 如何将多个样本数据拼接成一个batch
        max_dialog_len = max([len(d[16]) for d in data])  ## 这个得到的是每个batch中对话话语最多的是多少 一个整数

        r1 = pad_sequence([d[0] for d in data])  # (B, N, D)
        r2 = pad_sequence([d[1] for d in data])
        r3 = pad_sequence([d[2] for d in data])
        r4 = pad_sequence([d[3] for d in data])

        xi = pad_sequence([d[4] for d in data])
        xa = pad_sequence([d[5] for d in data])
        xn = pad_sequence([d[6] for d in data])
        xw = pad_sequence([d[7] for d in data])
        xe = pad_sequence([d[8] for d in data])
        xr = pad_sequence([d[9] for d in data])
        ow = pad_sequence([d[10] for d in data])
        oe = pad_sequence([d[11] for d in data])
        oR = pad_sequence([d[12] for d in data])

        """以上这些都是张量 seq，batch，dim r1--r4 dim=1024
            剩下九个知识的  dim=768
            以下 qmask 是  seq,batch,party(2)     标记每句话是谁说的（用独热向量来标记）
            umask      是 batch，seq     这一段对话有话语的地方就用1来标记 没话语就用0标记
            labels  每句话的情感标签 batch，seq
            lengths 张量 batch   里边元素是每段对话的话语数量
            s_mask, s_mask_onehot    s_mask 是 batch，seq，seq  记录的是每句话的说话人 如果是同一人 就用1 不是就用0   s_mask_onehot batch，seq，seq，2  没有使用 就是将对话人是否是同一人标记成独热的
            speakers  batch，seq   里边标记了每句话的说话人编号 
            front_sdj_den, back_sdj_den       batch，seq，seq  得到的前向和后向要注意的句子
            vid  是每段对话的标记 一个字符串
        """

        qmask = pad_sequence([d[13] for d in data])  #
        umask = pad_sequence([d[14] for d in data], batch_first=True)  #
        labels = pad_sequence([d[15] for d in data], batch_first=True, padding_value=-1)  # batch,seq

        lengths = torch.LongTensor([d[14].size(0) for d in data])  ##每段对话的真实长度
        # adj = self.get_adj_v1([d[16] for d in data], max_dialog_len,lengths)  ##把说话人矩阵和对话长度放进去
        s_mask, s_mask_onehot = self.get_s_mask([d[16] for d in data], max_dialog_len)  #
        speakers = pad_sequence([torch.LongTensor(d[16]) for d in data], batch_first=True, padding_value=-1)  ##说话人矩阵
        front_sdj_den, back_sdj_den, global_sdj_den = self.get_adj_density([d[16] for d in data], max_dialog_len, lengths)  ##把说话人矩阵和对话长度放进去
        vid = [d[17] for d in data]
        # adj1 = self.get_adj1(adj)
        # adj2 = self.get_adj2(adj)
        # adj3 = self.get_adj3(adj)
        # adj_num = self.get_adj_number([d[16] for d in data], max_dialog_len,lengths)  ##把说话人矩阵和对话长度放进去
        return r1, r2, r3, r4, xi, xa, xn, xw, xe, xr, ow, oe, oR, labels, front_sdj_den, back_sdj_den, global_sdj_den, s_mask, s_mask_onehot, lengths, speakers, qmask, umask, vid

    # def collate_fn(self, data):  # 如何将多个样本数据拼接成一个batch
    #     max_dialog_len = max([len(d[16]) for d in data])  ## 这个得到的是每个batch中对话话语最多的是多少 一个整数
    #
    #     r1 = pad_sequence([d[0] for d in data])  # (B, N, D)
    #     r2 = pad_sequence([d[1] for d in data])
    #     r3 = pad_sequence([d[2] for d in data])
    #     r4 = pad_sequence([d[3] for d in data])
    #
    #     xi = pad_sequence([d[4] for d in data])
    #     xa = pad_sequence([d[5] for d in data])
    #     xn = pad_sequence([d[6] for d in data])
    #     xw = pad_sequence([d[7] for d in data])
    #     xe = pad_sequence([d[8] for d in data])
    #     xr = pad_sequence([d[9] for d in data])
    #     ow = pad_sequence([d[10] for d in data])
    #     oe = pad_sequence([d[11] for d in data])
    #     oR = pad_sequence([d[12] for d in data])
    #
    #     """以上这些都是张量 seq，batch，dim r1--r4 dim=1024
    #         剩下九个知识的  dim=768
    #         以下 qmask 是  seq,batch,party(2)     标记每句话是谁说的（用独热向量来标记）
    #         umask      是 batch，seq     这一段对话有话语的地方就用1来标记 没话语就用0标记
    #         labels  每句话的情感标签 batch，seq
    #         lengths 张量 batch   里边元素是每段对话的话语数量
    #         s_mask, s_mask_onehot    s_mask 是 batch，seq，seq  记录的是每句话的说话人 如果是同一人 就用1 不是就用0   s_mask_onehot batch，seq，seq，2  没有使用 就是将对话人是否是同一人标记成独热的
    #         speakers  batch，seq   里边标记了每句话的说话人编号
    #         front_sdj_den, back_sdj_den       batch，seq，seq  得到的前向和后向要注意的句子
    #         vid  是每段对话的标记 一个字符串
    #     """
    #
    #     qmask = pad_sequence([d[13] for d in data])  #
    #     umask = pad_sequence([d[14] for d in data], batch_first=True)  #
    #     labels = pad_sequence([d[15] for d in data], batch_first=True, padding_value=-1)  # batch,seq
    #
    #     lengths = torch.LongTensor([d[14].size(0) for d in data])  ##每段对话的真实长度
    #     # adj = self.get_adj_v1([d[16] for d in data], max_dialog_len,lengths)  ##把说话人矩阵和对话长度放进去
    #     s_mask, s_mask_onehot = self.get_s_mask([d[16] for d in data], max_dialog_len)  #
    #     speakers = pad_sequence([torch.LongTensor(d[16]) for d in data], batch_first=True, padding_value=-1)  ##说话人矩阵
    #     front_sdj_den, back_sdj_den, global_sdj_den = self.get_adj_density([d[16] for d in data], max_dialog_len,
    #                                                                        lengths)  ##把说话人矩阵和对话长度放进去
    #     vid = [d[17] for d in data]
    #     # adj1 = self.get_adj1(adj)
    #     # adj2 = self.get_adj2(adj)
    #     # adj3 = self.get_adj3(adj)
    #     # adj_num = self.get_adj_number([d[16] for d in data], max_dialog_len,lengths)  ##把说话人矩阵和对话长度放进去
    #     return r1, r2, r3, r4, xi, xa, xn, xw, xe, xr, ow, oe, oR, labels, front_sdj_den, back_sdj_den, global_sdj_den, s_mask, s_mask_onehot, lengths, speakers, qmask, umask, vid






def get_EmoryNLP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = EmoryNLPDataset('train')
    validset = EmoryNLPDataset('valid')
    testset = EmoryNLPDataset('test')
    #train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)


    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


class DailyDialogDataset(Dataset):

    def __init__(self, split, classify='emotion'):

        '''
        label index mapping =  {'Joyful': 0, 'Mad': 1, 'Peaceful': 2, 'Neutral': 3, 'Sad': 4, 'Powerful': 5, 'Scared': 6}
        '''

        self.speakers, self.emotion_labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainId, self.testId, self.validId \
            = pickle.load(open('dailydialog/dailydialog_features_roberta.pkl', 'rb'), encoding='latin1')

        sentiment_labels = {}
        for item in self.emotion_labels:
            array = []
            # 0 negative, 1 neutral, 2 positive
            for e in self.emotion_labels[item]:
                if e in [1, 4, 6]:
                    array.append(0)
                elif e == 3:
                    array.append(1)
                elif e in [0, 2, 5]:
                    array.append(2)
            sentiment_labels[item] = array

        self.xIntent, self.xAttr, self.xNeed, self.xWant, self.xEffect, self.xReact, self.oWant, self.oEffect, self.oReact \
            = pickle.load(open('dailydialog/dailydialog_features_comet.pkl', 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]

        if classify == 'emotion':
            self.labels = self.emotion_labels
        elif classify == 'sentiment':
            self.labels = sentiment_labels

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]), \
               torch.FloatTensor(self.roberta2[vid]), \
               torch.FloatTensor(self.roberta3[vid]), \
               torch.FloatTensor(self.roberta4[vid]), \
               torch.FloatTensor(self.xIntent[vid]), \
               torch.FloatTensor(self.xAttr[vid]), \
               torch.FloatTensor(self.xNeed[vid]), \
               torch.FloatTensor(self.xWant[vid]), \
               torch.FloatTensor(self.xEffect[vid]), \
               torch.FloatTensor(self.xReact[vid]), \
               torch.FloatTensor(self.oWant[vid]), \
               torch.FloatTensor(self.oEffect[vid]), \
               torch.FloatTensor(self.oReact[vid]), \
               torch.FloatTensor([[1,0] if x=='0' else [0,1] for x in self.speakers[vid]]), \
               torch.FloatTensor([1] * len(self.labels[vid])), \
               torch.LongTensor(self.labels[vid]), \
               torch.FloatTensor([1 if x == '0' else 0 for x in self.speakers[vid]]).int().view(-1).tolist(), \
               vid

    def __len__(self):
        return self.len

    def get_adj_density(self, speakers, max_dialog_len, lengths):
        '''
        输入是这段对话的说话人 以及这个batch中对话最多的话语数量      根据出现的说话人密度来进行注意矩阵的输出
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        front_adj = []  ##前向空列表
        back_adj = []  ##后向空列表
        global_adj = []  ##全局空列表
        for m, speaker in enumerate(speakers):  ##拿出一个batch来 形成话语密度注意矩阵
            speaker = torch.Tensor(speaker).int()  ##将元素是整数型的
            front_a = torch.zeros(max_dialog_len, max_dialog_len)  # 全0前向注意矩阵
            back_a = torch.zeros(max_dialog_len, max_dialog_len)  # 全0后向注意矩阵
            global_a = torch.zeros(max_dialog_len, max_dialog_len)  # 全0全局注意矩阵

            for i in range(0, lengths[m]):  ## i指这个batch中的第几句话  第一句话和最后一句话要使用特殊方式 第一句不算前向 最后一句不算后向

                num_front_list = [0] * 9  ## 出现的前向说话人数量列表  2 这里可以在别的数据集中改大 说话人数量  num_list
                s_front_list = []  ## 出现的前向说话人  s_list
                max_s_front = 0  ## 出现的前向说话人标记  max_s

                """
                在这句话之前的话语进行循环 循环完成 得到 第i句话时的说话人标记的最大值 出现的说话人密度列表 出现的说话人列表
                """
                if i != 0:  ##i是0 就不要进行前向操作了
                    for _, s in enumerate(speaker[:i]):  ## 这句话之前的话语    循环完之后会得到 在第i句话时 出现的说话人数量列表和出现的说话人列表
                        if max_s_front < s:
                            max_s_front = s  ## 标记最大说话人标号
                        if s in s_front_list:  # 如果这句话的说话人本来就在说话人列表中 那么这句话的说话人密度要加1
                            num_front_list[s] = num_front_list[s] + 1
                        else:  # 如果这句话的说话人不在说话人列表中 就把他加到说话人列表中 并且这句话的说话人数量要加1
                            s_front_list.append(s)
                            num_front_list[s] = num_front_list[s] + 1
                    """
                    将前向循环得到的list 变成张量 方便后边操作
                    """
                    s_front_tens = torch.Tensor(s_front_list).int()  ##出现的前向说话人
                    num_front_tens = torch.Tensor(num_front_list)  ##所有出现的前向说话人在此时的出现次数
                    base_front_min = (torch.min(torch.index_select(num_front_tens, dim=0,
                                                                   index=s_front_tens)).item()) / i  ## 得到暂时出现的前向说话人 出现次数最少的那个说话人的次数 再得到此时的密度
                    new_front_num_tens = torch.zeros(max_s_front + 1)  ## 一个新的全部为0的说话人次数张量
                    """
                    得到数据后 前向密度说话人注意的计算                
                    """
                    for j in range(i - 1, -1, -1):
                        front_a[i, j] = 1  ## 第i段对话与它前边的那个第j段对话要注意
                        new_front_num_tens[speaker[j]] = new_front_num_tens[
                                                             speaker[j]] + 1  ## 将第j句话的说话人 放进到新的存放说话人出现次数的张量中
                        if (torch.min(torch.index_select(new_front_num_tens, dim=0, index=s_front_tens))) / (
                                i - j) < base_front_min:  ##现在出现的说话人次数太少 那就继续循环 否则 就跳出来  (i-j) 为现在话语数量
                            pass
                        else:
                            break

                num_back_list = [0] * 9  ## 出现的后向说话人数量列表  2 这里可以在别的数据集中改大 说话人数量
                s_back_list = []  ## 出现的后向说话人
                max_s_back = 0  ## 出现的后向说话人标记

                if i != lengths[m] - 1:  ##i是最后一句 就不要进行后向操作了
                    for _, s in enumerate(speaker[i + 1:]):  ## 这句话之后的话语    循环完之后会得到 在第i句话时 后边出现的说话人数量列表和出现的说话人列表
                        if max_s_back < s:
                            max_s_back = s  ## 标记最大说话人标号
                        if s in s_back_list:  # 如果这句话的说话人本来就在说话人列表中 那么这句话的说话人数量要加1
                            num_back_list[s] = num_back_list[s] + 1
                        else:  # 如果这句话的说话人不在说话人列表中 就把他加到说话人列表中 并且这句话的说话人数量要加1
                            s_back_list.append(s)
                            num_back_list[s] = num_back_list[s] + 1
                    """
                    将后向循环得到的list 变成张量 方便后边操作
                    """
                    s_back_tens = torch.Tensor(s_back_list).int()  ##出现的后向说话人
                    num_back_tens = torch.Tensor(num_back_list)  ##所有出现的后向说话人在此时的出现次数
                    base_back_min = (torch.min(torch.index_select(num_back_tens, dim=0, index=s_back_tens)).item()) / (
                            lengths[m] - i)  ## 得到出现的后向说话人 出现次数最少的那个说话人的次数 再得到此时的密度
                    new_back_num_tens = torch.zeros(max_s_back + 1)  ## 一个新的全部为0的说话人次数张量
                    """
                    得到数据后 后向密度说话人注意的计算                
                    """
                    for j in range(i + 1, lengths[m]):
                        back_a[i, j] = 1  ## 第i段对话与它前边的那个第j段对话要注意
                        new_back_num_tens[speaker[j]] = new_back_num_tens[
                                                            speaker[j]] + 1  ## 将第j句话的说话人 放进到新的存放说话人出现次数的张量中
                        if (torch.min(torch.index_select(new_back_num_tens, dim=0, index=s_back_tens))) / (
                                j - i) < base_back_min:  ##现在出现的说话人次数太少 那就继续循环 否则 就跳出来  (i-j) 为现在话语数量
                            pass
                        else:
                            break

            """
            每当把一个batch循环完成 就把这个batch形成的注意矩阵放到列表中 下一个batch时再将注意矩阵归零        
            """
            global_a = front_a + back_a + torch.eye(max_dialog_len)  ##形成全局注意矩阵
            front_adj.append(front_a)
            back_adj.append(back_a)
            global_adj.append(global_a)

        return torch.stack(front_adj), torch.stack(back_adj), torch.stack(global_adj)

    def get_s_mask(self, speakers, max_dialog_len):
        '''
        :param speakers:
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:]
         means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         表示节点i的前辈的说话人信息，其中1表示同一说话人，0表示不同说话人
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        s_mask = []
        s_mask_onehot = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype=torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 2)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i, j] = 1
                        s_onehot[i, j, 1] = 1
                    else:
                        s_onehot[i, j, 0] = 1

            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)

    def collate_fn(self, data):  # 如何将多个样本数据拼接成一个batch
        max_dialog_len = max([len(d[16]) for d in data])  ## 这个得到的是每个batch中对话话语最多的是多少 一个整数

        r1 = pad_sequence([d[0] for d in data])  # (B, N, D)
        r2 = pad_sequence([d[1] for d in data])
        r3 = pad_sequence([d[2] for d in data])
        r4 = pad_sequence([d[3] for d in data])

        xi = pad_sequence([d[4] for d in data])
        xa = pad_sequence([d[5] for d in data])
        xn = pad_sequence([d[6] for d in data])
        xw = pad_sequence([d[7] for d in data])
        xe = pad_sequence([d[8] for d in data])
        xr = pad_sequence([d[9] for d in data])
        ow = pad_sequence([d[10] for d in data])
        oe = pad_sequence([d[11] for d in data])
        oR = pad_sequence([d[12] for d in data])

        """以上这些都是张量 seq，batch，dim r1--r4 dim=1024
            剩下九个知识的  dim=768
            以下 qmask 是  seq,batch,party(2)     标记每句话是谁说的（用独热向量来标记）
            umask      是 batch，seq     这一段对话有话语的地方就用1来标记 没话语就用0标记
            labels  每句话的情感标签 batch，seq
            lengths 张量 batch   里边元素是每段对话的话语数量
            s_mask, s_mask_onehot    s_mask 是 batch，seq，seq  记录的是每句话的说话人 如果是同一人 就用1 不是就用0   s_mask_onehot batch，seq，seq，2  没有使用 就是将对话人是否是同一人标记成独热的
            speakers  batch，seq   里边标记了每句话的说话人编号 
            front_sdj_den, back_sdj_den       batch，seq，seq  得到的前向和后向要注意的句子
            vid  是每段对话的标记 一个字符串
        """

        qmask = pad_sequence([d[13] for d in data])  #
        umask = pad_sequence([d[14] for d in data], batch_first=True)  #
        labels = pad_sequence([d[15] for d in data], batch_first=True, padding_value=-1)  # batch,seq

        lengths = torch.LongTensor([d[14].size(0) for d in data])  ##每段对话的真实长度
        # adj = self.get_adj_v1([d[16] for d in data], max_dialog_len,lengths)  ##把说话人矩阵和对话长度放进去
        s_mask, s_mask_onehot = self.get_s_mask([d[16] for d in data], max_dialog_len)  #
        speakers = pad_sequence([torch.LongTensor(d[16]) for d in data], batch_first=True, padding_value=-1)  ##说话人矩阵
        front_sdj_den, back_sdj_den, global_sdj_den = self.get_adj_density([d[16] for d in data], max_dialog_len,
                                                                           lengths)  ##把说话人矩阵和对话长度放进去
        vid = [d[17] for d in data]
        # adj1 = self.get_adj1(adj)
        # adj2 = self.get_adj2(adj)
        # adj3 = self.get_adj3(adj)
        # adj_num = self.get_adj_number([d[16] for d in data], max_dialog_len,lengths)  ##把说话人矩阵和对话长度放进去
        return r1, r2, r3, r4, xi, xa, xn, xw, xe, xr, ow, oe, oR, labels, front_sdj_den, back_sdj_den, global_sdj_den, s_mask, s_mask_onehot, lengths, speakers, qmask, umask, vid

def get_DailyDialog_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = DailyDialogDataset('train')
    validset = DailyDialogDataset('valid')
    testset = DailyDialogDataset('test')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)


    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader



if __name__ == '__main__':
    # for i in range(trainset.__len__()):
    #     for j in range(17):
    #         if j != 16:
    #             if j==13:
    #                 print(len(trainset.__getitem__(i)[j]))
    #             else:
    #                 print(trainset.__getitem__(i)[j].size())
    #         else:
    #             print(trainset.__getitem__(i)[j])
    batch_size = 16
    train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0,
                                                                  batch_size=2,
                                                                  num_workers=0)

    for data in valid_loader:  # 是训练集的话就把梯度归零

        r1, r2, r3, r4, \
        x1, x2, x3, x4, x5, x6, \
        o1, o2, o3, \
        label, front_sdj_den, back_sdj_den, global_sdj_den, s_mask, s_mask_onehot, lengths, speakers, qmask, umask, vid = data[:]
        # print("#################")
        # print('r1:',type(r1),'r2:',type(r2),'r3:',type(r3),'r4:',type(r4))
        # print('r1:', r1.size(), 'r2:', r2.size(), 'r3:',r3.size(), 'r4:', r4.size())
        # print('x1:',type(x1),'x2:',type(x2),'x3:',type(x3),'x4:',type(x4))
        # print('x1:', x1.size(), 'x2:', x2.size(), 'x3:', x3.size(), 'x4:', x4.size())
        # print('x5:',type(x5),'x6:',type(x6),'o1:',type(o1),'o2:',type(o2))
        # print('x5:', x5.size(), 'x6:', x6.size(), 'o1:', o1.size(), 'o2:', o2.size())
        # print('o3:', type(o3), 'label:', type(label), 'front_sdj_den:', type(front_sdj_den), 's_mask:', type(s_mask))
        # print('o3:', o3.size(), 'label:', label.size(), 'front_sdj_den:', front_sdj_den.size(), 's_mask:', s_mask.size())
        # print('s_mask_onehot:', type(s_mask_onehot), 'lengths:', type(lengths), 'speakers:', type(speakers))
        # print('back_sdj_den:', back_sdj_den.size(), 'global_sdj_den:', global_sdj_den.size())
        print('front_sdj_den:', front_sdj_den, front_sdj_den.size())
        print('speakers:', speakers, speakers.size())
        # print('s_mask_onehot:', s_mask_onehot.size(), 'lengths:', lengths.size(), 'speakers:', speakers.size())
        # print('qmask:', qmask.size(), 'umask:', umask.size())
        # torch.set_printoptions(profile="full")
        # print(qmask)
        # print(lengths[0])
        # print('adj',adj[0])
        # print('adj1',adj1[0])
        # print('adj2',adj2[0])
        # print('adj3',adj3[0])
        # torch.set_printoptions(profile="full")
        # # print(speakers[0])
        # print('front_sdj_den', front_sdj_den[0])
        # print('back_sdj_den', back_sdj_den[0])
        # print('global_sdj_den', global_sdj_den[0])
