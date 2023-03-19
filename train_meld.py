import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

from model import MaskedNLLLoss, LSTMModel, GRUModel, DialogRNNModel, DialogueGCNModel, FocalLoss, Pretreatment
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support
import pandas as pd
import pickle as pk
import datetime
import ipdb
from dataset2 import get_IEMOCAP_loaders, get_MELD_loaders, get_EmoryNLP_loaders

# We use seed = 27350 for reproduction of the results reported in the paper.
seed = 68218 #random.randint(1, 100000) #67519 #68218 meld 50578
print(seed)


# t
def seed_everything():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# t
def train_or_eval_graph_model(args, model, pre_model, loss_function, dataloader, epoch, cuda, modals, optimizer=None,
                              train=False, dataset='IEMOCAP'):
    """
    输入为模型 model -->model.py--DialogueGCNModel()    损失函数 loss_function -->= nn.NLLLoss()  数据 dataloader
    输出为
    avg_loss,       这个dataloder的平均损失 标量
    avg_accuracy,   这个dataloder的平均准确率 标量
    labels,         这个dataloder的标签 ndarry
    preds,          这个dataloder的预测值 ndarry
    avg_fscore,     这个dataloder的平均f1得分 标量
    vids,           这个dataloder的每个batch的标记 ndarry
    ei, et, en, el  都是none
    """
    losses, preds, labels = [], [], []
    scores, vids = [], []

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []

    if cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    """
    刚开始提取出来的特征 进行预处理 主要是进行维度的变换 不涉及网络层
    """
    seed_everything()
    for data in dataloader:  ##data ->>list 长度为7  一个batch中的数据
        if train:
            optimizer.zero_grad()
        """
        初始数据的获取
        r1, r2, r3, r4            seq, batch, dim(1024)   原始句子级文本特征
        x1, x2, x3, x4, x5, x6
        o1, o2, o3                seq, batch, dim(768)    原始的根据句子得到的九种知识
        label                     batch, seq              句子的情感标签   
        front_sdj_den, back_sdj_den, global_sdj_den     batch, seq, seq   分别标记着句子间是否需要注意 需要注意用1 不用注意用0
        s_mask    batch, seq, seq     标记两个句子是否是同一说话人说的 是的用1 不是用0
        s_mask_onehot     batch, seq, seq, party（2）    party表示对话人数量     和 s_mask 表示同样的意义 没有使用
        speakers           batch, seq            记录每句话说话人的标记 如（0, 1, 2等）    没有话语的用-1填充
        qmask,             seq, batch, party(人数)   记录该句话是谁说的 用d独热向量来表示
        umask              batch, seq      记录该位置有没有话语  有话语用1表示 没话语用0表示
        其中还有一个值用_省略了 该值是 lengths   batch      记录每段对话的话语数量  
        """
        r1, r2, r3, r4, \
        x1, x2, x3, x4, x5, x6, \
        o1, o2, o3, \
        label, front_sdj_den, back_sdj_den, global_sdj_den, \
        s_mask, s_mask_onehot, _, \
        speakers, qmask, umask = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        """
        这里是对知识和文本向量的预处理 将文本和知识变成多少维度 方便后边使用
        """
        textf, acouf, visuf, k1, k2, k3, k4, know_k1, know_k2, know_k3 = pre_model(r1, r2, r3, r4, x1, x2, x3, x4, x5, x6, o1, o2, o3)

        if args.multi_modal:  # args.multi_modal = Ture 但是 mm_fusion_mthd 不属于这个语句下的几项
            if args.mm_fusion_mthd == 'concat':
                if modals == 'avl':
                    textf = torch.cat([acouf, visuf, textf], dim=-1)
                elif modals == 'av':
                    textf = torch.cat([acouf, visuf], dim=-1)
                elif modals == 'vl':
                    textf = torch.cat([visuf, textf], dim=-1)
                elif modals == 'al':
                    textf = torch.cat([acouf, textf], dim=-1)
                else:
                    raise NotImplementedError
            elif args.mm_fusion_mthd == 'gated':
                textf = textf
        else:
            if modals == 'a':
                textf = acouf
            elif modals == 'v':
                textf = visuf
            elif modals == 'l':
                textf = textf
            else:
                raise NotImplementedError

        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in
                   range(len(umask))]  ##lengths ->list:batch  表示每段对话所含话语数量
        """ multi_modal = Ture 且 mm_fusion_mthd=='concat_subsequently'
            执行这一项 进到model.py--DialogueGCNModel()
            输入 textf, tensor(seq,batch,100) 文本
                qmask, tensor(seq,batch,2) 说话人矩阵 独热向量表示
                umask, tensor(batch,seq) 说话人标记向量 有话语的用1 没话语的用0
                lengths, list:batch  表示每段对话所含话语数量
                acouf, tensor(seq,batch,1582) 声音
                visuf, tensor(seq,batch,342) 视觉
                global_sdj_den  应注意的话语 batch,seq,seq
            返回的是 log_prob ->tensor(sum_seq,classic) 话语情感分类 
            e_i, e_n, e_t, e_l都是 none
        """
        if args.multi_modal and args.mm_fusion_mthd == 'gated':
            log_prob, e_i, e_n, e_t, e_l = model(textf, qmask, umask, lengths, acouf, visuf)
        elif args.multi_modal and args.mm_fusion_mthd == 'concat_subsequently':
            log_prob, e_i, e_n, e_t, e_l, loss1, prob = model(label, train, textf, qmask, umask, lengths,
                                                              global_sdj_den, front_sdj_den, back_sdj_den, s_mask, k1,
                                                              k2, k3, k4, acouf, visuf, know_k1, know_k2, know_k3)
        else:
            log_prob, e_i, e_n, e_t, e_l = model(textf, qmask, umask, lengths)

        """ 
            label ->tensor(sum_seq) 将原lable进行变换 变成拼接起来的形式 
            loss_function()是 NLLLoss() 输入 log_prob->tensor(sum_seq,classic)  label->tensor(sum_seq)
            输出 loss tensor() 
            preds 是一个list 每次将整个batch的预测标签转成 ndaary 放进去 每个list元素为 ndaary:sum_seq
            labels 是同样的操作 每个list元素为 ndaary:sum_seq
            losses 一样 每个 losses 中元素都是这个batch得到的损失标量
        """
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        loss = loss_function(log_prob, label) - (loss1 / args.batch_size)

        if args.use_crf_classic:
            preds.append(prob.cpu().numpy())
        else:
            preds.append(torch.argmax(log_prob, 1).cpu().numpy())

        labels.append(label.cpu().numpy())
        losses.append(loss.item())

        if train:  ##训练集的话就进行反向传播 更新
            loss.backward()
            optimizer.step()

    """
        进行至这里 就是一个epoch中 train_loader 或 valid_loader 或 test_loader 的完成 
        np.concatenate() 将 preds 列表的每个元素拼接起来    labels 同样
    """

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    """ 
        vids 是list 每个batch有一个vids list : batch数量
        ei et en el 都是none
        labels  preds  vids  都变成 ndaary 为这个 dataloder 中的标签和预测值
    """

    vids += data[-1]
    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    labels = np.array(labels)
    preds = np.array(preds)
    vids = np.array(vids)

    """ 
        然后计算平均损失 平均准确率 平均f1得分
        accuracy_score(labels, preds) 输入是  labels ->ndaary:5810  preds ->ndaary:5810
        f1_score(labels,preds, average='weighted') 输入多加一个权重 
    """

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, vids, ei, et, en, el


def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    """

    """
    losses, preds, labels, masks = [], [], [], []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    max_sequence_len = []

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything()
    for data in dataloader:
        if train:
            optimizer.zero_grad()

        max_sequence_len.append(textf.size(0))

        log_prob, alpha, alpha_f, alpha_b, _ = model(textf, qmask, umask)
        lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])
        labels_ = label.view(-1)
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_, 1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), []

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)

    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, [alphas, alphas_f, alphas_b, vids]


## t
if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = './saved/IEMOCAP/'

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=True, help='does not use GPU')
    parser.add_argument('--base-model', default='LSTM', help='图卷积前对文本处理的递归模型，必须是DialogRNN/LSTM/GRU之一')
    parser.add_argument('--graph-model', action='store_true', default=True, help='是否使用图网络')
    parser.add_argument('--nodal-attention', action='store_true', default=True, help='是否在图模型中使用节点注意:Paper中的方程4,5,6')
    parser.add_argument('--windowp', type=int, default=10, help='在过去话语图模型中构造边的上下文窗口大小')
    parser.add_argument('--windowf', type=int, default=10, help='在未来话语的图模型中构造边的上下文窗口大小')
    parser.add_argument('--lr', type=float, default=0.0025, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=3e-05, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.17, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=50, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')
    parser.add_argument('--attention', default='general', help='DialogRNN模型中的注意类型')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='是否记录到tensorboard中去')
    parser.add_argument('--graph_type', default='MMGCN', help='使用的图卷积方法/GCN3/DeepGCN/MMGCN/MMGCN2')
    parser.add_argument('--use_topic', action='store_true', default=False, help='whether to use topic information')
    parser.add_argument('--multiheads', type=int, default=6, help='multiheads')
    parser.add_argument('--graph_construct', default='direct', help='single/window/fc for MMGCN2; direct/full for others')
    parser.add_argument('--use_gcn', action='store_true', default=False, help='whether to combine spectral and none-spectral methods or not')
    parser.add_argument('--use_residue', action='store_true', default=True, help='是否使用图卷积后的与之前的拼接起来')
    parser.add_argument('--multi_modal', action='store_true', default=True, help='是否使用多模信息')
    parser.add_argument('--mm_fusion_mthd', default='concat_subsequently', help='如何使用这些多模信息: concat, gated, concat_subsequently')
    parser.add_argument('--modals', default='avl', help='modals to fusion')
    parser.add_argument('--av_using_lstm', action='store_true', default=False, help='在知识向量进入图卷积前 经过了线性层 是否还要经过双向lstm')
    parser.add_argument('--Deep_GCN_nlayers', type=int, default=32, help='卷积层的深度')

    parser.add_argument('--Dataset', default='MELD', help='dataset to train and test')

    parser.add_argument('--use_speaker', action='store_true', default=False, help='是否在文本表示中加入说话人向量')
    parser.add_argument('--use_modal', action='store_true', default=False, help='是否使用模态/知识嵌入')
    parser.add_argument('--use_pickle', action='store_true', default=False, help='是否记录最佳数据')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
    parser.add_argument('--lamda', type=float, default=0.5, help='lamda')

    parser.add_argument('--pre_know', type=int, default=3, help='外部知识的预处理方式')
    parser.add_argument('--text_model', type=int, default=2, help='用哪个文本向量来作为代表')
    parser.add_argument('--text_norm', type=int, default=2, help='文本向量的归一化方式')
    parser.add_argument('--gnn_layers', type=int, default=1, help='每次顺序层的层数')
    parser.add_argument('--attn_type', type=str, default='rgcn', choices=['mgcn', 'linear', 'bilinear', 'rgcn'], help='顺序注意层的注意方式')
    parser.add_argument('--batch_integration', action='store_true', default=False, help='是否要对batch进行重整')

    parser.add_argument('--hidden_dim', type=int, default=128, help='进到顺序注意层的维度 也是参与图卷积的维度')
    parser.add_argument('--D_audio', type=int, default=300, help='知识维度从预处理出来和初始要进去图卷积的维度')
    parser.add_argument('--D_visual', type=int, default=300, help='知识维度从预处理出来和初始要进去图卷积的维度')
    parser.add_argument('--D_text', type=int, default=100, help='文本维度从预处理出来和初始要进去图卷积的文本向量维度')
    parser.add_argument('--D_m', type=int, default=1024, help='初始提取的文本向量维度 即要进入预处理模块的文本维度')
    parser.add_argument('--D_s', type=int, default=768, help='初始提取的知识向量维度 即要进入预处理模块的知识维度')
    parser.add_argument('--order_know_dim', type=int, default=100, help='进到顺序注意层的知识维度')
    parser.add_argument('--graph_fusion_dim', type=int, default=256, help='图融合过程中使用的维度 三种模式的维度在图融合过程中需相同')
    parser.add_argument('--use_density', action='store_true', default=True, help='创建文本和知识内部权重矩阵过程中 是否使用密度注意方式')
    parser.add_argument('--use_know_graph', action='store_true', default=True, help='是否将知识也参与图卷积')
    parser.add_argument('--use_order', action='store_true', default=False, help='是否使用文本顺序注意')
    parser.add_argument('--text_and_know', default='tnok', help='知识内部也融合all 让文本与自己的知识融合和不同来源的知识融合tandk 否则 只让文本和知识融合tnok 不同知识间没有边 其他都有tink')
    parser.add_argument('--know_fusion_att', default='selfk', help='知识全融合all 让自己的内知识融合selfk 只密度注意知识融合denk')

    parser.add_argument('--know_fusion_num', type=int, default=5, help='参与图卷积的知识数量')
    parser.add_argument('--use_huge_adj', action='store_true', default=False, help='是否使用多知识用于卷积')
    parser.add_argument('--only_text', action='store_true', default=True, help='在知识参与图卷积后 在分类时是否只使用文本向量')
    parser.add_argument('--use_lstm_classic', action='store_true', default=False, help='最后分类前是否要用lstm隐藏层')
    parser.add_argument('--use_crf_classic', action='store_true', default=False, help='最后分类前要不要加crf层')
    parser.add_argument('--use_cnn', action='store_true', default=False, help='要不要使用cnn')
    parser.add_argument('--use_cnnd', action='store_true', default=True, help='要不要使用深度cnn')



    args = parser.parse_args()
    today = datetime.datetime.now()
    print(args)
    if args.av_using_lstm:  ## av_using_lstm=False
        name_ = args.mm_fusion_mthd + '_' + args.modals + '_' + args.graph_type + '_' + args.graph_construct + 'using_lstm_' + args.Dataset
    else:  ## name_ = 'concat_subsequently_avl_MMGCN_direct4_IEMOCAP'
        name_ = args.mm_fusion_mthd + '_' + args.modals + '_' + args.graph_type + '_' + args.graph_construct + str(
            args.Deep_GCN_nlayers) + '_' + args.Dataset
    if args.use_speaker:  ## name_ = 'concat_subsequently_avl_MMGCN_direct4_IEMOCAP_speaker'
        name_ = name_ + '_speaker'
    if args.use_modal:  ##  False
        name_ = name_ + '_modal'

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:  ##  tensorboard = ture  画图工具
        from torch.utils.tensorboard.writer import SummaryWriter

        # from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    cuda = args.cuda  # False
    n_epochs = args.epochs  # 20
    batch_size = args.batch_size  # 16
    modals = args.modals  # ‘avl’
    feat2dim = {'IS10': 1582, '3DCNN': 512, 'textCNN': 100, 'bert': 768, 'denseface': 342, 'MELD_text': 600,
                'MELD_audio': 300}
    D_audio = 300  # feat2dim['IS10'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_audio'] #1582    最初进去的知识维度
    D_visual = 300  # feat2dim['denseface'] #342
    D_text = feat2dim['textCNN'] if args.Dataset == 'IEMOCAP' else feat2dim['MELD_text']  # 100

    seed_everything()
    pre_model = Pretreatment(args)

    """确定D_m的值 即网络融合过程中节点的维度
    """
    if args.multi_modal:  # Ture 是否将知识加入使用图卷积
        if args.mm_fusion_mthd == 'concat':  # 这是指开始前就将各类向量拼接 并确定后边要用的维度 而D_mmm_fusion_mthd = 'concat_subsequently'
            if modals == 'avl':
                D_m = D_audio + D_visual + D_text
            elif modals == 'av':
                D_m = D_audio + D_visual
            elif modals == 'al':
                D_m = D_audio + D_text
            elif modals == 'vl':
                D_m = D_visual + D_text
            else:
                raise NotImplementedError
        else:
            D_m = args.D_text  # 100
    else:
        if modals == 'a':
            D_m = D_audio
        elif modals == 'v':
            D_m = D_visual
        elif modals == 'l':
            D_m = args.D_text
        else:
            raise NotImplementedError

    D_g = 150
    D_p = 150
    D_e = int(args.graph_fusion_dim / 2)
    D_h = 100
    D_a = 100
    graph_h = args.hidden_dim  # 具体图卷积过程中的隐藏态维度
    n_speakers = 9 if args.Dataset == 'MELD' else 2
    n_classes = 7 if args.Dataset == 'MELD' else 6 if args.Dataset == 'IEMOCAP' else 1
    """
    模型的确定
    根据graph_model来判断是否使用图网络 确定model是什么 
    如果不用图网络 那就看 base_model 是什么 来决定model是什么
    """
    if args.graph_model:  ## graph_model=True   是否使用图网络
        seed_everything()
        model = DialogueGCNModel(args, D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                                 n_speakers=n_speakers,
                                 max_seq_len=200,
                                 n_classes=n_classes, )

        print('Graph NN with', args.base_model, 'as base model.')
        name = 'Graph'

    else:
        if args.base_model == 'DialogRNN':
            model = DialogRNNModel(D_m, D_g, D_p, D_e, D_h, D_a,
                                   n_classes=n_classes,
                                   listener_state=args.active_listener,
                                   context_attention=args.attention,
                                   dropout_rec=args.rec_dropout,
                                   dropout=args.dropout)

            print('Basic Dialog RNN Model.')


        elif args.base_model == 'GRU':
            model = GRUModel(D_m, D_e, D_h,
                             n_classes=n_classes,
                             dropout=args.dropout)

            print('Basic GRU Model.')


        elif args.base_model == 'LSTM':
            model = LSTMModel(D_m, D_e, D_h,
                              n_classes=n_classes,
                              dropout=args.dropout)

            print('Basic LSTM Model.')

        else:
            print('Base model must be one of DialogRNN/LSTM/GRU/Transformer')
            raise NotImplementedError

        name = 'Base'

    if cuda:
        model.cuda()

    if args.Dataset == 'MELD':
        loss_weights = torch.FloatTensor([0.30427062, 1.19699616, 5.47007183, 1.95437696,
                                          0.84847735, 5.42461417, 1.21859721])
    """ 损失函数选择
        'MELD' 执行 loss_function = FocalLoss()
        'IEMOCAP'执行 loss_function  = nn.NLLLoss()     
    """
    if args.Dataset == 'MELD':
        loss_function = FocalLoss() #FocalLoss()
    else:
        if args.class_weight:
            if args.graph_model:
                loss_function = nn.NLLLoss(loss_weights.cuda() if cuda else loss_weights)
            else:
                loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        else:
            if args.graph_model:
                loss_function = nn.NLLLoss()
            else:
                loss_function = MaskedNLLLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)  # weight_decay=args.l2

    if args.Dataset == 'MELD':
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=0)
    else:
        print("There is no such dataset")

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []

    test_label = False
    if test_label:
        state = torch.load('best_model_IEMOCAP/model.pth')
        model.load_state_dict(state['net'])
        test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_graph_model(args, model,
                                                                                                           loss_function,
                                                                                                           test_loader,
                                                                                                           0, cuda,
                                                                                                           args.modals,
                                                                                                           dataset=args.Dataset)

    for e in range(n_epochs):
        start_time = time.time()

        if args.graph_model:  # T
            train_loss, train_acc, _, _, train_fscore, _, _, _, _, _ = train_or_eval_graph_model(args, model, pre_model,
                                                                                                 loss_function,
                                                                                                 train_loader, e, cuda,
                                                                                                 args.modals, optimizer,
                                                                                                 True,
                                                                                                 dataset=args.Dataset)
            valid_loss, valid_acc, _, _, valid_fscore, _, _, _, _, _ = train_or_eval_graph_model(args, model, pre_model,
                                                                                                 loss_function,
                                                                                                 valid_loader, e, cuda,
                                                                                                 args.modals,
                                                                                                 dataset=args.Dataset)
            test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_graph_model(args,
                                                                                                               model,
                                                                                                               pre_model,
                                                                                                               loss_function,
                                                                                                               test_loader,
                                                                                                               e, cuda,
                                                                                                               args.modals,
                                                                                                               dataset=args.Dataset)
            all_fscore.append(test_fscore)

        else:
            train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(model, loss_function, train_loader, e,
                                                                                  optimizer, True)
            valid_loss, valid_acc, _, _, _, valid_fscore, _ = train_or_eval_model(model, loss_function, valid_loader, e)
            test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(model,
                                                                                                                 loss_function,
                                                                                                                 test_loader,
                                                                                                                 e)
            all_fscore.append(test_fscore)

        """
        记录整个循环中最佳测试集loss
        同时根据最佳测试集f1值 来确定需要输出哪次epoch的具体测试集数据 
        每运行10次输出一次 找每10个里的最佳
        """
        if best_loss == None or best_loss > test_loss:
            best_loss = test_loss
        if best_fscore == None or best_fscore < test_fscore:
            best_fscore, best_label, best_pred = test_fscore, test_label, test_pred
            # test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_graph_model(model, pre_model,loss_function, test_loader, e, cuda, args.modals, dataset=args.Dataset)
        """将每一次epoch的运行结果记录在tensorboard中
        """
        if args.tensorboard:
            writer.add_scalar('test: accuracy', test_acc, e)
            writer.add_scalar('test: fscore', test_fscore, e)
            writer.add_scalar('train: accuracy', train_acc, e)
            writer.add_scalar('train: fscore', train_fscore, e)
        print(
            'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
            format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc,
                   test_fscore, round(time.time() - start_time, 2)))

        if (e + 1) % 10 == 0:  # 每十个epoch输出一次具体数据
            print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
            print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))

    ## 以下不用管
    if args.tensorboard:
        writer.close()
    print('Test performance..')
    print('F-Score:', max(all_fscore))

    if args.use_pickle:
        if not os.path.exists(
                "record_{}_{}_{}_{}_{}.pk".format(today.year, today.month, today.day, today.hour, today.minute)):
            with open("record_{}_{}_{}_{}_{}.pk".format(today.year, today.month, today.day, today.hour, today.minute),
                      'wb') as f:
                pk.dump({}, f)
        with open("record_{}_{}_{}_{}_{}.pk".format(today.year, today.month, today.day, today.hour, today.minute),
                  'rb') as f:
            record = pk.load(f)
        key_ = name_
        if record.get(key_, False):
            record[key_].append(max(all_fscore))
        else:
            record[key_] = [max(all_fscore)]
        if record.get(key_ + 'record', False):
            record[key_ + 'record'].append(
                classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
        else:
            record[key_ + 'record'] = [classification_report(best_label, best_pred, sample_weight=best_mask, digits=4)]
        with open("record_{}_{}_{}_{}_{}.pk".format(today.year, today.month, today.day, today.hour, today.minute),
                  'wb') as f:
            pk.dump(record, f)


