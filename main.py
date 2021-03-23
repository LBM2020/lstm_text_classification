import os
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import SequentialSampler
from sklearn.metrics import classification_report

from tool.tool import FGM,seed_everything
from config.config import opt
from process.process import create_tensors,create_dataset,create_dataloader,read_data,load_vector


class model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, num_class, batch_first=True):
        super(model, self).__init__()
        self.layer1 = nn.LSTM(input_dim, hidden_dim, num_layer,bidirectional=True)
        self.layer2 = nn.Linear(hidden_dim*2, num_class)

    def forward(self,inputs):
        layer1_output, layer1_hidden = self.layer1(inputs)
        layer1_output,layer1_output_len = torch.nn.utils.rnn.pad_packed_sequence(layer1_output,batch_first=True)#对压缩后的数据进行解压以符合全连接层的输入格式

        layer2_output = self.layer2(layer1_output)
        layer2_output = layer2_output[:, -1, :]  # 取出一个batch中每个句子最后一个单词的输出向量即该句子的语义向量！！！！！！！!！
        return layer2_output
        # -------或者使用隐藏层向量作为线性层的输入-------
        # layer1_output, layer1_hidden = self.layer1(inputs)
        # layer2_output = self.layer2(layer1_hidden)
        # return layer2_output


def train(model,train_dataloader,valid_dataloader,epoches):
    loss_fun = BCEWithLogitsLoss()#多标签分类任务使用的损失函数
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    #训练和验证
    model.zero_grad()
    #固定种子
    seed_everything(opt['seed'])

    for epoch in range(epoches):
        print(f"开始第{epoch}轮训练")
        fgm = FGM(model)#对抗训练的一部分
        for batch,length in tqdm(train_dataloader):
            #分别取出向量和标签
            contents = [line[0] for line in batch] # line的数据例子为[vector,label],其中vector的维度为[seq_len,dim]
            true_labels = [line[1] for line in batch]

            #对向量进行一定的处理以满足格式要求
            contents = torch.tensor([item.detach().numpy() for item in contents])#当列表中含有多维tensor时，需要经过这一步的处理，不能直接使用torch.tensor(contents)
            contents = torch.nn.utils.rnn.pack_padded_sequence(contents, lengths=length, batch_first=True)#对数据进行压缩，避免补充的0浪费计算资源。

            #模型计算
            #*********正常训练方式*********
            # model_output = model(contents)
            #
            # #计算损失函数、反向传播、优化器优化
            # loss = loss_fun(model_output, torch.tensor(true_labels))  # 交叉熵损失函数的输入model_output是模型预测各类别的概率，CrossEntropyLoss中包含softmax的计算过程，不用人为计算。
            # loss.backward()
            # optimizer.step()
            #
            # model.zero_grad()
            # *********正常训练方式*********

            # *********对抗训练方式*********
            fgm.attack()# 在embedding上添加对抗扰动
            model_output = model(contents)
            loss = loss_fun(model_output, torch.tensor(true_labels))
            loss.backward()# 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            fgm.restore()# 恢复embedding参数
            optimizer.step()
            model.zero_grad()
            # *********对抗训练方式*********

        #验证
        print('')#避免输出在同一行
        print(f"第{epoch}轮验证效果")
        valid(model,valid_dataloader)

        #模型保存
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        output_dir = opt['model_output_path']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_dir = os.path.join(opt['model_output_path'], f'model_checkpoint_epoch{epoch}.pkl')
        torch.save(state,output_dir)

def valid(model,valid_dataloader):
    pre_labels = []
    true_labels = []

    model.eval()
    for batch,length in tqdm(valid_dataloader):
        with torch.no_grad():
            contents = [line[0] for line in batch]  # line的数据例子为[vector,label],其中vector的维度为[seq_len,dim]
            true_label = [line[1] for line in batch]

            contents = torch.tensor([item.detach().numpy() for item in contents])
            contents = torch.nn.utils.rnn.pack_padded_sequence(contents, lengths=length, batch_first=True)

            #模型预测
            model_pre_output = model(contents)
            softmax_output = F.softmax(model_pre_output, dim=-1)
            pre_label = np.argmax(softmax_output,axis=-1)

            #汇总每个batch的预测结果和真实结果，用于计算混淆矩阵
            pre_labels.extend(pre_label.numpy())
            true_labels.extend(torch.tensor(true_label).numpy())

    target_names = opt['target_names']
    print('')  # 避免输出在同一行
    print(classification_report(y_true=np.array(true_labels), y_pred=np.array(pre_labels), target_names=target_names))

def run():

    #获取训练和验证数据
    print('读取数据')
    train_contents,train_labels = read_data(opt['train_filepath'])
    valid_contents,valid_labels = read_data(opt['valid_filepath'])

    print('加载词向量')
    word_vec = load_vector(opt['word2vec_filepath'])

    # train_contents = ['今天去看展览','今天加班','中午刚睡醒','明天休息']
    # train_labels = [0,1,0,0]
    #
    # valid_contents = ['今天去看展览', '今天加班', '中午刚睡醒', '明天休息']
    # valid_labels = [1, 0, 0, 0]

    #形成tensor、dataset、dataloader等数据形式
    print('创建tensor')
    train_tensors_data, train_tensors_label = create_tensors(train_contents, train_labels, word_vec)#找的网上公开词向量
    valid_tensors_data, valid_tensors_label = create_tensors(valid_contents, valid_labels, word_vec)

    print('创建dataset')
    train_dataset = create_dataset(train_tensors_data, train_tensors_label)
    valid_dataset = create_dataset(valid_tensors_data, valid_tensors_label)

    train_sampler = SequentialSampler(train_dataset)
    valid_sampler = SequentialSampler(valid_dataset)

    print('创建dataloader')
    train_dataloader = create_dataloader(dataset=train_dataset, sampler=train_sampler, batch_size=opt['batch_size'])
    valid_dataloader = create_dataloader(dataset=valid_dataset, sampler=valid_sampler, batch_size=opt['batch_size'])

    #创建模型实例
    print('创建模型实例')
    lstm_model = model(input_dim=opt['input_dim'], hidden_dim=opt['hidden_dim'], num_layer=opt['num_layer'], num_class=opt['num_class'])

    #训练模型（训练函数中调用验证函数）
    print('开始训练和验证')
    train(model = lstm_model,train_dataloader = train_dataloader,valid_dataloader=valid_dataloader,epoches=opt['epoches'])

if __name__ == '__main__':
    run()

