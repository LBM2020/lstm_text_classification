import torch
from torch.utils.data import TensorDataset, DataLoader

def read_data(filepath):
    with open(filepath,'r') as rf:
        lines = rf.readlines()
    contents = [line.split('\t')[0] for line in lines]
    labels = [int(line.split('\t')[1]) for line in lines]
    return contents,labels

def load_vector(filepath):
    #读取字向量，并形成字和向量之间的kv对应关系
    with open(filepath,'r') as rf:
        lines = rf.readlines()

    word_vec = {}
    for line in lines[1:]:#第一行是字段名
        line = line.split(' ')
        word = line[0]
        vec = list(map(float,line[1:-1]))#最后一个字符是换行符
        word_vec[word] = vec
    return word_vec

def create_dataset(tensor_data, tensor_label):
    dataset = TensorDataset(tensor_data, tensor_label)
    return dataset

def create_dataloader(dataset, sampler, batch_size):
    dataloader = DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size,collate_fn=collate_fn)
    return dataloader

def create_tensors(contents,labels,word_vec):
    #依据字和向量的对应关系获取评论文本中每个字的向量
    contents_vec = []
    for content in contents:
        content_vec = []
        content_vec.extend([word_vec[word] for word in content if word in word_vec.keys()])#获取每一句话的向量，content_vec维度为[seq_len,dim]
        contents_vec.append(torch.tensor(content_vec))#contents_vec维度为[batch,seq_len,dim]

    contents_vec = torch.nn.utils.rnn.pad_sequence(contents_vec, batch_first=True, padding_value=0)  # 对不等长的句子进行补齐
    tensor_data = torch.tensor(contents_vec, dtype=torch.float32)

    tensor_label = torch.tensor(labels, dtype=torch.int64)
    return tensor_data, tensor_label

def collate_fn(train_data):
    train_data.sort(key=lambda data: len(data), reverse=True)
    data_length = [len(data) for data in train_data]
    # train_data = torch.nn.utils.rnn.pad_sequence(train_data, batch_first=True, padding_value=0)#对不等长的句子进行补齐
    return train_data, data_length
