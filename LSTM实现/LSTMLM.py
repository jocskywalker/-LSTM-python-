import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import give_valid_test
import _pickle as cpickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")

def make_batch(train_path, word2number_dict, batch_size, n_step):
    all_input_batch = []
    all_target_batch = []

    text = open(train_path, 'r', encoding='utf-8') #open the file

    input_batch = []
    target_batch = []
    for sen in text:
        word = sen.strip().split(" ")  # space tokenizer

        word = ["<sos>"] + word
        word = word + ["<eos>"]


        if len(word) <= n_step:   #pad the sentence
            word = ["<pad>"]*(n_step+1-len(word)) + word

        for word_index in range(len(word)-n_step):
            input = [word2number_dict[n] for n in word[word_index:word_index+n_step]]  # create (1~n-1) as input
            target = word2number_dict[word[word_index+n_step]]  # create (n) as target, We usually call this 'casual language model'
            input_batch.append(input)
            target_batch.append(target)

            if len(input_batch) == batch_size:
                all_input_batch.append(input_batch)
                all_target_batch.append(target_batch)
                input_batch = []
                target_batch = []

    return all_input_batch, all_target_batch # (batch num, batch size, n_step) (batch num, batch size)


def make_dict(train_path):
    text = open(train_path, 'r', encoding='utf-8')  #open the train file
    word_list = set()  # a set for making dict

    for line in text:
        line = line.strip().split(" ")
        word_list = word_list.union(set(line))

    word_list = list(sorted(word_list))   #set to list

    word2number_dict = {w: i+2 for i, w in enumerate(word_list)}
    number2word_dict = {i+2: w for i, w in enumerate(word_list)}

    #add the <pad> and <unk_word>
    word2number_dict["<pad>"] = 0
    number2word_dict[0] = "<pad>"
    word2number_dict["<unk_word>"] = 1
    number2word_dict[1] = "<unk_word>"
    word2number_dict["<sos>"] = 2
    number2word_dict[2] = "<sos>"
    word2number_dict["<eos>"] = 3
    number2word_dict[3] = "<eos>"



    return word2number_dict, number2word_dict


class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()
        self.E = nn.Embedding(n_class, embedding_dim=emb_size)
        # n_class为字典大小，输出一个tensor类型，尺寸为sizeofinput * embedding_dim
        # parameter可训练参数
        # 遗忘门参数
        self.Wxf = nn.Linear(emb_size, n_hidden, bias=False)
        self.Whf = nn.Linear(n_hidden, n_hidden, bias=False)
        self.bf = nn.Parameter(torch.ones([n_hidden]))

        # 输入门参数
        self.Wxi = nn.Linear(emb_size, n_hidden, bias=False)
        self.Whi = nn.Linear(n_hidden, n_hidden, bias=False)
        self.bi = nn.Parameter(torch.ones([n_hidden]))

        self.Wxc = nn.Linear(emb_size, n_hidden, bias=False)
        self.Whc = nn.Linear(n_hidden, n_hidden, bias=False)
        self.bc = nn.Parameter(torch.ones([n_hidden]))

        # 输出门参数
        self.Wxo = nn.Linear(emb_size, n_hidden, bias=False)
        self.Who = nn.Linear(n_hidden, n_hidden, bias=False)
        self.bo = nn.Parameter(torch.ones([n_hidden]))

        self.Whq = nn.Linear(n_hidden, n_class, bias=False)
        self.bq = nn.Parameter(torch.ones([n_class]))

    def forward(self, X, H, C):
        X = self.E(X)
        # print(X.shape) 128 5 256
        X = X.transpose(0, 1)
        # 交换0轴和1轴
        # print(X.shape) n_step:5 batch_size:128 embedding:256
        for xt in X:
            # print(x_t.shape) 128 256
            # 遗忘门
            Ft = torch.sigmoid(self.Wxf(xt) + self.Whf(H) + self.bf)  # ft
            # 输入门
            It = torch.sigmoid(self.Wxi(xt) + self.Whi(H) + self.bi)  # it
            C_tilda = torch.tanh(self.Wxc(xt) + self.Whc(H) + self.bc)  # gt
            # 输出门
            Ot = torch.sigmoid(self.Wxo(xt) + self.Who(H) + self.bo)  # Ot
            H = Ot * torch.tanh(C)
            # 记忆更新
            C = Ft * C + It * C_tilda

        model = self.Whq(H) + self.bq
        return model


# 调用LSTM
# class TextLSTM(nn.Module):
#     def __init__(self):
#         super(TextLSTM, self).__init__()
#         self.C = nn.Embedding(n_class, embedding_dim=emb_size)
#         self.LSTM = nn.LSTM(input_size=emb_size, hidden_size=n_hidden)
#         self.W = nn.Linear(n_hidden, n_class, bias=False)
#         self.b = nn.Parameter(torch.ones([n_class]))
#
#     def forward(self, X):
#         X = self.C(X)
#
#         # hidden_state = torch.zeros(1, len(X), n_hidden)  # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
#         # cell_state = torch.zeros(1, len(X), n_hidden)     # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
#
#         X = X.transpose(0, 1) # X : [n_step, batch_size, embeding size]
#
#         # outputs, (_, _) = self.LSTM(X, (hidden_state, cell_state))
#         outputs, (_, _) = self.LSTM(X)
#         # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
#         # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
#         outputs = outputs[-1] # [batch_size, num_directions(=1) * n_hidden]
#         model = self.W(outputs) + self.b # model : [batch_size, n_class]
#         return model

class DoubleLayerTextLSTM(nn.Module):
    def __init__(self):
        super(DoubleLayerTextLSTM, self).__init__()
        self.E = nn.Embedding(n_class, embedding_dim=emb_size)
        # n_class为字典大小，输出一个tensor类型，尺寸为sizeofinput * embedding_dim
        # parameter可训练参数
        # 遗忘门参数
        self.Wxf = nn.Linear(emb_size, n_hidden, bias=False)
        self.Whf = nn.Linear(n_hidden, n_hidden, bias=False)
        self.bf = nn.Parameter(torch.ones([n_hidden]))

        # 输入门参数
        self.Wxi = nn.Linear(emb_size, n_hidden, bias=False)
        self.Whi = nn.Linear(n_hidden, n_hidden, bias=False)
        self.bi = nn.Parameter(torch.ones([n_hidden]))

        self.Wxc = nn.Linear(emb_size, n_hidden, bias=False)
        self.Whc = nn.Linear(n_hidden, n_hidden, bias=False)
        self.bc = nn.Parameter(torch.ones([n_hidden]))

        # 输出门参数
        self.Wxo = nn.Linear(emb_size, n_hidden, bias=False)
        self.Who = nn.Linear(n_hidden, n_hidden, bias=False)
        self.bo = nn.Parameter(torch.ones([n_hidden]))

        self.Whq = nn.Linear(n_hidden, n_class, bias=False)
        self.bq = nn.Parameter(torch.ones([n_class]))

        #第二层参数
        # 遗忘门参数
        self.Wxfl2 = nn.Linear(emb_size, n_hidden, bias=False)
        self.Whfl2 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.bfl2 = nn.Parameter(torch.ones([n_hidden]))

        # 输入门参数
        self.Wxil2 = nn.Linear(emb_size, n_hidden, bias=False)
        self.Whil2 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.bil2 = nn.Parameter(torch.ones([n_hidden]))

        self.Wxcl2 = nn.Linear(emb_size, n_hidden, bias=False)
        self.Whcl2 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.bcl2 = nn.Parameter(torch.ones([n_hidden]))

        # 输出门参数
        self.Wxol2 = nn.Linear(emb_size, n_hidden, bias=False)
        self.Whol2 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.bol2 = nn.Parameter(torch.ones([n_hidden]))

        #Hl2更新参数
        self.Wxn = nn.Linear(n_hidden, n_hidden, bias=False)
        self.Wnn = nn.Linear(n_hidden, n_hidden, bias=False)
        self.bn = nn.Parameter(torch.ones([n_hidden]))

    def forward(self, X, H, C):
        # X是128 5的tensor,E(X)后变成 128 5 256
        X = self.E(X)
        # print(X.shape) 128 5 256
        X = X.transpose(0, 1)
        # 交换0轴和1轴
        # print(X.shape) n_step:5 batch_size:128 embedding:256

        Hl2 = H
        Cl2 = C

        for xt in X:
            # print(x_t.shape) 128 256
            #遗忘门
            Ft = torch.sigmoid(self.Wxf(xt) + self.Whf(H) + self.bf)  # ft
            #输入门
            It = torch.sigmoid(self.Wxi(xt) + self.Whi(H) + self.bi)     #it
            C_tilda = torch.tanh(self.Wxc(xt) + self.Whc(H) + self.bc)   #gt
            #输出门
            Ot = torch.sigmoid(self.Wxo(xt) + self.Who(H) + self.bo)  # Ot
            H = Ot * torch.tanh(C)
            #记忆更新
            C = Ft * C + It * C_tilda
            #第二层layer2
            #更新Hl2
            HL2 = torch.sigmoid(self.Wxn(H) + self.Wnn(Hl2) + self.bn)
            # 遗忘门
            Ftl2 = torch.sigmoid(self.Wxfl2(xt) + self.Whfl2(H) + self.bfl2)  # ft
            # 输入门
            Itl2 = torch.sigmoid(self.Wxil2(xt) + self.Whil2(H) + self.bil2)  # it
            C_tildal2 = torch.tanh(self.Wxcl2(xt) + self.Whcl2(H) + self.bcl2)  # gt
            # 输出门
            Otl2 = torch.sigmoid(self.Wxol2(xt) + self.Whol2(Hl2) + self.bol2)  # Ot
            Hl2 = Otl2 * torch.tanh(Cl2)
            # 记忆更新
            Cl2 = Ftl2 * Cl2 + Itl2 * C_tildal2

        model = self.Whq(Hl2) + self.bq
        return model

def train_LSTMlm():
    model = DoubleLayerTextLSTM() #双层
    # model = TextLSTM() #单层
    model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    
    # Training
    batch_number = len(all_input_batch)
    # print(batch_number) 603
    # print(all_input_batch.shape) 603 128 5  thenumofbatch batchsize n_step
    # batch数量
    for epoch in range(all_epoch):
        count_batch = 0
        for input_batch, target_batch in zip(all_input_batch, all_target_batch):
            optimizer.zero_grad()

            # input_batch : [batch_size, n_step, n_class]
            H = torch.zeros(batch_size, n_hidden, device=device)
            C = torch.zeros(batch_size, n_hidden, device=device)
            # print(input_batch.shape) 128 5
            # print(target_batch.shape) 128
            output = model(input_batch, H, C)

            # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
            loss = criterion(output, target_batch)
            ppl = math.exp(loss.item())
            if (count_batch + 1) % 100 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
                      'loss =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))

            loss.backward()
            optimizer.step()

            count_batch += 1
        print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
                'loss =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))

        # valid after training one epoch
        all_valid_batch, all_valid_target = give_valid_test.give_valid(data_root, word2number_dict, n_step)
        all_valid_batch = torch.LongTensor(all_valid_batch).to(device)  # list to tensor
        all_valid_target = torch.LongTensor(all_valid_target).to(device)

        total_valid = len(all_valid_target)*128  # valid and test batch size is 128
        with torch.no_grad():
            total_loss = 0
            count_loss = 0
            for valid_batch, valid_target in zip(all_valid_batch, all_valid_target):
                valid_output = model(valid_batch, H, C)
                valid_loss = criterion(valid_output, valid_target)
                total_loss += valid_loss.item()
                count_loss += 1
          
            print(f'Valid {total_valid} samples after epoch:', '%04d' % (epoch + 1), 'loss =',
                  '{:.6f}'.format(total_loss / count_loss),
                  'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))

        if (epoch+1) % save_checkpoint_epoch == 0:
            torch.save(model, f'models/LSTMlm_model_epoch{epoch+1}.ckpt')

def test_LSTMlm(select_model_path):
    model = torch.load(select_model_path, map_location="cpu")  #load the selected model
    model.to(device)

    #load the test data
    all_test_batch, all_test_target = give_valid_test.give_test(data_root, word2number_dict, n_step)
    all_test_batch = torch.LongTensor(all_test_batch).to(device)  # list to tensor
    all_test_target = torch.LongTensor(all_test_target).to(device)
    total_test = len(all_test_target)*128  # valid and test batch size is 128
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    count_loss = 0
    H = torch.zeros(batch_size, n_hidden, device=device)
    C = torch.zeros(batch_size, n_hidden, device=device)
    for test_batch, test_target in zip(all_test_batch, all_test_target):
        test_output = model(test_batch, H, C)
        test_loss = criterion(test_output, test_target)
        total_loss += test_loss.item()
        count_loss += 1

    print(f"Test {total_test} samples with {select_model_path}……………………")
    print('loss =','{:.6f}'.format(total_loss / count_loss),
                  'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))

if __name__ == '__main__':
    n_step = 5 # number of cells(= number of Step)
    n_hidden = 128 # number of hidden units in one cell
    batch_size = 128 # batch size
    learn_rate = 0.0005
    all_epoch = 5 #the all epoch for training
    emb_size = 256 #embeding size
    save_checkpoint_epoch = 5 # save a checkpoint per save_checkpoint_epoch epochs !!! Note the save path !!!
    data_root = 'penn_small'
    train_path = os.path.join(data_root, 'train.txt') # the path of train dataset

    print("print parameter ......")
    print("n_step:", n_step)
    print("n_hidden:", n_hidden)
    print("batch_size:", batch_size)
    print("learn_rate:", learn_rate)
    print("all_epoch:", all_epoch)
    print("emb_size:", emb_size)
    print("save_checkpoint_epoch:", save_checkpoint_epoch)
    print("train_data:", data_root)

    word2number_dict, number2word_dict = make_dict(train_path)
    #print(word2number_dict)

    print("The size of the dictionary is:", len(word2number_dict))
    n_class = len(word2number_dict)  #n_class (= dict size)

    print("generating train_batch ......")
    all_input_batch, all_target_batch = make_batch(train_path, word2number_dict, batch_size, n_step)  # make the batch
    train_batch_list = [all_input_batch, all_target_batch]
    
    print("The number of the train batch is:", len(all_input_batch))
    all_input_batch = torch.LongTensor(all_input_batch).to(device)   #list to tensor
    all_target_batch = torch.LongTensor(all_target_batch).to(device)
    # print(all_input_batch.shape)
    # print(all_target_batch.shape)
    all_input_batch = all_input_batch.reshape(-1, batch_size, n_step)
    all_target_batch = all_target_batch.reshape(-1, batch_size)

    # print("\nTrain the LSTMLM……………………")
    print("\nTrain the DoublelayerLSTMLM……………………")
    train_LSTMlm()

    print("\nTest the DoublelayerLSTMLM……………………")
    # print("\nTest the LSTMLM……………………")
    select_model_path = "models/LSTMlm_model_epoch5.ckpt"
    test_LSTMlm(select_model_path)
