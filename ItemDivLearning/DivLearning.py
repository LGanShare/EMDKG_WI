import torch.nn as nn
import torch
from sklearn.metrics import ndcg_score
# from numpy.random import default_rng
from numpy.random import randint
import random
import pandas as pd
import numpy as np
import os
import copy
import math
from .metrics import MetronAtK
import json

class DivRepr(nn.Module):
    def __init__(self,engine):
        super(DivRepr,self).__init__()
        self.num_items = engine.itemTotal
        self.hidden_dim = engine.hidden_size

        self.item_embedding = nn.Embedding(self.num_items, self.hidden_dim)
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.item_embedding.weight)

    def forward(self,first_item, second_item):
        # for each entry of sorted_items_div_val:
        # there are first the ids of items and them the diversity value of the items
        # the diversity value are in descending order
        # if num of items is 2 in each entry
        first_item = first_item.type(torch.LongTensor).cuda()
        second_item = second_item.type(torch.LongTensor).cuda()
        first_item = first_item.squeeze()
        second_item = second_item.squeeze()
        # print('first item: {}'.format(first_item.size()))
        # print('seond item: {}'.format(second_item.size()))
        
        emb_div_val = self.div_eval(self.item_embedding(first_item),self.item_embedding(second_item))
        return emb_div_val

    def div_eval(self,first,second,t='cos'):
        if t=='cos':
            return nn.functional.cosine_similarity(first,second)
        elif t=='pairwise':
            return nn.functional.pairwise_distance(first,second)
       
class Engine(object):
    def __init__(self,num_items,hidden_dim,epoch,batch_size,learning_rate,use_cuda,optimizer_type,l2_lambda,item_div_dict):
        self.itemTotal = num_items
        self.hidden_size = hidden_dim
        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = learning_rate
        self.use_cuda = use_cuda
        self.item_div_dict = item_div_dict
        self.checkpoint_dir = '/home/lgan/divrecom/data/itemEmbedding/'
        self.result_dir = '/home/lgan/divrecom/data/itemEmbedding_res/'
        self.save_steps = 10
        self.valid_steps = 10
        self.early_stopping_patience = 10
        self.model = DivRepr(engine=self)
        self._metron = MetronAtK(top_k=10)
        if use_cuda == True:
            self.model.cuda()

        self.optimizer = self.use_optimizer(self.model,optimizer_type,learning_rate,l2_lambda)
       
    def save_checkpoint(self, model, epoch):
        path = os.path.join(
                self.checkpoint_dir, 'DivItem-anime-' + str(epoch) + '.cpkt')
        torch.save(model, path)

    def save_best_checkpoint(self, best_model):
        path = os.path.join(self.result_dir, 'DivItem-anime.ckpt')
        torch.save(best_model, path)

    def get_parameters(self, param_dict,mode='numpy'):
        for param in param_dict:
            param_dict[param] = param_dict[param].cpu()
        res = {}
        for param in param_dict:
            if mode == 'numpy':
                res[param] = param_dict[param].numpy()
            elif mode == 'list':
                res[param] = param_dict[param].numpy().tolist()
            else:
                res[param] = param_dict[param]
        return res
    
    def save_embedding_matrix(self, best_model):
        path = os.path.join(self.result_dir, 'item_div.json')
        with open(path, 'w') as fw:
            fw.write(json.dumps(self.get_parameters(best_model,'list')))

    def train_one_batch(self,batch):
        self.optimizer.zero_grad()
        first,second,neg = batch
        first = torch.tensor(first)
        second = torch.tensor(second)
        neg = torch.tensor(neg)
        if self.use_cuda == True:
            first, second, neg = first.cuda(),second.cuda(),neg.cuda()
        pos_emb_div_val = self.model(first,second)
        neg_emb_div_val = self.model(first,neg)
        loss = self.use_criteron(pos_emb_div_val,neg_emb_div_val)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return loss
    
    def train(self,train_data, valid_data=None):
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        best_epoch = 0
        best_hit = 0.0
        best_model = None
        bad_counts = 0
        for epoch in range(self.epoch):
            self.model.train()
            total_loss = 0
            total_batch = int(len(train_data)/self.batch_size)
            for i in range(total_batch):
                # print('Batch {}'.format(i))
                batch = self.get_random_block_from_data(train_data,self.batch_size)
                loss = self.train_one_batch(batch)
                total_loss += loss
            print('Epoch {} | loss: {}'.format(epoch,total_loss))
            
            if (epoch + 1) % self.save_steps == 0:
                print('Epoch {} has finished, saving...'.format(epoch))
                self.save_checkpoint(self.model.state_dict(),epoch)
            if (epoch + 1) % self.valid_steps == 0:
                print('Epoch {} has finished, saving...'.format(epoch))
                if valid_data:
                    hit, ndcg = self.evaluate(valid_data)
                    print('At {}th epoch, the hit and ndcg on valid data are: {}, {}'.format(epoch,hit,ndcg))
                    if hit > best_hit:
                        best_hit = hit
                        best_epoch = epoch
                        best_model = copy.deepcopy(self.model.state_dict())
                        print('Best model | hit of valid set is {}'.format(best_hit))
                        bad_counts = 0
                    else:
                        print('Hit of valid set is {} | bad count is {}'.format(hit, bad_counts))
                        bad_counts += 1
                    if bad_counts == self.early_stopping_patience:
                        print('Early stopping at epoch {}'.format(epoch))
                        break
        if best_model == None:
            best_model = self.model.state_dict()
            best_epoch = self.epochs - 1
            best_hit = self.evaluate(valid_data)
        print('Best epoch is {} | hit of valid set is {}'.format(best_epoch, best_hit))
        print('Store checkpoint of best result at epoch {}...'.format(best_epoch))
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)
        self.save_best_checkpoint(best_model)
        self.save_embedding_matrix(best_model)
        print('Finish storing')
        return best_model
    
    def get_random_block_from_data(self,data,batch_size):
        # print(data)
        first = []
        second, neg = [],[]
        # rng = default_rng()
        index_s = set()
        # print(type(first))
        while len(index_s) < batch_size:
            # index = rng.randint(0,len(data))
            index = randint(0,len(data))
            if index not in index_s:
                f, s, div_val = data[index]
                index_s.add(index)
                ## corrupt second item
                filtered_list = [e[0] for e in self.item_div_dict[f] if e[1]>= div_val ]
                # print('length of filtered list: {}'.format(len(filtered_list)))
                neg_candid_list = [e for e in range(self.itemTotal) if e not in filtered_list]
                n = random.sample(neg_candid_list,1)
                first.append(f)
                second.append(s)
                neg.append(n)
                ## corrupt first item
                # filtered_list = [e[0] for e in self.item_div_dict[s] if e[1] >= div_val]
                # neg_candid_list = [e for e in range(self.num_items) if e not in filtered_list]
                # n = random.sample(neg_candid_list,1)
                # first.append(f)
                # second.append(s)
                # neg.append(n)
        # res.sort(key=lambda x:x[2],reverse=True)
        return first, second, neg

    def use_optimizer(self,network,optimizer_type,learning_rate,l2_lambda):
        if optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(network.parameters(),lr=learning_rate,momentum=0.95,weight_decay=l2_lambda)
        elif optimizer_type == 'adam':
            optimizer = torch.optim.Adam(network.parameters(),lr=learning_rate,weight_decay=l2_lambda)
        return optimizer

    def use_criteron(self,pos_emb_div_val,neg_emb_div_val):
        # length = len(div_val)
        loss = -torch.log(torch.sigmoid( pos_emb_div_val - neg_emb_div_val))
        loss = loss.sum()
        return loss

    def evaluate(self,data,n=10):
        # diversity value as ranking
        # the validation data and test data give the item list and its corresponding feature diversity value in sorted order
        # calculate the ranking difference between embedding results and the given ones
        self.model.eval()
        data = pd.DataFrame(data,columns=['first','second','div_val'])
        first = list(data['first'])
        second = list(data['second'])
        div_val = list(data['div_val'])
        neg_first, neg_second = self.sample_negative(first, second,div_val, self.item_div_dict)
        if self.use_cuda:
            first = torch.LongTensor(first).cuda()
            second = torch.LongTensor(second).cuda()
            neg_first = torch.LongTensor(neg_first).cuda()
            neg_second = torch.LongTensor(neg_second).cuda()
        test_scores = self.model(first, second)
        negative_scores = self.model(neg_first, neg_second)
        if self.use_cuda:
            first = first.cpu()
            second = second.cpu()
            test_scores = test_scores.cpu()
            neg_first = neg_first.cpu()
            neg_second = second.cpu()
            negative_scores = negative_scores.cpu()
        print('first length:{}'.format(len(first.data.view(-1).tolist())))
        print('second length:{}'.format(len(second.data.view(-1).tolist())))
        print('test_scores length: {}'.format(len(test_scores.data.view(-1).tolist())))
        print('neg_first length:{}'.format(len(neg_first.data.view(-1).tolist())))
        print('neg_second length:{}'.format(len(neg_second.data.view(-1).tolist())))
        print('neg_scores length: {}'.format(len(negative_scores.data.view(-1).tolist())))
        
        hit, ndcg = 0, 0
        count = 0

        for f,s, test_val in zip(first.data.view(-1).tolist(),
               second.data.view(-1).tolist(),
               test_scores.data.view(-1).tolist()):
            neg_val = negative_scores.data.view(-1).tolist()[count*99:count*99 + 99]
            sorted(neg_val, reverse=True)
            if test_val >= neg_val[9]:
                hit += 1
                index = 9
                while test_val >= neg_val[index] and index > -1:
                    index -= 1
                # print('index: {}'.format(index))
                ndcg += 1./math.log2(index + 3)
            count += 1
        hit /= count
        ndcg /= count
        # hit, ndcg = self._metron.cal_hit_ratio(),self._metron.call_ndcg()
        print('[Evaluating current model] Hit= {:.4f}, NDCG= {:.4f}\n'.format(hit, ndcg))
        # data = data.sort_values(by=['div_val'],ascending=False)
        # data = torch.tensor(data.to_numpy()).cuda()
        # first_items = torch.LongTensor(data[:,:1]).cuda()
        # second_items = torch.LongTensor(data[:,1:2]).cuda()
        # first_items = first_items.squeeze()
        # second_items = second_items.squeeze()
        
        # div_val = data[:,-1]
        # div_val = div_val.squeeze()
        # div_val = div_val.detach().cpu().numpy()
        # self.model.eval()
        # f,s,dv, emb_div_val = self.model(data)
        # emb_div_val = emb_div_val.detach().cpu().numpy()
        # emb_sorted_index = np.argsort(emb_div_val)
        

        # Hit
        # count = 0
        # for i in range(n):
        #     if emb_sorted_index[i] < n:
        #         count += 1
        # hit = count/n

        # NDCG
        # ndcg = ndcg_score(div_val,emb_div_val,k=n)
        # ndcg = 0.0

        return hit,ndcg

    def sample_negative(self, first, second, div_val, item_div_dict):
        neg_first = []
        neg_second = []
        count_first = set()
        for f, s, val in zip(first, second, div_val):
            # if f in count_first:
            #     continue
            # count_first.add(f)
            #################### 0.25 for movielens datasets ###########
            #################### 0.18 for anime datasets ###############
            ## self.delta
            pos_f_list = [e[0] for e in item_div_dict[f] if e[1] > 0.18]
            candid_list = [i for i in range(self.itemTotal) if i not in pos_f_list]
            samples = random.sample(candid_list, 99)
            for s in samples:
                neg_first.append(f)
                neg_second.append(s)
        print('length neg_first, neg_second: {}, {}'.format(len(neg_first), len(neg_second)))
        return neg_first, neg_second
