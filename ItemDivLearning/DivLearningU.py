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

class DivReprU(nn.Module):
    def __init__(self,engine):
        super(DivReprU,self).__init__()
        self.num_items = engine.itemTotal
        self.hidden_dim = engine.hidden_size

        self.item_embedding = nn.Parameter(torch.empty(self.num_items, self.hidden_dim))
        masked_item_embedding = torch.zeros(1,self.hidden_dim, requires_grad=False)
        full_item_embedding = torch.empty(self.num_items+1, self.hidden_dim)
        full_item_embedding[:self.num_items] = self.item_embedding
        full_item_embedding[self.num_items:] = masked_item_embedding
        # full_item_embedding = torch.cat((self.item_embedding.weight, masked_item_embedding), 0)
        self.register_buffer('full_item_embedding', full_item_embedding)
        # self.normalize_item_emb = nn.BatchNorm1d(self.hidden_dim)
        self.init_weight()

    def init_weight(self):
        # p1 = torch.empty(self.num_items,self.hidden_dim)
        # nn.init.normal_(p1,0,0.05)
        nn.init.normal_(self.item_embedding,1,0.05)


    def forward(self,items):
        # for each entry of sorted_items_div_val:
        # there are first the ids of items and them the diversity value of the items
        # the diversity value are in descending order
        # if num of items is 2 in each entry
        items = items.type(torch.LongTensor).cuda()
        # print(items)
        # print(self.num_items)
        self.full_item_embedding.detach_()
        # self.item_embedding.weight = self.normalize_item_emb(self.item_embedding)
        self.full_item_embedding[:-1] = self.item_embedding
        emb_div_val = self.div_eval(nn.functional.embedding(items,self.full_item_embedding))
        # print(emb_div_val)
        return emb_div_val

    def div_eval(self,item_emb,t='det',delta=0.001):
        if t=='det':
            item_emb_T = torch.transpose(item_emb,1,-1)
            # print(item_emb_T.size())
            m = torch.matmul(item_emb,item_emb_T) 
            # item_emb_ = item_emb.unsqueeze(-1).expand(-1,-1,-1,)
            # print(m)
            return torch.logdet(m) # + delta*torch.linalg.norm(item_emb) 
        elif t=='cos':
            pass
        elif t=='pairwise':
            # pair_item = torch.combinations(torch.torch.size(item_emb)[0])
            
            nn.functional.pdist(item_emb)
       
class EngineU(object):
    def __init__(self,num_items,hidden_dim,epoch,batch_size,learning_rate,use_cuda,optimizer_type,l2_lambda,fi):
        self.itemTotal = num_items
        self.hidden_size = hidden_dim
        self.epoch = epoch
        self.batch_size = batch_size
        self.sample_size = 19
        self.lr = learning_rate
        self.use_cuda = use_cuda
        # self.item_div_set = item_div_set
        self.checkpoint_dir = '/home/lgan/divrecom/data/itemEmbedding/'
        self.result_dir = '/home/lgan/divrecom/data/itemEmbedding_res/'
        self.save_steps = 2
        self.valid_steps = 2
        self.dataset_name = 'anime'
        self.early_stopping_patience = 5
        self.model = DivReprU(engine=self)
        self._metron = MetronAtK(top_k=10)
        self.fi = fi
        self.fm = self.fi.gen_feature_map(list(range(len(self.fi.items))))
        if use_cuda == True:
            self.model.cuda()

        self.optimizer = self.use_optimizer(self.model,optimizer_type,learning_rate,l2_lambda)
    
    def set_dataset_name(self, dataset_name):
        self.dataset_name = dataset_name

    def set_delta(self, delta):
        self.delta = delta

    def set_item_div_set(self, item_div_set):
        self.item_div_set = item_div_set

    def set_item_mapping(self, i_mapping):
        # keys are item ids in knowledge graph entities
        # values of item ids in item diversity learning item ids
        self.i_mapping = i_mapping

    def set_pos_items(self, pos_items):
        self.pos_items = pos_items

    def save_checkpoint(self, model, epoch):
        path = os.path.join(
                self.checkpoint_dir, 'DivItem-{}-'.format(self.dataset_name) + str(epoch) + '.cpkt')
        torch.save(model, path)

    def save_best_checkpoint(self, best_model):
        path = os.path.join(self.result_dir, 'DivItem-{}.ckpt'.format(self.dataset_name))
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
        itemlist,corrupt_itemlist = batch
        itemlist = torch.tensor(itemlist)
        # print('itemlist: {}'.format(itemlist))
        corrupt_itemlist = torch.tensor(corrupt_itemlist)
        # print('corrupted list: {}'.format(corrupt_itemlist))
        if self.use_cuda == True:
            itemlist, corrupt_itemlist = itemlist.cuda(),corrupt_itemlist.cuda()
        pos_emb_div_val = self.model(itemlist)
        neg_emb_div_val = self.model(corrupt_itemlist)
        # print(pos_emb_div_val)
        loss = self.use_criteron(pos_emb_div_val,neg_emb_div_val)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return loss
    
    def train(self,train_data, valid_data=None):
        ## train_data, valid_data are dictionary with keys as user ids
        print('Enter training')
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        best_epoch = 0
        best_hit = 0.0
        best_model = None
        bad_counts = 0
        
        itemlists_size = 0
        for u in train_data:
            itemlists_size += len(train_data[u])

        total_batch = int(itemlists_size/self.batch_size)
        print('Total batch number: {}'.format(total_batch))
        for epoch in range(self.epoch):
            torch.cuda.empty_cache()
            self.model.train()
            total_loss = 0
            
            ## modifier the total itemlists size 
            # itemlists_size = 0
            # for u in train_data:
            #     itemlists_size += len(train_data[u])

            # total_batch = int(itemlists_size/self.batch_size)
            
            for i in range(total_batch):
                # if (i+1) %10 == 0:
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
        # data is a dictionary with keys as user ids
        # data_l is a list of all values from data
        # print('enter get random block')
        # data_l = sum(data.values(),[])
        data_interval = [len(data[k]) for k in data]
        data_interval = list(np.cumsum(data_interval))
        # print(len(data_interval))
        itemlists = []
        corrupt_itemlists = []
        index_s = set()
        maxlen = 0
        while len(index_s) < batch_size:
            # print(len(index_s))
            # index = rng.randint(0,len(data))
            index = randint(0, data_interval[-1])
            # find u based on index
            u = 0
            users = list(data.keys())
            while u < len(data):
                if u == 0 and index < data_interval[u]:
                    break
                elif index >= data_interval[u]:
                    u += 1
                    continue
                elif index < data_interval[u]:
                    break
            uu = users[u]

            if index not in index_s:
                try:
                    if u > 0:
                        itemlist, div_val = data[uu][index - data_interval[u-1]]
                    else:
                        itemlist, div_val = data[uu][index]
                except IndexError as e:
                    print('u:{}\n{}\n'.format(u,data_interval[u-1]))
                    # print(index)
                if len(itemlist) != 10:
                    continue
                index_s.add(index)
                ## corrupt itemlist
                # corrupt the k-th item (last item)
                filtered_list = [e for e in itemlist if not e == itemlist[-1]]
                pos_items_u = self.pos_items[uu]
                neg_candid_list = [e for e in range(self.itemTotal) if e not in (filtered_list or pos_items_u)]
                n = random.sample(neg_candid_list,1)
                candid_corrupt_list = itemlist[:-1] + n
                cnt = 0
                while (set(candid_corrupt_list) in self.item_div_set[uu]) and cnt < 3000:
                    n = random.sample(neg_candid_list,1)
                    candid_corrupt_list = itemlist[:-1] + n
                    cnt += 1
                itemlists.append(itemlist)
                corrupt_itemlists.append(candid_corrupt_list)

                if len(itemlist) > maxlen:
                     maxlen = len(itemlist)
        for i in range(len(itemlists)):
            if len(itemlists[i]) < maxlen:
                print(len(itemlists[i]))
                itemlists[i].extend(np.array([self.itemTotal]).repeat(maxlen - len(itemlists[i])))
            if len(corrupt_itemlists[i]) < maxlen:
                corrupt_itemlists[i].extend(np.array([self.itemTotal]).repeat(maxlen - len(itemlists[i])))
        return itemlists, corrupt_itemlists

    def use_optimizer(self,network,optimizer_type,learning_rate,l2_lambda):
        if optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(network.parameters(),lr=learning_rate,momentum=0.95,weight_decay=l2_lambda)
        elif optimizer_type == 'adam':
            optimizer = torch.optim.Adam(network.parameters(),lr=learning_rate,weight_decay=l2_lambda)
        return optimizer

    def use_criteron(self,pos_emb_div_val,neg_emb_div_val):
        loss = -torch.log(torch.sigmoid( pos_emb_div_val - neg_emb_div_val))
        loss = loss.sum()
        return loss

    def evaluate(self,data,n=10):
        torch.cuda.empty_cache()
        # diversity value as ranking
        # the validation data and test data give the item list and its corresponding feature diversity value in sorted order
        # calculate the ranking difference between embedding results and the given ones
        self.model.eval()
        # data = pd.DataFrame(data,columns=['first','second','div_val'])
        data_l = sum(data.values(), [])

        items = [e[0] for e in data_l]
        div_val = [e[1] for e in data_l]
        user_itemlist = []
        for u in data:
            user_itemlist += [u for i in range(len(data[u]))]

        items, neg_items = self.sample_negative(items,div_val, self.item_div_set,user_itemlist)
        if self.use_cuda:
            items = torch.LongTensor(items).cuda()
            neg_items = torch.LongTensor(neg_items).cuda()
        test_scores = self.model(items)
        negative_scores = self.model(neg_items)
        if self.use_cuda:
            items = items.cpu()
            test_scores = test_scores.cpu()
            neg_items = neg_items.cpu()
            negative_scores = negative_scores.cpu()
        print('test_scores length: {}'.format(len(test_scores.data.view(-1).tolist())))
        print('neg_scores length: {}'.format(len(negative_scores.data.view(-1).tolist())))
        
        hit, ndcg = 0, 0
        count = 0

        for i, test_val in enumerate(test_scores.data.view(-1).tolist()):
            neg_val = negative_scores.data.view(-1).tolist()[count*self.sample_size:count*self.sample_size + self.sample_size]
            if len(neg_val) == 0:
                break
            sorted(neg_val, reverse=True)
            if test_val >= neg_val[9]:
                hit += 1
                index = 9
                while test_val >= neg_val[index] and index > -1:
                    index -= 1
                ndcg += 1./math.log2(index + 3)
            count += 1
        hit /= count-1
        ndcg /= count-1
        print('[Evaluating current model] Hit= {:.4f}, NDCG= {:.4f}\n'.format(hit, ndcg))
        return hit,ndcg

    def sample_negative(self, itemlists, div_val, item_div_set, user_itemlist=[]):
        neg_items = []
        count_first = set()
        new_itemlists = []
        for idx, itemlist, val in zip(range(len(itemlists)),itemlists, div_val):
            index = 0
            neg_items_u = []
            new_itemlist_u = [] 
            u = user_itemlist[idx]
            while len(neg_items_u) < self.sample_size:
                filtered_list = [e for e in itemlist if not e == itemlist[index]]
                pos_items_u = self.pos_items[u]
                candid_list = [i for i in range(self.itemTotal) if i not in (filtered_list or pos_items_u)]
                sample_item = random.sample(candid_list, 1)[0]
                candid_neg_itemlist = list(np.insert(filtered_list,index,sample_item))
                while set(candid_neg_itemlist) in item_div_set[u]:
                    sample_item = random.sample(candid_list,1)[0]
                    candid_neg_itemlist = list(np.insert(filtered_list,index,sample_item))
                # print('candid_neg_itemlist:{}'.format(candid_neg_itemlist))
                neg_items_u.append(candid_neg_itemlist)
                new_itemlist_u.append(itemlist)
                if index < len(itemlist)-1:
                    index += 1
                else:
                    index = 0
            if len(neg_items) == 0:
                neg_items = neg_items_u
            else:
                neg_items = np.concatenate((neg_items,neg_items_u),0)
            if len(new_itemlists) == 0:
                new_itemlists = new_itemlist_u
            else:
                new_itemlists = np.concatenate((new_itemlists,new_itemlist_u),0)
            # print('Finish sampling itemlist {}'.format(itemlist))
        ## 
        # del neg_items_u,neg_itemlist_u    
        print('length neg_items: {}, {}'.format(len(new_itemlists), len(neg_items)))
        return new_itemlists,neg_items
