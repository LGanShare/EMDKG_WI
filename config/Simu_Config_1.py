import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import ctypes
import os
import json
import numpy as np
from numpy.random import randint
import random
import copy
import pandas as pd
import math
import sys

def to_var(x, use_gpu):
    if use_gpu:
        return Variable(torch.from_numpy(x).cuda())
    else:
        return Variable(torch.from_numpy(x))

class SimuConfig1(object):
    def __init__(self):
        base_file = '/home/lgan/divrecom/release/Base.so'
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        
        """argtypes"""
        # sampling 
        self.lib.sampling.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ]
        # valid
        self.lib.getValidHeadBatch.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ]

        self.lib.getValidTailBatch.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ]

        self.lib.validHead.argtypes = [ctypes.c_void_p]
        self.lib.validTail.argtypes = [ctypes.c_void_p]

        self.lib.getHeadBatch.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ]
        self.lib.getTailBatch.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ]
        self.lib.testHead.argtypes = [ctypes.c_void_p]
        self.lib.testTail.argtypes = [ctypes.c_void_p]
        self.lib.getValidBatch.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ]
        self.lib.getTestBatch.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ]
        self.lib.getBestThreshold.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ]
        self.lib.test_triple_classification.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ]
        self.lib.getValidHit10.restype = ctypes.c_float

        """set parameters"""
        self.in_path = "./"
        self.batch_size = 100
        self.hidden_size = 75
        self.bern = 0
        self.work_threads = 8
        self.valid_steps = 10
        self.save_steps = 10
        self.opt_method = 'adam'
        self.divItemOptimizer = None
        self.recomOptimizer = None
        self.lr_decay = 0
        self.weight_decay = 0
        self.lmbda = 0.0
        self.alpha = 0.001
        self.early_stopping_patience = 10
        self.nbatches = 100
        self.p_norm = 2
        self.DivItemModel = None
        self.RecomModel = None
        self.trainDivItemModel = None
        self.trainRecomModel = None
        self.testDivItemModel = None
        self.testRecomModel = None
        self.pretrain_model = None
        self.use_gpu = True
        self.delta = 0.18
        pass
    
    def init(self):
        # set inPath
        self.lib.setInPath(
                ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2))
        # set outPath
        self.lib.setOutPath(
                ctypes.create_string_buffer(self.result_dir.encode(), len(self.result_dir) * 2))
        self.lib.setBern(self.bern)
        self.lib.setWorkThreads(self.work_threads)
        self.lib.randReset()

        self.lib.importTrainFiles()
        self.lib.importTestFiles()
        self.lib.importTypeFiles()
        self.relTotal = self.lib.getRelationTotal()
        self.entTotal = self.lib.getEntityTotal()
        self.trainTotal = self.lib.getTrainTotal()
        self.testTotal = self.lib.getTestTotal()
        self.validTotal = self.lib.getValidTotal()
        self.userTotal = self.lib.getUserTotal()
        self.itemTotal = self.lib.getItemTotal()

        self.batch_size = int(self.trainTotal/self.nbatches)
        self.batch_seq_size = self.batch_size * (1 + self.negative_ent + self.negative_rel)
        
        self.batch_h = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_t = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_r = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_y = np.zeros(self.batch_seq_size, dtype=np.float32)

        self.batch_h_addr = self.batch_h.__array_interface__['data'][0]
        self.batch_t_addr = self.batch_t.__array_interface__['data'][0]
        self.batch_r_addr = self.batch_r.__array_interface__['data'][0]
        self.batch_y_addr = self.batch_y.__array_interface__['data'][0]

        self.valid_h = np.zeros(self.entTotal, dtype=np.int64)
        self.valid_t = np.zeros(self.entTotal, dtype=np.int64)
        self.valid_r = np.zeros(self.entTotal, dtype=np.int64)

        self.valid_h_addr = self.valid_h.__array_interface__['data'][0]
        self.valid_t_addr = self.valid_t.__array_interface__['data'][0]
        self.valid_r_addr = self.valid_r.__array_interface__['data'][0]

        self.test_h = np.zeros(self.entTotal, dtype=np.int64)
        self.test_t = np.zeros(self.entTotal, dtype=np.int64)
        self.test_r = np.zeros(self.entTotal, dtype=np.int64)
        self.test_h_addr = self.test_h.__array_interface__['data'][0]
        self.test_t_addr = self.test_t.__array_interface__['data'][0]
        self.test_r_addr = self.test_r.__array_interface__['data'][0]

        self.valid_pos_h = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_pos_t = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_pos_r = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_pos_h_addr = self.valid_pos_h.__array_interface__['data'][0]
        self.valid_pos_t_addr = self.valid_pos_t.__array_interface__['data'][0]
        self.valid_pos_r_addr = self.valid_pos_r.__array_interface__['data'][0]
        
        self.valid_neg_h = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_neg_t = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_neg_r = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_neg_h_addr = self.valid_neg_h.__array_interface__['data'][0]
        self.valid_neg_t_addr = self.valid_neg_t.__array_interface__['data'][0]
        self.valid_neg_r_addr = self.valid_neg_r.__array_interface__['data'][0]
        
        self.test_pos_h = np.zeros(self.testTotal, dtype=np.int64)
        self.test_pos_t = np.zeros(self.testTotal, dtype=np.int64)
        self.test_pos_r = np.zeros(self.testTotal, dtype=np.int64)
        self.test_pos_h_addr = self.test_pos_h.__array_interface__['data'][0]
        self.test_pos_t_addr = self.test_pos_t.__array_interface__['data'][0]
        self.test_pos_r_addr = self.test_pos_r.__array_interface__['data'][0]
        
        self.test_neg_h = np.zeros(self.testTotal, dtype=np.int64)
        self.test_neg_t = np.zeros(self.testTotal, dtype=np.int64)
        self.test_neg_r = np.zeros(self.testTotal, dtype=np.int64)
        self.test_neg_h_addr = self.test_neg_h.__array_interface__['data'][0]
        self.test_neg_t_addr = self.test_neg_t.__array_interface__['data'][0]
        self.test_neg_r_addr = self.test_neg_r.__array_interface__['data'][0]
        self.relThresh = np.zeros(self.relTotal, dtype=np.float32)
        self.relThresh_addr = self.relThresh.__array_interface__['data'][0]

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
    
    def set_delta(self, delta):
        self.delta = delta

    def set_test_recom(self, test_recom):
        self.test_recom = test_recom

    def set_test_triple(self, test_triple):
        self.test_triple = test_triple

    def set_margin(self, margin):
        self.margin = margin

    def set_in_path(self, in_path):
        self.in_path = in_path

    def set_nbatches(self, nbatches):
        self.nbatches = nbatches

    def set_p_norm(self, p_norm):
        self.p_norm = p_norm

    def set_valid_steps(self, valid_steps):
        self.valid_steps = valid_steps

    def set_save_steps(self, save_steps):
        self.save_steps = save_steps

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def set_result_dir(self, result_dir):
        self.result_dir = result_dir

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_lmbda(self, lmbda):
        self.lmbda = lmbda

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_bern(self, bern):
        self.bern = bern

    def set_dimension(self, dim):
        self.hidden_size = dim

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_work_threads(self, work_threads):
        self.work_threads = work_threads

    def set_ent_neg_rate(self, rate):
        self.negative_ent = rate

    def set_rel_neg_rate(self, rate):
        self.negative_rel = rate
    
    def set_item_div_dict(self,item_div_dict):
        self.item_div_dict = item_div_dict

    def set_early_stopping_patience(self, early_stopping_patience):
        self.early_stopping_patience = early_stopping_patience

    def set_pretrain_model(self, pretrain_model):
        self.pretrain_model = pretrain_model

    def get_parameters(self, param_dict, mode='numpy'):
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

    def save_embedding_matrix(self, best_model, best_model1):
        path = os.path.join(self.result_dir, self.RecomModel.__name__ + '.json')
        path1 = os.path.join(self.result_dir, self.DivItemModel.__name__ + '.json')
        with open(path, 'w') as fw:
            fw.write(json.dumps(self.get_parameters(best_model, 'list')))
        with open(path1, 'w') as fw:
            fw.write(json.dumps(self.get_parameters(best_model1, 'list')))


    def set_train_model(self, model1, model2):
        print('Initializing training model...')
        self.RecomModel = model1
        self.DivItemModel = model2
        self.trainRecomModel = self.RecomModel(config=self)
        self.trainDivItemModel = self.DivItemModel(engine=self)
        if self.use_gpu:
            self.trainRecomModel.cuda()
            self.trainDivItemModel.cuda()
        if self.recomOptimizer != None:
            pass
        elif self.opt_method == 'Adam' or self.opt_method == 'adam':
            self.recomOptimizer = optim.Adam(
                    self.trainRecomModel.parameters(),
                    lr = self.alpha,
                    weight_decay = self.weight_decay)
        else:
            self.recomOptimizer = optim.SGD(
                    self.trainRecomModel.parameters(),
                    lr = self.alpha,
                    weight_decay = self.weight_decay)
        if self.divItemOptimizer != None:
            pass
        elif self.opt_method == 'Adam' or self.opt_method == 'adam':
            self.divItemOptimizer = optim.Adam(
                    self.trainDivItemModel.parameters(),
                    lr = self.alpha,
                    weight_decay = self.weight_decay)
        else:
            self.divItemOptimizer = optim.SGD(
                    self.trainDivItemModel.parameters(),
                    lr = self.alpha,
                    weight_decay = self.weight_decay)
        print('Finish initializing')

    def set_test_model(self, model, path=None):
        print('Initializing test model ...')
        self.RecomModel = model
        self.testRecomModel = self.RecomModel(config=self)
        if path == None:
            path = os.path.join(self.result_dir, self.RecomModel.__name__ + '.cpkt')
        self.testRecomModel.load_state_dict(torch.load(path))
        if self.use_gpu:
            self.testRecomModel.cuda()
        self.testRecomModel.eval()
        print('Finish initializing')

    def sampling(self):
        self.lib.sampling(
                self.batch_h_addr,
                self.batch_t_addr,
                self.batch_r_addr,
                self.batch_y_addr,
                self.batch_size,
                self.negative_ent,
                self.negative_rel,
                )

    def save_checkpoint(self, model1, model2, epoch):
        path1 = os.path.join(
                self.checkpoint_dir, self.DivItemModel.__name__ + '-' + str(epoch) + '.cpkt')
        path2 = os.path.join(
                self.checkpoint_dir, self.RecomModel.__name__ + '-' + str(epoch) + '.cpkt')
        torch.save(model1, path1)
        torch.save(model2, path2)

    def save_best_checkpoint(self, best_model1, best_model2):
        path1 = os.path.join(self.result_dir, self.DivItemModel.__name__ + '.cpkt')
        path2 = os.path.join(self.result_dir, self.RecomModel.__name__  + '.cpkt')
        torch.save(best_model1, path1)
        torch.save(best_model2, path2)

    def train_one_step(self):
        self.trainRecomModel.batch_h = to_var(self.batch_h, self.use_gpu)
        self.trainRecomModel.batch_t = to_var(self.batch_t, self.use_gpu)
        self.trainRecomModel.batch_r = to_var(self.batch_r, self.use_gpu)
        self.trainRecomModel.batch_y = to_var(self.batch_y, self.use_gpu)
        self.recomOptimizer.zero_grad()
        loss = self.trainRecomModel() + F.kl_div(self.trainRecomModel.item_embeddings.data, self.trainDivItemModel.item_embedding.weight.data)
        loss.backward()
        self.recomOptimizer.step()
        return loss.item()

    def train_one_batch(self, batch):
        self.divItemOptimizer.zero_grad()
        first, second, neg = batch
        first = torch.tensor(first)
        second = torch.tensor(second)
        neg = torch.tensor(neg)
        if self.use_gpu:
            first, second, neg = first.cuda(), second.cuda(), neg.cuda()
        pos_emb_div_val = self.trainDivItemModel(first, second)
        neg_emb_div_val = self.trainDivItemModel(first, neg)
        loss = self.use_criteron(pos_emb_div_val, neg_emb_div_val) + F.kl_div(self.trainRecomModel.item_embeddings.data, self.trainDivItemModel.item_embedding.weight.data)
 
        loss.backward()
        self.divItemOptimizer.step()
        loss = loss.item()
        return loss

    def test_one_step(self,model, test_u, test_i,test_r):
        model.batch_h = to_var(test_u, self.use_gpu)
        model.batch_t = to_var(test_i, self.use_gpu)
        model.batch_r = to_var(test_r, self.use_gpu)
        return model.predict()

    def valid(self, model):
        self.lib.validInit()
        for i in range(self.validTotal):
            sys.stdout.write('%d\r'%(i))
            sys.stdout.flush()
            self.lib.getValidHeadBatch(
                    self.valid_h_addr, self.valid_t_addr, self.valid_r_addr)
            res = self.test_one_step(model, self.valid_h, self.valid_t, self.valid_r)
            self.lib.validHead(res.__array_interface__['data'][0])

            self.lib.getValidTailBatch(
                    self.valid_h_addr, self.valid_t_addr, self.valid_r_addr)
            res = self.test_one_step(model, self.valid_h, self.valid_t, self.valid_r)
            self.lib.validTail(res.__array_interface__['data'][0])
        return self.lib.getValidHit10()

    def train(self, train_data, valid_data=None):
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        best_epoch = 0
        best_hit = 0.0
        best_model_divItem = None
        best_model_recom = None
        bad_counts = 0
        for epoch in range(self.train_times):
            self.trainDivItemModel.train()
            self.trainRecomModel.train()
            total_loss_DivItem = 0
            total_loss_recom = 0
            batch_size_DivItem = int(len(train_data)/self.nbatches)
            for i  in range(self.nbatches):
                batch_DivItem = self.get_random_block_from_data(train_data, batch_size_DivItem)
                # item_weight = self.RecomModel.item_embeddings.weight.data
                loss_DivItem = self.train_one_batch(batch_DivItem)
                total_loss_DivItem += loss_DivItem
                # print("Epoch {} | DivItem loss: {}".format(epoch,total_loss_DivItem))
                # item_weight = self.DivItemModel.item_embedding.weight.data
                self.sampling()
                loss_recom = self.train_one_step()
                total_loss_recom += loss_recom
            
            print("Epoch {} | DivItem loss: {}".format(epoch, total_loss_DivItem))
            print("Epoch {} | Recom loss: {}".format(epoch, total_loss_recom))
            if (epoch + 1) % self.valid_steps == 0:
                print("Epoch {} has finished, validating...".format(epoch))
                hit_DivItem, ndcg_DivItem = self.evaluate(valid_data)
                hit_recom = self.valid(self.trainRecomModel)
                if 9*hit_recom + hit_DivItem > best_hit:
                    best_hit = (9*hit_recom + hit_DivItem)/10
                    best_epoch = epoch
                    best_model_divItem = copy.deepcopy(self.trainDivItemModel.state_dict())
                    best_model_recom = copy.deepcopy(self.trainRecomModel.state_dict())
                    print("Best model | hit of valid set is {}".format(best_hit))
                    print("Best model | DivItem hit of valid set is {}".format(hit_DivItem))
                    bad_counts = 0
                else:
                    print("Hit of valid set is {} | bad count is {}".format(hit_recom, bad_counts))
                    bad_counts += 1
                if bad_counts == self.early_stopping_patience:
                    print("Early stopping at epoch {}".format(epoch))
                    break
        if best_model_recom == None:
            best_model_divItem = self.trainDivItemModel.state_dict()
            best_model_recom = self.trainRecomModel.state_dict()
            best_epoch = self.train_times - 1
            best_hit10 = self.valid(self.trainRecomModel)
        print("Best epoch is {} | hit of valid set is {}".format(best_epoch, best_hit))
        print('Store checkpoint of best result at epoch {}'.format(best_epoch))
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)
        self.save_best_checkpoint(best_model_divItem, best_model_recom)
        # self.save_best_checkpoint(best_model_divItem)
        self.save_embedding_matrix(best_model_recom, best_model_divItem)
        # self.save_embedding_matrix(best_model_divItem)
        print("Testing...")
        self.set_test_model(self.RecomModel)
        self.test()
        print("Finsh test")
        return best_model_recom, best_model_divItem

    
    def use_criteron(self,pos_emb_div_val,neg_emb_div_val):
        loss = -torch.log(torch.sigmoid( pos_emb_div_val - neg_emb_div_val))
        loss = loss.sum()
        return loss

    def get_random_block_from_data(self, data, batch_size):
        first = []
        second, neg = [],[]
        index_s = set()
        while len(index_s) < batch_size:
            index = randint(0,len(data))
            if index not in index_s:
                f, s, div_val = data[index]
                index_s.add(index)
                ## corrupt second item
                filtered_list = [e[0] for e in self.item_div_dict[f] if e[1]>= div_val ]
                neg_candid_list = [e for e in range(self.itemTotal) if e not in filtered_list]
                n = random.sample(neg_candid_list,1)
                first.append(f)
                second.append(s)
                neg.append(n)
        return first, second, neg

    def evaluate(self, data,n=10):
        self.trainDivItemModel.eval()
        data = pd.DataFrame(data,columns=['first','second','div_val'])
        first = list(data['first'])
        second = list(data['second'])
        div_val = list(data['div_val'])
        neg_first, neg_second = self.sample_negative(first, second,div_val, self.item_div_dict)
        if self.use_gpu:
            first = torch.LongTensor(first).cuda()
            second = torch.LongTensor(second).cuda()
            neg_first = torch.LongTensor(neg_first).cuda()
            neg_second = torch.LongTensor(neg_second).cuda()
        test_scores = self.trainDivItemModel(first, second)
        negative_scores = self.trainDivItemModel(neg_first, neg_second)
        if self.use_gpu:
            first = first.cpu()
            second = second.cpu()
            test_scores = test_scores.cpu()
            neg_first = neg_first.cpu()
            neg_second = second.cpu()
            negative_scores = negative_scores.cpu()
        
        hit, ndcg = 0, 0
        count = 0
        for f,s, test_val in zip(first.data.view(-1).tolist(),
               second.data.view(-1).tolist(),
               test_scores.data.view(-1).tolist()):
            neg_val = negative_scores.data.view(-1).tolist()[count*99:count*99 + 99]
            sorted(neg_val, reverse=True)
            if test_val >= neg_val[n-1]:
                hit += 1
                index = n-1
                while test_val >= neg_val[index] and index > -1:
                    index -= 1
                ndcg += 1./math.log2(index + 3)
            count += 1
        hit /= count
        ndcg /= count
        print('[Evaluating current model] Hit= {:.4f}, NDCG= {:.4f}\n'.format(hit, ndcg))
        return hit, ndcg

    def sample_negative(self, first, second, div_val, item_div_dict):
        neg_first = []
        neg_second = []
        count_first = set()
        for f, s, val in zip(first, second, div_val):
            pos_f_list = [e[0] for e in item_div_dict[f] if e[1] > self.delta]
            candid_list = [i for i in range(self.itemTotal) if i not in pos_f_list]
            samples = random.sample(candid_list, 99)
            for s in samples:
                neg_first.append(f)
                neg_second.append(s)
        # print('length neg_first, neg_second: {}, {}'.format(len(neg_first), len(neg_second)))
        return neg_first, neg_second
    
    def item_recommendation(self):
        print("The total of test triple is %d" % (self.testTotal))
        for i in range(self.testTotal):
            sys.stdout.write("%d\r" % (i))
            sys.stdout.flush()
            self.lib.getHeadBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
            res = self.test_one_step(
                self.testRecomModel, self.test_h, self.test_t, self.test_r
            )
            self.lib.testHead(res.__array_interface__["data"][0])
            
            self.lib.getTailBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
            res = self.test_one_step(
                self.testRecomModel, self.test_h, self.test_t, self.test_r
            )
            self.lib.testTail(res.__array_interface__["data"][0])
        self.lib.test_recommendation()


    def test(self):
        if self.test_recom:
            self.item_recommendation()
            # add renaming
            print('Renaming...')
            if os.path.exists(os.path.join(self.result_dir,  self.RecomModel.__name__ + ".json")):
                os.rename(os.path.join(self.result_dir,  self.RecomModel.__name__ + ".json"),
                os.path.join(self.result_dir,'{}_{}_{}_{}_{}_{}_{}.json'.format(self.RecomModel.__name__,
                self.margin, self.hidden_size,self.batch_size, self.bern, self.alpha,self.weight_decay)))
            if os.path.exists(os.path.join(self.result_dir,  self.DivItemModel.__name__ + ".json")):
                os.rename(os.path.join(self.result_dir,  self.DivItemModel.__name__ + ".json"),
                os.path.join(self.result_dir,'{}_{}_{}_{}_{}_{}_{}.json'.format(self.DivItemModel.__name__,
                self.margin, self.hidden_size,self.batch_size, self.bern, self.alpha,self.weight_decay)))

            if os.path.exists(os.path.join(self.result_dir,'no_constraint_test_link.txt')):
                os.rename(os.path.join(self.result_dir,"no_constraint_test_link.txt"),
                os.path.join(self.result_dir,"{}_{}_{}_{}_{}_{}_{}_no_constraint_test_link.txt".format(self.RecomModel.__name__,
                self.margin, self.hidden_size,self.batch_size,self.bern, self.alpha,self.weight_decay)))
                os.rename(os.path.join(self.result_dir,"type_constraint_test_link.txt"),
                os.path.join(self.result_dir, "{}_{}_{}_{}_{}_{}_{}_type_constraint_test_link.txt".format(self.RecomModel.__name__,
                self.margin, self.hidden_size,self.batch_size,self.bern, self.alpha,self.weight_decay)))
    
