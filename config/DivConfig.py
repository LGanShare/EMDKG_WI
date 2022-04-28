import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import sys
import os

class DivConfig(object):
    def __init__(self):
        self.inPath = '../data'
        self.batch_size = 100
        self.hidden_size = 75
        self.valid_steps = 10
        self.save_steps = 10
        self.opt_method = 'adam'
        self.optimizer = None
        self.alpha = alpha
        self.lr_decay = 0
        self.weight_decay = 0
        self.early_stopping_patience = 10
        self.nbatches = 100
        self.p_norm = 2
        self.test_accuracy = True
        self.test_diversity = True
        self.model = None
        self.trainModel = None
        self.testModel =None
        self.pretrainModel = None
        self.use_gpu = True

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_test_accuracy(self, test_accuracy):
        self.test_accuracy = test_accuracy

    def set_test_diversity(self, test_diversity):
        self.test_diversity = test_diversity

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
        self.alpha

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_dimension(self, dim):
        self.hidden_size = dim

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_ent_neg_rate(self,rate):
        self.negative_ent = rate

    def set_rel_neg_rate(self, rate):
        self.negative_rel = rate

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

    def save_embedding_matrix(self, model):
        path = os.path.join(self.result_dir, self.model.__name__ + '.json')
        with open(path, 'w') as f:
            f.write(json.dumps(self.get_parameters(best_model,'list')))
        
    def set_train_model(self, model):
        print('Initializing training model...')
        self.model = model
        self.trainModel = self.model(config=self)
        if self.use_gpu:
            self.trainModel.cuda()
        if self.optimizer != None:
            pass
        elif self.opt_method == 'Adam' or self.opt_method == 'adam':
            self.optimizer = optim.Adam(
                    self.trainModel.parameters(),
                    lr = self.alpha,
                    weight_decay = self.weight_decay)
        else:
            self.optimizer = optim.SGD(
                    self.trainModel.parameters(),
                    lr = self.alpha,
                    weight_decay = self.weight_decay)
        print('Finish initializing')

    def set_test_model(self, model, path=None):
        print('Initializing test model...')
        self.model = model
        self.testModel = self.model(config=self)
        if path == None:
            path = os.path.join(self.result_dir, self.model.__name__ + '.ckpt')
        self.testModel.load_state_dict(torch.load(path))
        if self.use_gpu:
            self.testModel.cuda()
        self.testModel.eval()
        print('Finish initializing')

    def save_checkpoint(self, model, epoch):
        path = os.path.join(
                self.checkpoint_dir, self.model.__name__ + '-'+ str(epoch) + '.cpkt')
        torch.save(model, path)

    def save_best_checkpoint(self, best_model):
        path = os.path.join(self.result_dir, self.model.__name__ + '.cpkt')
        torch.save(best_model, path)

    def train_one_step(self):
        pass

    def test_one_step(self):
        pass

    def valid(self, model):
        pass

    def train(self):
        pass

    def test(self):
        pass






