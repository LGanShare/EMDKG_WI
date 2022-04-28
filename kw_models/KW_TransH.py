import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .BaseModel import BaseModel
from torch.autograd import Variable

class TransH(BaseModel):
    def __init__(self,config):
        super(TransH, self).__init__(config)
        ent_embeddings = torch.empty(self.config.entTotal,self.config.hidden_size)
        self.user_embeddings = nn.Parameter(torch.empty(self.config.userTotal,self.config.hidden_size))
        self.item_embeddings = nn.Parameter(torch.empty(self.config.itemTotal,self.config.hidden_size))
        self.trainable_ent_embeddings = nn.Parameter(torch.empty(self.config.entTotal-self.config.userTotal-self.config.itemTotal,self.config.hidden_size))
        self.rel_embeddings = nn.Embedding(self.config.relTotal,self.config.hidden_size)
        self.norm_vector = nn.Embedding(self.config.relTotal,self.config.hidden_size)
        self.criterion = nn.MarginRankingLoss(self.config.margin,False)
        
        ent_embeddings[:self.config.userTotal] = self.user_embeddings
        # add item embedding
        ent_embeddings[self.config.userTotal:self.config.userTotal + self.config.itemTotal] = self.item_embeddings
        ent_embeddings[self.config.userTotal+self.config.itemTotal:] = self.trainable_ent_embeddings
        self.register_buffer('ent_embeddings', ent_embeddings)
        # self.register_parameter('user_embeddings',user_embeddings)
        # self.register_parameter('item_embeddings',item_embeddings)
        # self.register_parameter('rel_embeddings',rel_embeddings)
        # self.register_parameter('norm_vector',norm_vector)
        self.init_weight()

    def init_weight(self, item_weight=None):
        if item_weight == None:
            pass
        else:
            self.item_embeddings.weight = item_weight
        nn.init.xavier_uniform_(self.user_embeddings)
        # add item embedding init
        nn.init.xavier_uniform_(self.item_embeddings)
        nn.init.xavier_uniform_(self.trainable_ent_embeddings)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform_(self.norm_vector.weight.data)
        pass

    def _calc(self, h, t, r):
        return torch.norm(h + r - t,self.config.p_norm,-1)

    def _transfer(self, e, norm):
        norm = F.normalize(norm, p = 2, dim = -1)
        return e - torch.sum(e * norm, -1, True) * norm

    def loss(self, p_score, n_score):
        if self.config.use_gpu:
            y = Variable(torch.Tensor([-1]).cuda())
        else:
            y = Variable(torch.Tensor([-1]))
        return self.criterion(p_score,n_score,y)

    def forward(self):
        self.ent_embeddings.detach_()
        self.ent_embeddings[:self.config.userTotal] = self.user_embeddings
        # add item embedding
        self.ent_embeddings[self.config.userTotal: self.config.userTotal+self.config.itemTotal] = self.item_embeddings
        self.ent_embeddings[self.config.userTotal+self.config.itemTotal:] = self.trainable_ent_embeddings
        h = nn.functional.embedding(self.batch_h,self.ent_embeddings)
        t = nn.functional.embedding(self.batch_t,self.ent_embeddings)
        r = self.rel_embeddings(self.batch_r)
        r_norm = self.norm_vector(self.batch_r)
        h = self._transfer(h, r_norm)
        t = self._transfer(t, r_norm)
        score = self._calc(h,t,r)
        p_score = self.get_positive_score(score)
        n_score = self.get_negative_score(score)
        return self.loss(p_score,n_score)

    def predict(self):
        self.ent_embeddings.detach_()
        self.ent_embeddings[:self.config.userTotal] = self.user_embeddings
        # add item embedding
        self.ent_embeddings[self.config.userTotal:self.config.userTotal+self.config.itemTotal] = self.item_embeddings
        self.ent_embeddings[self.config.userTotal+self.config.itemTotal:] = self.trainable_ent_embeddings
        u = nn.functional.embedding(self.batch_h, self.ent_embeddings)
        i = nn.functional.embedding(self.batch_t, self.ent_embeddings)
        r = self.rel_embeddings(self.batch_r)
        r_norm = self.norm_vector(self.batch_r)
        u = self._transfer(u, r_norm)
        i = self._transfer(i, r_norm)
        score = self._calc(u,i,r)
        return score.detach().cpu().numpy()


