import numpy as np
import pandas as pd
import random
import csv 
# from DivLearning import DivRepr, Engine


def feature_div(itemlist,feature_map,t='pairwise'):
    # feature_map is a one_hot matrix of size M*P
    # M is the num of items and P is the num of features
    if t=='pairwise':
        rs = 0
        for i in range(len(itemlist)):
            for j in range(i+1, len(itemlist)):
                # print('item {} feature map is : {}'.format(itemlist[i],feature_map[itemlist[i]]))
                # print('item {} feature map is : {}'.format(itemlist[j],feature_map[itemlist[j]]))
                for k in range(len(feature_map[0])):
                    rs += feature_map[itemlist[i]][k]^feature_map[itemlist[j]][k]
        rs /= len(feature_map[0])
        return rs
    elif t=='piecewise':
        rs = 0
        for k in range(len(feature_map[0])):
            for i in range(len(itemlist)):
                if feature_map[i][k] == 1:
                    rs += 1
                    break
        rs /= len(feature_map[0])
        return rs
    elif t=='entropy':
        pass

def gen_div_val(itemlists,feature_map,t='pairwise'):
    # for i in itemlists:
    #     print('itemlist: {}\t diversity value:{}'.format(i, feature_div(i,feature_map,t)))
    div_val = [feature_div(i,feature_map,t) for i in itemlists]
    print('diversity value :{}'.format(div_val))
    # res = np.concatenate((itemlists,div_val),axis=1)
    # for i in range(len(itemlists)):
    #     print('itemlist with div val: {}'.format(itemlists[i]+[div_val[i]]))
    res = [itemlists[i]+[div_val[i]] for i in range(len(itemlists))]
    return res


class feature_item(object):
    def __init__(self, item_features):
        self.item_features = item_features
        self.items, self.reverse_items = self._get_items(item_features)
        self.features, self.reverse_features =  self._get_features(item_features)
        self.delta = 0.25

    # delta is the threshold of div value
    def set_delta(self, val):
        self.delta = val

    def _get_items(self,item_features):
        count = 0
        item_dict = {}
        reverse_item_dict = {}
        for row in item_features:
            item = row[0]
            item_dict[count] = item
            reverse_item_dict[item] = count
            count += 1
            # print('item original id: {}, new id:{}'.format(item,count-1))
        return item_dict, reverse_item_dict

    def _get_features(self,item_features):
        count = 0
        feature_dict = {}
        reverse_feature_dict = {}
        for row in item_features:
            for f in row[1:]:
                if not f == None:
                    if f not in reverse_feature_dict:
                        reverse_feature_dict[f] = count
                        feature_dict[count] = f
                        count += 1
                        # print('feature original id: {}, new id: {}'.format(f,count))
        return feature_dict, reverse_feature_dict
    
    def get_item_features(self,itemlist=None):
        # if itemlist == None:
        #     itemlist = self.items.keys()
        res = {}
        # to do
        for nrow in range(len(self.item_features)):
            row = self.item_features[nrow]
            i = self.reverse_items[row[0]]
            for fo in row[1:]:
                if not fo == '':
                    f = self.reverse_features[fo]
                    res.setdefault(i,[]).append(f)
        return res
        
    def gen_feature_map(self,itemlist):
        # itemlist uses original item ids
        feature_map = []
        for item in itemlist:
            tmp_fm = [0 for i in range(len(self.features))]
            # print('item: {}'.format(item))
            # print(self.item_features[item])
            for i in range(1,len(self.item_features[item])):
                f = self.item_features[item][i]
                if not f == '':
                    index = self.reverse_features[f]
                    tmp_fm[index] = 1
            feature_map.append(tmp_fm)
        return feature_map

    def gen_itemlists(self,sample_size=200,prop=0.8):
        # item pair (item list of size 2)
        total_pairs = []
        item_div_dict = {}
        for i in range(len(self.items)):
            ### movielens sample 200 ####
            ### anime sample 500 #####
            sampled_items = random.sample(range(len(self.items)),sample_size)
            for j in sampled_items:
                if not j < i:
                    # get feature map of [i,j]
                    fm = self.gen_feature_map([i,j])
                    divVal = feature_div([0,1],fm)
                    # print('divVal: {}'.format(divVal))
                    ################### divVal~0.18 for anime; ~0.30 for movie
                    if divVal > self.delta:
                        total_pairs.append([i,j])
                        item_div_dict.setdefault(i,[]).append((j,divVal))
                        item_div_dict.setdefault(j,[]).append((i,divVal))
                        # print('({},{}) div value:{}'.format(i,j,divVal))
        self.item_div_dict = item_div_dict
        random.shuffle(total_pairs)
        train_pairs = total_pairs[:int(len(total_pairs)*0.8)]
        valid_pairs = total_pairs[int(len(total_pairs)*0.8):int(len(total_pairs)*0.9)]
        test_pairs = total_pairs[int(len(total_pairs)*0.9):]
        return train_pairs, valid_pairs,test_pairs

    def feature_div(self,itemlist,feature_map,t='pairwise'):
    # feature_map is a one_hot matrix of size M*P
    # M is the num of items and P is the num of features
        if t=='pairwise':
            rs = 0
            for i in range(len(itemlist)):
                for j in range(i+1, len(itemlist)):
                # print('item {} feature map is : {}'.format(itemlist[i],feature_map[itemlist[i]]))
                # print('item {} feature map is : {}'.format(itemlist[j],feature_map[itemlist[j]]))
                    for k in range(len(feature_map[0])):
                        rs += feature_map[itemlist[i]][k]^feature_map[itemlist[j]][k]
            rs /= len(feature_map[0])
            return rs
        elif t=='piecewise':
            rs = 0
            for k in range(len(feature_map[0])):
                for i in range(len(itemlist)):
                    if feature_map[i][k] == 1:
                        rs += 1
                        break
            rs /= len(feature_map[0])
            return rs
        elif t=='entropy':
            pass

    def gen_div_val(self,itemlists,feature_map,t='pairwise'):
    # for i in itemlists:
    #     print('itemlist: {}\t diversity value:{}'.format(i, feature_div(i,feature_map,t)))
        div_val = [self.feature_div(i,feature_map,t) for i in itemlists]
        # print('diversity value :{}'.format(div_val))
    # res = np.concatenate((itemlists,div_val),axis=1)
    # for i in range(len(itemlists)):
    #     print('itemlist with div val: {}'.format(itemlists[i]+[div_val[i]]))
        res = [itemlists[i]+[div_val[i]] for i in range(len(itemlists))]
        return res


                

if __name__=='__main__':
    item_feature_file ='../data/anime_features.csv'
    with open(item_feature_file,'r') as fr:
        csv_item_feature = csv.reader(fr,delimiter=',')
        item_features = [list(row) for row in csv_item_feature]
        print('Succesfully read item features from file.')
    # print('Movie features:{}'.format(item_features))
    
    # extract item features for feature map and train/test item lists
    fi = feature_item(item_features)
    train_itemlists, valid_itemlists,test_itemlists = fi.gen_itemlists()
    item_div_dict = fi.item_div_dict
    #print('train_itemlists: {}'.format(train_itemlists)) 
    
    full_itemlist = list(range(len(fi.items)))
    feature_map = fi.gen_feature_map(full_itemlist)
    # feature_map = fi.gen_feature_map(full_itemlist)
    print('Finish generating feature map.')
    # print('Feature map: {}'.format(feature_map))
    
    train_itemlists_with_div_val = fi.gen_div_val(train_itemlists, feature_map)
    valid_itemlists_with_div_val = fi.gen_div_val(valid_itemlists, feature_map)
    test_itemlists_with_div_val = fi.gen_div_val(test_itemlists, feature_map)
    print('Finish diversity value calculation for each itemlist.')
    print(train_itemlists_with_div_val[0])

    num_items = len(fi.items)
    hidden_dim = 75
    epoch = 500
    batch_size = 1000
    lr = 0.01
    use_cuda = True
    optimizer_type = 'adam'
    l2_lambda = 0.00001

    model = DivRepr(num_items,hidden_dim)
    engine = Engine(num_items,hidden_dim,epoch, batch_size,lr,use_cuda,optimizer_type,l2_lambda,item_div_dict)
    print('Begin training.')
    engine.train(train_itemlists_with_div_val, valid_itemlists_with_div_val)
    print('End training.')
    print('Begin evaluation.')
    hit,ndcg = engine.evaluate(test_itemlists_with_div_val,200)
    print('Test hit, ndcg: {}'.format(hit, ndcg))

