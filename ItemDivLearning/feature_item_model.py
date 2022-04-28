import numpy as np
import random
import math

class feature_item_x(object):
    def __init__(self, item_features):
        self.item_features = item_features
        # self.item_features = self.get_item_features(item_features)
        self.items, self.reverse_items = self._get_items(item_features)
        self.features, self.reverse_features =  self._get_features(item_features)
        self.item_features = self.get_item_features(item_features)

        self.delta = 0.25

    # delta is the threshold of div value
    def set_delta(self, val):
        self.delta = val

    def _get_items(self,item_features):
        # count = 0
        max_id = 0
        item_dict = {}
        reverse_item_dict = {}
        for row in item_features:
            item = int(row[0])
            if max_id < item:
                max_id = item
            # item_dict[count] = item
            # reverse_item_dict[item] = count
            # count += 1
        for i in range(max_id+1):
            item_dict[i] = i
            reverse_item_dict[i] = i
            # print('item original id: {}, new id:{}'.format(i,i))
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
            i = self.reverse_items[int(row[0])]
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
            if item not in self.item_features:
                # print('item {} unknown feature information'.format(item))
                feature_map.append(tmp_fm)
                continue
            for i in range(1,len(self.item_features[item])):
                f = self.item_features[item][i]
                # print('feature index {} of item {}: {}'.format(i-1,item, f))
                if not f == '':
                    # index = self.reverse_features[f]
                    tmp_fm[f] = 1
            feature_map.append(tmp_fm)
        # print('feature map of itemlist {}: {}'.format(itemlist,feature_map))
        return feature_map
    
    def gen_itempairs(self, sample_size=200,prop=0.8,t='alpha-ndcg',n=5):
        total_lists = []
        item_div_set = []
        for i in range(len(self.items)):
            if (i+1)%500 == 0:
                print('Finish item {}'.format(i))
            count = 0
            while count < sample_size:
                candid_items = [k for k in range(len(self.items)) if k != i]
                sampled_items = random.sample(candid_items,n)
                candid_list = set(np.append(sampled_items,i))
                if candid_list not in total_lists:
                    fm = self.gen_feature_map(candid_list)
                    candid_div_val = self.feature_div(list(candid_list),fm,t)
                    # print('div val: {}'.format(candid_div_val))
                    if candid_div_val > self.delta:
                        ## candid_div_val is still a set
                        total_lists.append(candid_list)
                    count += 1
            # print('last two itemlists: {}'.format(total_lists[-2:]))
        self.item_div_set = total_lists
        total_lists = [list(l) for l in total_lists]
        train_lists, valid_lists,test_lists = [],[],[]
        random.shuffle(total_lists)
        train_lists = total_lists[:int(len(total_lists)*prop)]
        valid_lists = total_lists[int(len(total_lists)*prop):int(len(total_lists)*(.5+prop/2.))]
        test_lists = total_lists[int(len(total_lists)*(0.5+0.5*prop)):]
        return train_lists,valid_lists,test_lists
    
    
    def get_itemlists(self,itemlists_dict,item_mapping):
        train_itemlists = {}
        valid_itemlists = {}
        test_itemlists = {}
        for u in itemlists_dict:
            itemlists_u = itemlists_dict[u]
            
            if len(itemlists_u) < 10:
                continue
            item_div_set_u = [set(e) for e in itemlists_u]
            self.item_div_set[u] = item_div_set_u
            # idx = random.choice(range(len(itemlists_u)))
            # test_itemlist_u = itemlists_u[idx]
            # itemlists_u.pop(idx)
            random.shuffle(itemlists_u)
            train_itemlists_u = itemlists_u[:int(len(itemlists_u)*.8)]
            valid_itemlists_u = itemlists_u[int(len(itemlists_u)*.8):int(len(itemlists_u)*.9)]
            test_itemlists_u = itemlists_u[int(len(itemlists_u)*.9):]
            ## use item mapping to project to div item learning id
            test_itemlists_u = [[item_mapping[e] for e in test_l_u] for test_l_u in test_itemlists_u]
            train_itemlists_u = [[item_mapping[e] for e in train_l_u] for train_l_u in train_itemlists_u]
            valid_itemlists_u = [[item_mapping[e] for e in valid_l_u] for valid_l_u in valid_itemlists_u]
            
            train_itemlists[u] = train_itemlists_u
            valid_itemlists[u] = valid_itemlists_u
            test_itemlists[u] = test_itemlists_u
        return train_itemlists, valid_itemlists, test_itemlists


    def gen_itemlists(self,sample_size=200,prop=0.8,t='pairwise'):
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
                    divVal = self.feature_div([0,1],fm,t)
                    print('divVal: {}'.format(divVal))
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

    def feature_div(self,itemlist,feature_map,t='pairwise',alpha=0.5):
    # feature_map is a one_hot matrix of size M*P
    # M is the num of items and P is the num of features
        if t=='pairwise':
            rs = 0
            for i in range(len(itemlist)):
                item_i = itemlist[i]
                for j in range(i+1, len(itemlist)):
                    item_j = itemlist[j]
                    for k in range(len(feature_map[0])):
                        # print('k is :{}'.format(k))
                        # print('features of item {} is : {}'.format(i,feature_map[i]))
                        # print('features of item {} is :{}'.format(j,feature_map[j]))
                        rs += feature_map[i][k] ^ feature_map[j][k]
            rs /= len(feature_map[0])
            print('pairwise div val for {} is: {}'.format(itemlist,rs))
            return rs
        elif t=='cc':
            rs = 0
            for k in range(len(feature_map[0])):
                for i in range(len(itemlist)):
                    if feature_map[i][k] == 1:
                        rs += 1
                        break
            rs /= len(feature_map[0])
            # print(rs)
            return rs
        elif t=='alpha-ndcg':
            rs = 0
            # print('total item features number: {}'.format(len(feature_map[0])))
            # print('feature_map:{}'.format(feature_map))
            num_features = len(feature_map[0])
            r_i = [0 for k in range(num_features)]

            for i in range(len(itemlist)):
                G_i = 0
                for index,g in enumerate(feature_map[i]):
                    if not g == 0:
                        # print('type g: {}'.format(type(g)))
                        G_i += math.pow(1-alpha,r_i[index])
                        r_i[index] += 1
                rs += G_i/math.log(2 + i) 
            rs /= num_features
            # print('Div val alpha-ndcg for {} is {}'.format(itemlist,rs))
            return rs
        elif t == 'dCC':
            rs = 0
            num_features = len(feature_map[0])
            r_i = [0 for k in range(num_features)]

            for i in range(len(itemlist)):
                G_i = 0
                for index,g in enumerate(feature_map[i]):
                    if not g == 0:
                        # print('type g: {}'.format(type(g)))
                        G_i += math.pow(1-alpha,r_i[index])
                        r_i[index] += 1
                rs += G_i
            rs /= num_features
            # print('dCC div val for {} is: {}'.format(itemlist,rs))
            return rs
            

    def gen_div_val(self,itemlists,feature_map,t='alpha-ndcg'):
    # for i in itemlists:
    #     print('itemlist: {}\t diversity value:{}'.format(i, feature_div(i,feature_map,t)))
        div_val = [self.feature_div(i,feature_map,t) for i in itemlists]
        # print('diversity value :{}'.format(div_val))
    # res = np.concatenate((itemlists,div_val),axis=1)
    # for i in range(len(itemlists)):
    #     print('itemlist with div val: {}'.format(itemlists[i]+[div_val[i]]))
        res = [(itemlists[i],div_val[i]) for i in range(len(itemlists))]
        # res = [itemlists,div_val]
        return res



