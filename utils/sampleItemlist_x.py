import numpy as np
import csv
import random
import math
from scipy import stats

def gen_itemlist_random_x(candid_items,ui_weights,item_features,div_type='cc',delta=3,num_features=18,max_per_user=10000):
    """
    candid_items: dict, keys are user ids
    ui_weights: dict, keys are user ids, values are dict with keys item ids
    item_features: dict, keys are item ids
    delta: the threshold of item score to be taken considered as a positive item
    """
    res = {}
    for u in candid_items:
        weights = ui_weights[u]
        print('================')
        print('Generating itemlists for user {}...'.format(u))
        print('weights: {}'.format(weights))
        print('candid items for user {}: {}'.format(u,candid_items[u]))
        items = [i for i in candid_items[u] if (i in weights) and (weights[i] >= delta)]
        j = 0
        
        ## dynamicly deduce the barrier delta when the average item weight is low
        while len(items) == 0 and j < 4:
            items = [i for i in candid_items[u] if weights[i] > delta//math.pow(2,j+1)]
            j += 1
        
        if len(items) <= 10:
            print('Not enough candidate items for user {}. Just ignore it.'.format(u))
            continue

        features = []
        
        for i in items:
            try:
                i = int(i)
                features += item_features[i]
                features = list(set(features))
            except KeyError as e:
                # print('item {} is not contained in the keys of item_feature.'.format(i))
                pass
        print('features: {}'.format(features))
        if len(features) == 0:
            continue
        

        item_weights = [weights[i] for i in items if weights[i]>=delta]
        sign_items = [False for i in items]
        count_items = list(np.cumsum(sign_items))[-1]

        res_u = []
        _val_u = []
        count_loop = 0
        while len(res_u) < max(max_per_user,50000) and count_loop < 50000:
            count_loop += 1
            if len(items) >= 10:
                candid_list_u = set(random.sample(items,10))
            # elif len(items) >= 5:
            #     candid_list_u= set(random.sample(items,5))
            
            div_val = get_diversity(list(candid_list_u),item_features,div_type=div_type,num_features=num_features)
            
            # print('diversity of candid list {}: {}'.format(candid_list_u,div_val))
            # if get_diversity(list(candid_list_u),item_features) > 0.132 and candid_list_u not in res_u:
            #     print('diversity of candid list {}: {}'.format(candid_list_u,div_val))
            #     res_u.append(candid_list_u)
            if candid_list_u not in res_u:
                res_u.append(candid_list_u)
                _val_u.append((candid_list_u,div_val))
        print(_val_u[0])        
        _val_u = sorted(_val_u,key=lambda x:x[1], reverse=True)
        if len(_val_u) >= 5:
            for i in range(5):
                print('{}:{}\n'.format(_val_u[i][0],_val_u[i][1]))

        max_index = min(100,int(0.5*len(_val_u)))
        res_u = [list(e[0]) for e in _val_u[:max_index]]
        res[u] = res_u

    return res


def gen_itemlist_timely(candid_items, weights, item_features, delta=5, overlap=2):
    """
    candid_items: dict, keys are user ids, values are pairs of item ids and timestamps
    weights: dict, keys are item ids
    item_features: dict, keys are item ids
    delta: the threshold for an item to be considered as a positive item
    overlap: int, the maixmum number of common items between any two generated itemlists
    """
    res = {}
    for u in candid_items:
        print('================')
        print('Generating itemlists for user {}...'.format(u))
        # i is a tuple containing an item id and a timestamp
        items = [i for i in candid_items[u] if weights[i[0]] > delta]
        items = sorted(items, key=lambda x:x[1], reverse=True)
        features = []
        for i in items:
            features += item_features[i[0]]
            features = list(set(features))

def _find(val, intervals):
    """
    val: int, value to be found in the intervals
    intervals: list of increasing positive int values
    """
    if len(intervals) == 1:
        if val < intervals[0]:
            return 0
        else:
            # -1 represents no valid index returned 
            return -1
    lef = 0
    rig = len(intervals)-1
    
    if val >= intervals[rig] or val < 0:
        return -1
    elif val < intervals[lef]:
        return 0

    while lef < rig:
        mid = (lef + rig) // 2
        if val < intervals[mid]:
            rig = mid
        elif val >= intervals[mid+1]:
            lef = mid + 1
        else:
            return mid + 1
    return lef

def get_diversity(itemlist, item_features, div_type='alpha-ndcg',num_features=18,alpha=0.5):
    """
    itemlist: list, list of item ids
    item_features: dict, keys are item ids
    =========
    type:
    -> coverage: count the number of categories
    -> entropy: count the frequency entropy of the itemlist
    """
    itemlist = list(itemlist)
    
    if div_type == 'cc':
        rs = 0
        r_i = [0 for k in range(num_features)]
        for i in itemlist:
            # print(i)
            try:
                i = int(i)
                if i not in item_features:
                    # print('item {} does not have feature information'.format(i))
                    continue
                for e in item_features[i]:
                    r_i[e] = 1
            except IndexError as e0: 
                print('Index error in function get_diversity')
                pass
            except KeyError as e1:
                pass
        rs = sum(r_i)/ num_features
        return rs
    elif div_type == 'entropy':
        f_dict = {}
        res = 0
        for i in itemlist:
            try:
                i = int(i)
                for e in item_features[i]:
                    f_dict[e] = f_dict.setdefault(e,0) + 1 
                count_f = sum(f_dict.values())
                prob_f = [f_dict[e]/count_f for e in f_dict]
                res = stats.entropy(prob_f)
            except IndexError as e0:
                pass
                # print('Index error in function ge_diversity')
            except KeyError as e1:
                pass
        return res
    elif div_type=='alpha-ndcg':
        rs = 0
        r_i = [0 for k in range(num_features)]
        for i in itemlist:
            try:
                i = int(i)
                G_i = 0
                # for e in item_features[i]:
                #     print(e)
                for idx,g in enumerate(range(num_features)):
                    # print('type g: {}'.format(type(g)))
                    if g in item_features[i]:
                        G_i += math.pow(1-alpha,r_i[idx])
                        r_i[idx] += 1
                rs += G_i/math.log(2 + i) 
            except IndexError as e0:
                pass
                # print('Index error in function get_diversity')
            except KeyError as e1:
                pass
        rs /= num_features
        return rs
    elif div_type == 'dCC':
        rs = 0
        r_i = [0 for k in range(num_features)]
        for i in itemlist:
            # print(i)
            try:
                i = int(i)
                G_i = 0
                if i not in item_features:
                    # print('item {} does not have feature information'.format(i))
                    continue
                for e in item_features[i]:
                    G_i += math.pow(1-alpha,r_i[e])
                    r_i[e] += 1
                rs += G_i/math.log(2+i)
            except IndexError as e0:
                pass 
                print('Index error in function get_diversity')
            except KeyError as e1:
                pass
        rs /= num_features
        return rs


if __name__ == '__main__':
    val = 5
    l = [1,1,2,2,3,1]
    cumsum_l = list(np.cumsum(l))
    print('value: {}'.format(val))
    print('cumsum list: {}'.format(cumsum_l))
    print('Find interval index: {}'.format(_find(val,cumsum_l)))

