import numpy as np
import csv
import random
from scipy import stats

def gen_itemlist_random(candid_items,ui_weights,item_features,delta=3,max_per_user=10000):
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
        items = [i for i in candid_items[u] if weights[i] > delta]
        j = 0
        while len(items) == 0 and j < 4:
            items = [i for i in candid_items[u] if weights[i] > delta//2]
            j += 1
        
        if len(items) == 0:
            print('No candidate items for user {}. Just ignore it.'.format(u))
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
        
        num_loop = 0
        while num_loop < max_per_user and count_items < len(items):
            tmp_l = []
            sign_features = [0 for i in features]
            count_features = np.cumsum(sign_features)[-1]
            
            count_lists = 0
            count_ll = 0
            while count_ll < 3000 and count_lists < max_per_user and count_items< 2* len(items) and  count_features < len(features) and len(tmp_l) < 10:
                count_ll += 1 
                c_items = [i for i in items if i not in tmp_l]
                c_weights = [weights[i] for i in c_items]
                i = random.choices(c_items,weights=c_weights)[0]
                
                index_i = items.index(i)
                if sign_items[index_i] == 2:
                    # print('Randomly select an invalid item.')
                    continue
                
                # print('tmp_l diversity: {}'.format(get_diversity(tmp_l,item_features)))
                # print('tmp_l + [{}] diversity: {}'.format(i, get_diversity(tmp_l + [i], item_features)))

                if get_diversity(tmp_l, item_features) <= get_diversity(tmp_l+[i],item_features):
                    # print('tmp_l diversity: {}'.format(get_diversity(tmp_l,item_features)))
                    # print('tmp_l + [{}] diversity: {}'.format(i, get_diversity(tmp_l, item_features)))
                    tmp_l.append(i)
                    sign_items[index_i] += 1
                    count_items= list(np.cumsum(sign_items))[-1]
                    if int(i) not in item_features:
                        # print('item {} not in item features keys.'.format(i))
                        pass
                    else:
                        for feature in item_features[int(i)]:
                            sign_features[features.index(feature)] = True
                    count_features = np.cumsum(sign_features)[-1]
                count_lists += 1
                pass
            if len(tmp_l) > 1:
                print(tmp_l)
                res.setdefault(u, []).append(tmp_l)
            num_loop += 1
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

        index = 0
        while index < len(items):
            tmp_l = []
            
            sign_features = [False for f in features]
            count_features = np.cumsum(sign_features)[-1]
            while index < len(items) and count_features < len(features):
                print('tmp_l : {}'.format(tmp_l))
                if get_diversity(tmp_l, item_features) < get_diversity(tmp_l+[items[index][0]], item_features):
                    tmp_l.append(items[index][0])
                    for feature in item_features[items[index][0]]:
                        sign_features[features.index(feature)] = True
                    count_features = np.cumsum(sign_features)[-1]
                else:
                    pass
                index += 1
            if len(tmp_l) > 1:
                res.setdefault(u, []).append(tmp_l)
            index = max(item.index(tmp_l[1]),item.index(tmp_l[-2]))

    return res
    pass

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

def get_diversity(itemlist, item_features, div_type='entropy'):
    """
    itemlist: list, list of item ids
    item_features: dict, keys are item ids
    =========
    type:
    -> coverage: count the number of categories
    -> entropy: count the frequency entropy of the itemlist
    """
    itemlist = list(itemlist)
    
    if div_type == 'coverage':
        f_set = set()
        for i in itemlist:
            try:
                i = int(i)
                for e in item_features[i]:
                    f_set.add(e)
                return len(f_set)
            except IndexError as e0:
                print('Index error in function get_diversity')
                print(e0)
            except KeyError as e1:
                # print('item {} is not contained in the keys.'.format(i))
                pass
        return len(f_set)
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
                print('Index error in function ge_diversity')
            except KeyError as e1:
                pass
        return res
    pass


if __name__ == '__main__':
    val = 5
    l = [1,1,2,2,3,1]
    cumsum_l = list(np.cumsum(l))
    print('value: {}'.format(val))
    print('cumsum list: {}'.format(cumsum_l))
    print('Find interval index: {}'.format(_find(val,cumsum_l)))

