from ItemDivLearning import feature_item
from utils import gen_itemlist_random_y, gen_itemlist_random_x
from utils import user_items, get_diversity
import numpy as np
import csv
import json
import os

def get_user_item_weight(user_item,dirpath, o_u_i_filei,sep='\t'):
    user_mapping = {}
    reverse_user_mapping = {}
    item_mapping = {}
    reverse_item_mapping = {}

    entity_file = os.path.join(dirpath, 'entity2id.txt')
    with open(entity_file,'r') as fr:
        count = int(fr.readline())
        line = fr.readline().split('\t')
        while line and line != ['']:
            if line[0][:4] == 'user':
                u_o_id = int(line[0][5:])
                u_id = int(line[1])
                user_mapping[u_id] = u_o_id
                reverse_user_mapping[u_o_id] = u_id
            if line[0][:5] == 'movie':
                m_o_id = int(line[0][6:])
                m_id = int(line[1])
                item_mapping[m_id] = m_o_id
                reverse_item_mapping[m_o_id] = m_id
            if line[0][:4] == 'User':
                u_o_id = int(line[0][5:])
                u_id = int(line[1])
                user_mapping[u_id] = u_o_id
                reverse_user_mapping[u_o_id] = u_id
            if line[0][:5] == 'Anime':
                m_o_id = int(line[0][6:])
                m_id = int(line[1])
                item_mapping[m_id] = m_o_id
                reverse_item_mapping[m_o_id] = m_id
            line = fr.readline().split('\t')
    
    full_u_o_i_o = {}
    with open(o_u_i_file, 'r') as fr:
        ## for anime 
        line = fr.readline()
        ## \t for ml-100k_imdb, , for anime
        line = fr.readline().split(sep)
        while line and line != ['']:
            u_o_id, i_o_id, u_i_weight = int(line[0]), int(line[1]), int(line[2])
            line = fr.readline().split(sep)
            if u_i_weight > 0:
                full_u_o_i_o.setdefault(u_o_id, {}).setdefault(i_o_id, u_i_weight)
   
    res = {}
    for u in user_item:
        items = user_item[u]
        u_o_id = user_mapping[u]
        items_weight_o = full_u_o_i_o[u_o_id]
        for i in items:
            i_o_id = item_mapping[i]
            if i_o_id in items_weight_o:
                w = items_weight_o[i_o_id]
                res.setdefault(u,{}).setdefault(i, w)
    return res



def get_train_valid_test_items(dirpath):
    train_file = os.path.join(dirpath, 'train2id.txt')
    valid_file = os.path.join(dirpath, 'valid2id.txt')
    test_file = os.path.join(dirpath, 'test2id.txt')

    train_user_item = {}
    with open(train_file, 'r') as fr:
        count = int(fr.readline())
        line = fr.readline().split('\t')
        while line and line != ['']:
            if int(line[2]) == 0:
                u_id = int(line[0])
                m_id = int(line[1])
                train_user_item.setdefault(u_id,[]).append(m_id)
            line = fr.readline().split('\t')

    valid_user_item = {}
    with open(valid_file, 'r') as fr:
        count = int(fr.readline())
        line = fr.readline().split('\t')
        while line and line != ['']:
            if int(line[2]) == 0:
                u_id = int(line[0])
                m_id = int(line[1])
                valid_user_item.setdefault(u_id,[]).append(m_id)
            line = fr.readline().split('\t')

    test_user_item = {}
    with open(test_file, 'r') as fr:
        count = int(fr.readline())
        line = fr.readline().split('\t')
        while line and line != ['']:
            if int(line[2]) == 0:
                u_id = int(line[0])
                m_id = int(line[1])
                test_user_item.setdefault(u_id,[]).append(m_id)
            line = fr.readline().split('\t')
    return train_user_item, valid_user_item, test_user_item

def read_item_features(dirpath, item_feature_file, sep='|'):
    entity_file = os.path.join(dirpath, 'entity2id.txt')
    item_mapping = {}
    reverse_item_mapping = {}
    features = {}
    reverse_features = {}
    item_feature = {}
    count_features = 0

    with open(entity_file,'r') as fr:
        count = int(fr.readline())
        line = fr.readline().split('\t')
        while line and line != ['']:
            ### movie for movielens , Anime for anime
            if line[0][:5] == 'movie':
                m_o_id = int(line[0][6:])
                m_id = int(line[1])
                item_mapping[m_id] = m_o_id
                reverse_item_mapping[m_o_id] = m_id
            line = fr.readline().split('\t')
    
    count_item = 1
    with open(item_feature_file, 'r') as fr:
        ## | for movielens , for anime
        line = fr.readline().split(sep)
        # print(line)
        while line and line != ['']:
            item = line[0]
            # print(item)
            fs = [e for e in line[1:] if e!='' and e!='\n']
            # print(fs)
            i_o_id = int(item)
            if i_o_id not in reverse_item_mapping:
                print('item {} is not in the list.'.format(item))
                count_item += 1
                line = fr.readline().split(sep)
                continue
            for f in fs:
                if f not in reverse_features:
                    reverse_features[f] = count_features
                    features[count_features] = f
                    count_features += 1
                i_id = reverse_item_mapping[i_o_id]
                item_feature.setdefault(i_id,[]).append(reverse_features[f])
            count_item += 1
            line = fr.readline().split(sep)
    return item_feature, features, reverse_features,item_mapping, reverse_item_mapping


if __name__=='__main__': 
    """item_feature_file ='data/movie_features.csv'
    # item_feature_file ='data/anime_features.csv'
    with open(item_feature_file,'r') as fr:
        csv_item_feature = csv.reader(fr,delimiter=',')
        item_features = [list(row) for row in csv_item_feature]
        print('Succesfully read item features from file.')
    # print('Movie features:{}'.format(item_features))
   
    
    anime_rating = 'data/u.data'
    # anime_rating = 'data/anime_rating.csv'
    with open(anime_rating, 'r') as fr:
        csv_anime_rating = csv.reader(fr,delimiter='\t')
        # csv_anime_rating = csv.reader(fr,delimiter=',')
        next(csv_anime_rating)
        rating_entries = [[row[0], row[1], int(row[2])] for row in csv_anime_rating]
        print('Successfully read movie ratings from file.')
        # print('User rating samples:{}'.format(rating_entries[:10]))
    """ 
    
    """
    # extract item features for feature map and train/test item lists
    fi = feature_item(item_features)
    reverse_item_dict = fi.reverse_items
    # print('reverse_item_dict: {}'.format(reverse_item_dict))
    item_features = fi.get_item_features()
    # print('item_features key: {} value: {}'.format(148, item_features[148]))
    ui = user_items(rating_entries, reverse_item_dict)
    candid_items, ui_weights = ui.user_items, ui.ui_weights
    # print('candid_items: {}'.format(candid_items))
    # print('ui_weights: {}'.format(ui_weights))
    """
    ### 'data/movie_features.csv'
    item_feature_file = 'data/movie_features.csv'
    ### 'data/ml100k_imdb_sep3/', 'data/anime_640_800/'
    dirpath = 'data/ml100k_imdb_sep5/'
    ### 'data/original/ml-100k/u.data', 'data/anime_rating.csv'
    o_u_i_file = 'data/ml-100k/u.data'
    candid_items,valid_items,test_items = get_train_valid_test_items(dirpath)
    # print(test_items)
    item_features,features,reverse_features,item_mapping,reverse_item_mapping = read_item_features(dirpath, item_feature_file) 
    print(item_features)
    print(features)
    # print(item_mapping)
    ui_weights = get_user_item_weight(candid_items,dirpath,o_u_i_file)
    
    
    itemlists = gen_itemlist_random_y(candid_items,ui_weights,item_features,features,div_type='cc',delta=4)
    print('Finsh generating itemlists.')
    # break
    with open(os.path.join(dirpath,'itemlists_cc_100_y_8_2.json'), 'w') as fw:
        json.dump(itemlists, fw)
    with open(os.path.join(dirpath, 'train_items.json'), 'w') as fw:
        json.dump(candid_items, fw)
    with open(os.path.join(dirpath, 'valid_items.json'), 'w') as fw:
        json.dump(valid_items,fw)
    with open(os.path.join(dirpath, 'test_items.json'),'w') as fw:
        json.dump(test_items, fw)
    with open(os.path.join(dirpath, 'item_genre.json'), 'w') as fw:
        json.dump(item_features, fw)
    count = 0
    for i in itemlists:
        count += len(itemlists[i])
    print('{}'.format(count))
    
    """
    itemlists = None
    with open(os.path.join(dirpath,'itemlists_dCC_100.json'),'r') as fr:
        itemlists = json.load(fr)
    for i in itemlists:
        l = itemlists[i]
        for ll in l:
            print(ll)
            print(get_diversity(ll,item_features,div_type='entropy'), get_diversity(ll,item_features))
    """
