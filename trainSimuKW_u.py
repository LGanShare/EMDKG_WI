import torch
from config import SimuConfigU
from kw_models import TransE, TransH, TransD
from ItemDivLearning import DivReprU, EngineU, feature_item_u
from trainItemDivEmb_u import get_train_valid_test_items
import csv
import json
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES']= '0'

kw_method = str(sys.argv[1])
print(kw_method)
margin = float(sys.argv[2])
k = int(sys.argv[3])
nbatches = int(sys.argv[4])
learning_rate = float(sys.argv[5])
bern = int(sys.argv[6])
weight_decay = float(sys.argv[7])

print("Embedding method: {} \t margin: {} \t dimension: {} \t number of batches: {} \t learning rate: {} \t bern: {}\n".format(kw_method,margin, k, nbatches, learning_rate, bern))

dirpath = 'data/ml100k_imdb_sep5/'
    
item_mapping_file = '{}/entity2id.txt'.format(dirpath)
item_mapping = {}
reverse_item_mapping = {}
with open(item_mapping_file,'r') as fr:
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


item_feature_file ='data/movie_features_divlearning.csv'
with open(item_feature_file,'r') as fr:
    csv_item_feature = csv.reader(fr,delimiter='|')
    item_features = [list(row) for row in csv_item_feature]
    print('Succesfully read item features from file.')
# fi = feature_item_u(item_features)
# train_itemlists, valid_itemlists,test_itemlists = fi.gen_itempairs(sample_size=200,prop=0.8,t='dCC',n=4)
itemlists_file = 'data/ml100k_imdb_sep5/itemlists_alpha-ndcg_100.json'
with open(itemlists_file, 'r') as fr:
    itemlists = json.load(fr)
    itemlists = {int(u):itemlists[u] for u in itemlists}
# extract item features for feature map and train/test item lists
fi = feature_item_u(item_features)
train_itemlists, valid_itemlists, test_itemlists = fi.get_itemlists(itemlists, item_mapping)

pos_items, valid_items, test_items = get_train_valid_test_items(dirpath)
pos_items = [[item_mapping[e] for e in pos_items[u]] for u in pos_items]


item_div_set = fi.item_div_set
full_itemlist = list(range(len(fi.items)))
feature_map = fi.gen_feature_map(full_itemlist)
print('Finish generating feature map.')

train_itemlists_with_div_val = fi.gen_div_val(train_itemlists, feature_map)
valid_itemlists_with_div_val = fi.gen_div_val(valid_itemlists, feature_map)
test_itemlists_with_div_val = fi.gen_div_val(test_itemlists, feature_map)

con = SimuConfigU()
con.set_in_path("./data/ml100k_imdb_sep5/")
# con.set_item_weight("./data/itemEmbedding_res/DivItem.ckpt")
con.set_work_threads(8)
con.set_train_times(400)
con.set_fi(fi)
con.set_pos_items(pos_items)
con.set_nbatches(nbatches)
con.set_alpha(learning_rate)
con.set_bern(bern)
con.set_dimension(k)
con.set_weight_decay(weight_decay)
con.set_margin(margin)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("adam")
con.set_save_steps(10)
con.set_valid_steps(10)
con.set_early_stopping_patience(5)
con.set_checkpoint_dir("./checkpoint")
con.set_result_dir("./data/embeddings/simu_ml100k_imdb_sep5_u_alpha-ndcg_100/")
con.set_test_recom(True)
con.set_test_triple(False)
con.set_item_div_set(item_div_set)
con.init()

# con.set_train_model(TransH, DivRepr)
if kw_method == 'TransE':
    con.set_train_model(TransE, DivReprU)
elif kw_method == 'TransH':
    con.set_train_model(TransH,DivReprU)

con.train(train_itemlists_with_div_val,valid_itemlists_with_div_val)
