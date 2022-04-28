import torch
import torch.optim as optim
from torch.autograd import Variable
import os
import ctypes
import json
import sys
import copy
import numpy as np

def to_var(x, use_gpu):
    if use_gpu:
        return Variable(torch.from_numpy(x).cuda())
    else:
        return Variable(torch.from_numpy(x))

class Config(object):
    def __init__(self):
        # to modify the base file location
        base_file  = '/home/lgan/divrecom/release/Base.so'
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
        # test triple classification
        # test recommendation
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
        # res type
        self.lib.getValidHit10.restype = ctypes.c_float
        ## parameters for training
        self.in_path = './'
        self.batch_size = 100
        self.bern = 0
        self.work_threads = 8
        self.hidden_dim = 75
        
        # num of negatives to one correct triple
        self.negative_ent = 1
        self.negative_rel = 0

        self.margin = 1.0
        self.valid_steps = 10
        self.save_steps = 10
        self.opt_method ='SGD'
        self.optimizer = None
        self.lr_decay = 0
        self.weight_decay = 0
        self.lmbda = 0
        self.alpha = 0
        self.early_stopping_patience = 10
        self.nbatches = 100
        self.p_norm = 2
        self.test_triple = True
        self.test_recom = False
        self.model = None
        self.trainModel = None
        self.testModel = None
        self.pretrain_model = None
        self.use_gpu = True
        self.item_embedding_dir = '/home/lu/divrecom/data/itemEmbedding_res/'
        self.item_embedding_filename = 'item_div.json'

    def init(self):
        # set InPath
        self.lib.setInPath(
                ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2)
                )
        # set OutPath
        self.lib.setOutPath(
                ctypes.create_string_buffer(self.result_dir.encode(), len(self.result_dir) * 2)
                )
        self.lib.setBern(self.bern)
        self.lib.setWorkThreads(self.work_threads)
        self.lib.randReset()
        
        # add import user-item interaction train files
        # self.lib.importUITrainFiles()

        self.lib.importTrainFiles()
        # test file only contains user-item relation triples
        self.lib.importTestFiles()
        self.lib.importTypeFiles()
        self.relTotal = self.lib.getRelationTotal()
        self.entTotal = self.lib.getEntityTotal()
        self.trainTotal = self.lib.getTrainTotal()
        self.testTotal = self.lib.getTestTotal()
        self.validTotal = self.lib.getValidTotal()
        # add user, item and trainUi counts 
        self.userTotal = self.lib.getUserTotal()
        self.itemTotal = self.lib.getItemTotal()
        # self.trainUiTotal = self.lib.getTrainUiTotal()

        # modify batch_size into two parts: ui and er
        # self.batch_size_ui = int(self.trainUiTotal /self.nbatches)
        # self.batch_size_er = int(self.trainTotal/self.nbatches) - self.batch_size_ui
        self.batch_size = int(self.trainTotal / self.nbatches)
        
        self.batch_seq_size = self.batch_size * (1 + self.negative_ent + self.negative_rel)
        # add separate ui and er parts
        # self.batch_seq_size_ui = self.batch_size_ui * (1 + self.negative_ent + self.negative_rel)
        # self.batch_seq_size_er = self.batch_size_er * (1 + self.negative_ent + self.negative_rel)

        # add user and item batches
        # self.batch_u = np.zeros(self.batch_seq_size_ui, dtype=np.int64)
        # self.batch_i = np.zeros(self.batch_seq_size_ui, dtype=np.int64)
        self.batch_h = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_t = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_r = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_y = np.zeros(self.batch_seq_size, dtype=np.float32)
        
        # self.batch_u_addr = self.batch_u.__array_interface__["data"][0]
        # self.batch_i_addr = self.batch_i.__array_interface__["data"][0]

        self.batch_h_addr = self.batch_h.__array_interface__["data"][0]
        self.batch_t_addr = self.batch_t.__array_interface__["data"][0]
        self.batch_r_addr = self.batch_r.__array_interface__["data"][0]
        self.batch_y_addr = self.batch_y.__array_interface__["data"][0]

        self.valid_h = np.zeros(self.entTotal, dtype=np.int64)
        self.valid_t = np.zeros(self.entTotal, dtype=np.int64)
        self.valid_r = np.zeros(self.entTotal, dtype=np.int64)
        self.valid_h_addr = self.valid_h.__array_interface__["data"][0]
        self.valid_t_addr = self.valid_t.__array_interface__["data"][0]
        self.valid_r_addr = self.valid_r.__array_interface__["data"][0]

        self.test_h = np.zeros(self.entTotal, dtype=np.int64)
        self.test_t = np.zeros(self.entTotal, dtype=np.int64)
        self.test_r = np.zeros(self.entTotal, dtype=np.int64)
        self.test_h_addr = self.test_h.__array_interface__["data"][0]
        self.test_t_addr = self.test_t.__array_interface__["data"][0]
        self.test_r_addr = self.test_r.__array_interface__["data"][0]

        self.valid_pos_h = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_pos_t = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_pos_r = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_pos_h_addr = self.valid_pos_h.__array_interface__["data"][0]
        self.valid_pos_t_addr = self.valid_pos_t.__array_interface__["data"][0]
        self.valid_pos_r_addr = self.valid_pos_r.__array_interface__["data"][0]

        self.valid_neg_h = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_neg_t = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_neg_r = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_neg_h_addr = self.valid_neg_h.__array_interface__["data"][0]
        self.valid_neg_t_addr = self.valid_neg_t.__array_interface__["data"][0]
        self.valid_neg_r_addr = self.valid_neg_r.__array_interface__["data"][0]

        self.test_pos_h = np.zeros(self.testTotal, dtype=np.int64)
        self.test_pos_t = np.zeros(self.testTotal, dtype=np.int64)
        self.test_pos_r = np.zeros(self.testTotal, dtype=np.int64)
        self.test_pos_h_addr = self.test_pos_h.__array_interface__["data"][0]
        self.test_pos_t_addr = self.test_pos_t.__array_interface__["data"][0]
        self.test_pos_r_addr = self.test_pos_r.__array_interface__["data"][0]

        self.test_neg_h = np.zeros(self.testTotal, dtype=np.int64)
        self.test_neg_t = np.zeros(self.testTotal, dtype=np.int64)
        self.test_neg_r = np.zeros(self.testTotal, dtype=np.int64)
        self.test_neg_h_addr = self.test_neg_h.__array_interface__["data"][0]
        self.test_neg_t_addr = self.test_neg_t.__array_interface__["data"][0]
        self.test_neg_r_addr = self.test_neg_r.__array_interface__["data"][0]
        self.relThresh = np.zeros(self.relTotal, dtype=np.float32)
        self.relThresh_addr =self.relThresh.__array_interface__["data"][0]

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_test_recom(self,test_recom):
        self.test_recom = test_recom

    
    def set_test_triple(self,test_triple):
        self.test_triple = test_triple

    def set_margin(self, margin):
        self.margin = margin

    def set_in_path(self, inpath):
        self.in_path = inpath

    def set_nbatches(self, nbatches):
        self.nbatches = nbatches

    def set_p_norm(self, p_norm):
        self.p_norm = p_norm

    def set_valid_steps(self, valid_steps):
        self.valid_steps = valid_steps

    def set_save_steps(self, save_steps):
        self.save_steps = save_steps

    def set_checkpoint_dir(self,checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def set_result_dir(self, result_dir):
        self.result_dir = result_dir

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_lambda(self,lmbda):
        self.lmbda = lmbda

    def set_lr_decay(self,lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self,weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_bern(self, bern):
        self.bern = bern

    def set_dimension(self, dim):
        self.hidden_size = dim
        self.ent_size = dim
        self.rel_size = dim

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_work_threads(self,work_threads):
        self.work_threads = work_threads

    def set_ent_neg_rate(self,rate):
        self.negative_ent = rate

    def set_rel_neg_rate(self,rate):
        self.negative_rel = rate

    def set_early_stopping_patience(self, early_stopping_patience):
        self.early_stopping_patience = early_stopping_patience
    
    def set_pretrain_model(self,pretrain_model):
        self.pretrain_model = pretrain_model

    def set_item_weight(self, filename):
        path = filename
        item_weight = torch.load(path)
        print('item weight: {}'.format(item_weight))
        self.item_weight = item_weight

    def get_parameters(self,param_dict,mode='numpy'):
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

    def save_embedding_matrix(self,best_model):
        path = os.path.join(self.result_dir,self.model.__name__ + '.json')
        with open(path, 'w') as fw:
            fw.write(json.dumps(self.get_parameters(best_model,'list')))
    
    def set_train_model(self, model):
        print('Initializing training model...')
        self.model = model
        self.trainModel = self.model(config=self)
        if self.use_gpu:
            self.trainModel.cuda()
        if self.optimizer != None:
            pass
        elif  self.opt_method == 'Adam' or self.opt_method == 'adam':
            self.optimizer = optim.Adam(
                    self.trainModel.parameters(),
                    lr = self.alpha,
                    weight_decay = self.weight_decay,
                )
        else:
            self.optimizer = optim.SGD(
                    self.trainModel.parameters(),
                    lr = self.alpha,
                    weight_decay = self.weight_decay,
            )
        print('Finish initializing')

    def set_test_model(self,model,path=None):
        print('Initializing test model ...')
        self.model = model
        self.testModel = self.model(config=self)
        if path == None:
            path = os.path.join(self.result_dir,self.model.__name__ +'.ckpt')
        self.testModel.load_state_dict(torch.load(path))
        if self.use_gpu:
            self.testModel.cuda()
        self.testModel.eval()
        print('Finish initializing')

    def sampling(self):
        # to modify
        self.lib.sampling(
                # add batch_u_addr, batch_i_addr
                # self.batch_u_addr,
                # self.batch_i_addr,
                self.batch_h_addr,
                self.batch_t_addr,
                self.batch_r_addr,
                self.batch_y_addr,
                # add batch_size_ui
                # self.batch_size_ui,
                self.batch_size,
                self.negative_ent,
                self.negative_rel,
                )

    def save_checkpoint(self,model,epoch):
        path = os.path.join(
                self.checkpoint_dir,self.model.__name__+ '-'+ str(epoch) + '.ckpt')
        torch.save(model,path)

    def save_best_checkpoint(self,best_model):
        path = os.path.join(self.result_dir,self.model.__name__+'.ckpt')
        torch.save(best_model,path)

    def train_one_step(self):
        ## add batch_u , batch_i
        # print('batch_u: {}'.format(self.batch_u.to_numpy()))
        # self.trainModel.batch_user = to_var(self.batch_u, self.use_gpu)
        # self.trainModel.batch_item = to_var(self.batch_i, self.use_gpu)

        self.trainModel.batch_h = to_var(self.batch_h, self.use_gpu)
        self.trainModel.batch_t = to_var(self.batch_t, self.use_gpu)
        self.trainModel.batch_r = to_var(self.batch_r, self.use_gpu)
        self.trainModel.batch_y = to_var(self.batch_y, self.use_gpu)
        self.optimizer.zero_grad()
        loss = self.trainModel()
        loss.backward()
        self.optimizer.step()
        return loss.item()
        pass

    def test_one_step(self,model,test_u,test_i,test_r):
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
                    self.valid_h_addr,self.valid_t_addr,self.valid_r_addr)
            res = self.test_one_step(model,self.valid_h,self.valid_t,self.valid_r)
            self.lib.validHead(res.__array_interface__['data'][0])

            self.lib.getValidTailBatch(
                    self.valid_h_addr,self.valid_t_addr,self.valid_r_addr)
            res = self.test_one_step(model,self.valid_h,self.valid_t,self.valid_r)
            self.lib.validTail(res.__array_interface__['data'][0])
        return self.lib.getValidHit10()

    def train(self):
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        best_epoch = 0
        best_hit10 = 0.0
        best_model = None
        bad_counts = 0
        for epoch in range(int(self.train_times)):
            total_loss = 0.0
            for batch in range(int(self.nbatches)):
                self.sampling()
                loss = self.train_one_step()
                total_loss += loss
            print('Epoch {} | loss: {}'.format(epoch,total_loss))
            if (epoch + 1) % self.save_steps == 0:
                print('Epoch {} has finished, saving...'.format(epoch))
                self.save_checkpoint(self.trainModel.state_dict(),epoch)
            if (epoch + 1) % self.valid_steps == 0:
                print('Epoch {} has finished, validating ...'.format(epoch))
                hit10 = self.valid(self.trainModel)
                if hit10 > best_hit10:
                    best_hit10 = hit10
                    best_epoch = epoch
                    best_model = copy.deepcopy(self.trainModel.state_dict())
                    print('Best model | hit@10 of valid set is {}'.format(best_hit10))
                    bad_counts = 0
                else:
                    print('Hit@10 of valid set is {} | bad count is {}'.format(hit10,bad_counts))
                    bad_counts += 1
                if bad_counts == self.early_stopping_patience:
                    print('Early stopping at epoch {}'.format(epoch))
                    break
        if best_model == None:
            best_model = self.trainModel.state_dict()
            best_epoch = self.train_times - 1
            best_hit10 = self.valid(self.trainModel)
        print('Best epoch is {} | hit@10 of valid set is {}'.format(best_epoch,best_hit10))
        print('Store checkpoint of best result at epoch {}'.format(best_epoch))
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)
        self.save_best_checkpoint(best_model)
        self.save_embedding_matrix(best_model)
        print('Finish storing')
        print('Testing ...')
        self.set_test_model(self.model)
        self.test()
        print('Finish test')
        return best_model

    def triple_classification(self):
        self.lib.getValidBatch(
            self.valid_pos_h_addr,
            self.valid_pos_t_addr,
            self.valid_pos_r_addr,
            self.valid_neg_h_addr,
            self.valid_neg_t_addr,
            self.valid_neg_r_addr,
        )
        res_pos = self.test_one_step(
            self.testModel, self.valid_pos_h, self.valid_pos_t, self.valid_pos_r
        )
        res_neg = self.test_one_step(
            self.testModel, self.valid_neg_h, self.valid_neg_t, self.valid_neg_r
        )
        self.lib.getBestThreshold(
            self.relThresh_addr,
            res_pos.__array_interface__["data"][0],
            res_neg.__array_interface__["data"][0],
        )

        self.lib.getTestBatch(
            self.test_pos_h_addr,
            self.test_pos_t_addr,
            self.test_pos_r_addr,
            self.test_neg_h_addr,
            self.test_neg_t_addr,
            self.test_neg_r_addr,
        )
        res_pos = self.test_one_step(
            self.testModel, self.test_pos_h, self.test_pos_t, self.test_pos_r
        )
        res_neg = self.test_one_step(
            self.testModel, self.test_neg_h, self.test_neg_t, self.test_neg_r
        )
        self.lib.test_triple_classification(
            self.relThresh_addr,
            res_pos.__array_interface__["data"][0],
            res_neg.__array_interface__["data"][0],
        )
        pass

    def item_recommendation(self):
        print("The total of test triple is %d" % (self.testTotal))
        for i in range(self.testTotal):
            sys.stdout.write("%d\r" % (i))
            sys.stdout.flush()
            self.lib.getHeadBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
            #print("Test_r size: {}".format(self.test_r))
            # print("sign 1")
            res = self.test_one_step(
                self.testModel, self.test_h, self.test_t, self.test_r
            )
            # print("res length : {}".format(res))
            # print("sign 2") 
            self.lib.testHead(res.__array_interface__["data"][0])
            # print("sign 3")
            self.lib.getTailBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
            res = self.test_one_step(
                self.testModel, self.test_h, self.test_t, self.test_r
            )
            self.lib.testTail(res.__array_interface__["data"][0])
        self.lib.test_recommendation()
        self.evaluate_recom()
        pass 

    def evaluate_recom(self):
        pass

    def get_neglected_item_in_training(self, u):
        pass

    def test(self):
        if self.test_triple:
            self.triple_classification()
            # add renaming
            if os.path.exists(os.path.join(self.result_dir, 'test_triple.txt')):
                os.rename(os.path.join(self.result_dir,"test_triple.txt"),
                 os.path.join(self.result_dir,"{}_{}_{}_{}_{}_{}_test_triple.txt".format(self.model.__name__,
                self.margin, self.ent_size,self.batch_size,self.bern, self.alpha)))

        if self.test_recom:
            self.item_recommendation()
            # add renaming
            print('Renaming...')
            if os.path.exists(os.path.join(self.result_dir,  self.model.__name__ + ".json")):
                os.rename(os.path.join(self.result_dir,  self.model.__name__ + ".json"),
                os.path.join(self.result_dir,'{}_{}_{}_{}_{}_{}_{}.json'.format(self.model.__name__,
                self.margin, self.ent_size,self.batch_size, self.bern, self.alpha,self.weight_decay)))
            if os.path.exists(os.path.join(self.result_dir,'no_constraint_test_link.txt')):
                os.rename(os.path.join(self.result_dir,"no_constraint_test_link.txt"),
                os.path.join(self.result_dir,"{}_{}_{}_{}_{}_{}_{}_no_constraint_test_link.txt".format(self.model.__name__,
                self.margin, self.ent_size,self.batch_size,self.bern, self.alpha,self.weight_decay)))
                os.rename(os.path.join(self.result_dir,"type_constraint_test_link.txt"),
                os.path.join(self.result_dir, "{}_{}_{}_{}_{}_{}_{}_type_constraint_test_link.txt".format(self.model.__name__,
                self.margin, self.ent_size,self.batch_size,self.bern, self.alpha,self.weight_decay)))

