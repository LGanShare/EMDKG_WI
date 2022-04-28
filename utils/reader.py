import numpy as np
import csv

class user_items(object):
    def __init__(self, rating, reverse_item_dict,sign=False):
        self.rating = rating
        self.reverse_item_dict = reverse_item_dict
        self.sign = sign
    
        self.user_dict, self.reverse_user_dict = self.get_users()
        # print('user dict:{}'.format(self.user_dict))
        self.user_items, self.ui_weights = self.get_user_items()
        

    def get_users(self):
        if len(self.rating) == 0:
            print('Please provide a valid rating file.')
            return
        count = 0
        user_dict = {}
        reverse_user_dict = {}
        for entry in self.rating:
            # print('entry in rating: {}'.format(entry))
            u = entry[0]
            # print('User original id: {}'.format(u))
            if u not in reverse_user_dict:
                # print('enter the if clause for {}'.format(u))
                user_dict[count] = u
                reverse_user_dict[u] = count
                count += 1
                # print('User original id: {}, new id: {}'.format(u, count))
        return user_dict, reverse_user_dict

    def get_user_items(self):
        if len(self.rating) == 0:
            print('Please provide a valid rating file.')
            return
        user_items = {}
        ui_weights = {}
        # to do
        for entry in self.rating:
            # print('entry in function get_user_item: {}'.format(entry))
            uo = entry[0]
            u = self.reverse_user_dict[uo]
            try:
                io = entry[1]
                if self.sign == True:
                    i = self.reverse_item_dict[io]
                else:
                    i = io
            except KeyError as e:
                # print('Item {} does not exist in item feature file. Just ignore it.'.format(io))
                continue
            score = int(entry[2])
            if score > 0:
                user_items.setdefault(u,[]).append(i)
            ui_weights.setdefault(u,{}).setdefault(i,score)
        return user_items, ui_weights
