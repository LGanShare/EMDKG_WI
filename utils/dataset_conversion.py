import numpy as np
import os
import csv
import random
import copy

def to_format_rcf(dirpath, user_count):
    test_file = os.path.join(dirpath, 'test2id.txt')
    train_file = os.path.join(dirpath, 'train2id.txt')
    valid_file = os.path.join(dirpath, 'valid2id.txt')

    num_user = user_count


    new_train = []
    aux_relations_genre = {}
    aux_relations_director = {}
    aux_relations_actor = {}

    # count number of different relation tuples
    rel_genre_num = 0
    rel_director_num = 0
    rel_actor_num = 0
    genres = {}

    # entity mapping
    genre_mapping = {}
    director_mapping = {}
    actor_mapping = {}

    with open(train_file, 'r') as fr:
        num = int(fr.readline())
        line = fr.readline()
        line = line.split('\t')
        while line and line != ['']:
            # print(line)
            # line = [int(e) for e in line]
            # print(line)
            if line[2] == '0':
                u = line[0]
                i = str(int(line[1])-num_user)
                new_train.append((u, i))
                # print('User: {}\t Movie: {}'.format(u,i))
            elif line[2] == '1':
                i = str(int(line[0]) -num_user)
                # print('i:{}\t line[1]:{}'.format(i,line[1]))
                if line[1] not in genres:
                    genres[line[1]] = rel_genre_num
                    rel_genre_num += 1
                # print(genres)
                # genre_mapping[i] =  genres[line[1]]
                # print('genre_mapping[{}]: {}'.format(i, genres[line[1]]))
                aux_relations_genre.setdefault(i, []).append(str(genres[line[1]]))
                # print('Movie: {}\t Genre: {}'.format(line[0], line[1]))
            elif line[2] == '3':
                i = str(int(line[1]) - num_user)
                if line[0] not in director_mapping:
                    director_mapping[line[0]] = str(rel_director_num)
                    rel_director_num += 1
                aux_relations_director.setdefault(i, []).append(director_mapping[line[0]])
                # print('Movie: {}\t Director: {}'.format(line[1], line[0]))
            elif line[2] == '8' or line[2] == '9':
                i = str(int(line[1]) - num_user)
                if line[0] not in actor_mapping:
                    actor_mapping[line[0]] = str(rel_actor_num)
                    rel_actor_num += 1
                aux_relations_actor.setdefault(i, []).append(actor_mapping[line[0]])
                # print('Movie: {}\t Actor: {}'.format(line[1], line[0]))
            line = fr.readline().strip().split('\t')
    # print('Genre mapping: {}'.format(genres))
    # print('Director_mapping: {}'.format(director_mapping))
    # print('Actor mapping: {}'.format(actor_mapping))

    new_test = []
    with open(test_file, 'r') as fr:
        num = int(fr.readline())
        line = fr.readline().strip().split('\t')
        while line and line != ['']:
            line = [int(e) for e in line]
            u = line[0]
            i = str(int(line[1]) - num_user)
            new_test.append((u, i))
            line = fr.readline().strip().split('\t')

    new_valid = []
    with open(valid_file, 'r') as fr:
        num = int(fr.readline())
        line = fr.readline().strip().split('\t')
        while line and line != ['']:
            line = [int(e) for e in line]
            u = line[0]
            i = str(int(line[1]) - num_user)
            new_valid.append((u, i))
            line = fr.readline().strip().split('\t')

    with open(os.path.join(dirpath, 'train.txt'), 'w') as fw:
        for item in new_train:
            fw.write(str(item[0]) + '\t' + str(item[1]) + '\n')
    with open(os.path.join(dirpath, 'test.txt'), 'w') as fw:
        for item in new_test:
            fw.write(str(item[0]) + '\t' + str(item[1]) + '\n')
    with open(os.path.join(dirpath, 'valid.txt'), 'w') as fw:
        for item in new_valid:
            fw.write(str(item[0]) + '\t' + str(item[1]) + '\n')
    with open(os.path.join(dirpath, 'auxiliary-mapping.txt'), 'w') as fw:
        visited = []
        for m in aux_relations_genre:
            visited.append(m)
            # print('User {}'.format(u))
            fw.write(str(m) + '|'+ ','.join(aux_relations_genre[m]) + '|')
            # print('Movie\t ' + str(m) + '|'+'Genre: \t'+ ','.join(aux_relations_genre[m]) + '|')
            if m in aux_relations_director:
                fw.write(','.join(aux_relations_director[m]) + '|')
                # print('Directors: \t' + ','.join(aux_relations_director[m]) + '|')
            else:
                fw.write('|')
                # print('Directors:\t' + 'None')
            if m in aux_relations_actor:
                fw.write(','.join(aux_relations_actor[m]) + '\n')
                # print('Actors:\t ' + ','.join(aux_relations_actor[m]) + '\n')
            else:
                fw.write('|\n')
                # print('Actors:\t ' + 'None' + '\n')
        for u in aux_relations_director:
            if u in visited:
                continue
            else:
                # print('User {}'.format(u))
                visited.append(u)
                fw.write(str(u) + '|' + '|')
                if u in aux_relations_director:
                    fw.write(','.join(aux_relations_director[u]) + '|')
                else:
                    fw.write('|')
                if u in aux_relations_actor:
                    fw.write(','.join(aux_relations_actor[u]) + '\n')
                else:
                    fw.write('|\n')
        for u in aux_relations_actor:
            if u in visited:
                continue
            else:
                # print('User {}'.format(u))
                visited.append(u)
                fw.write(str(u) + '|' + '|' + '|')
                if u in aux_relations_actor:
                    fw.write(','.join(aux_relations_actor[u]) + '\n')
                else:
                    fw.write('|\n')

def to_format_pmf(dirpath):
    test_file = os.path.join(dirpath, 'test2id.txt')
    train_file = os.path.join(dirpath, 'train2id.txt')
    valid_file = os.path.join(dirpath, 'valid2id.txt')

    user_mapping = {}
    movie_mapping = {}
    with open(os.path.join(dirpath, 'entity2id.txt'), 'r') as fr:
        entity_num = fr.readline()
        line = fr.readline().split('\t')
        while line and line!=['']:
            if line[0][:5] == 'user_':
                user_mapping[int(line[1])] = int(line[0][5:])
            elif line[0][:6] == 'movie_':
                movie_mapping[int(line[1])] = int(line[0][6:])
            else:
                break
            line = fr.readline().split('\t')


    ratings = {}
    with open('ml-100k/u.data') as fr:
    #    line =
        reader = csv.reader(fr, delimiter='\t')
        # first_row = next(reader)
        ratings = {(int(rows[0]),int(rows[1])):int(rows[2]) for rows in reader}
    res_test = []
    with open(test_file, 'r') as fr:
        test_num = int(fr.readline())
        line = fr.readline().split('\t')
        while line and line!= ['']:
            # print(line)
            uid = int(line[0])
            mid = int(line[1])
            if (user_mapping[uid],movie_mapping[mid]) in ratings:
                res_test.append([str(uid), str(mid),str(ratings[(user_mapping[uid],movie_mapping[mid])])])
            line = fr.readline().split('\t')

    with open(os.path.join(dirpath, 'test_pmf.txt'), 'w') as fw:
        for t in res_test:
            fw.write('\t'.join(t) + '\n')
    #################

    res_train = []
    with open(train_file, 'r') as fr:
        train_num = int(fr.readline())
        line = fr.readline().split('\t')
        while line and line!= ['']:
            # print(line)
            if int(line[0]) > 942:
                break
            uid = int(line[0])
            mid = int(line[1])
            # print('{}:::::{}'.format(uid, mid))
            if (user_mapping[uid],movie_mapping[mid]) in ratings:
                res_train.append([str(uid), str(mid),str(ratings[(user_mapping[uid],movie_mapping[mid])])])
            line = fr.readline().split('\t')

    with open(os.path.join(dirpath, 'train_pmf.txt'), 'w') as fw:
        for t in res_train:
            fw.write('\t'.join(t) + '\n')
    ##########

    res_valid = []
    with open(test_file, 'r') as fr:
        valid_num = int(fr.readline())
        line = fr.readline().split('\t')
        while line and line!= ['']:
            # print(line)
            if int(line[0]) > 942:
                break
            uid = int(line[0])
            mid = int(line[1])
            if (user_mapping[uid],movie_mapping[mid]) in ratings:
                res_valid.append([str(uid), str(mid),str(ratings[(user_mapping[uid],movie_mapping[mid])])])
            line = fr.readline().split('\t')

    with open(os.path.join(dirpath, 'valid_pmf.txt'), 'w') as fw:
        for t in res_valid:
            fw.write('\t'.join(t) + '\n')

def to_format_kg_jkr(dirpath):
    e_map = {}
    with open(os.path.join(dirpath, 'entity2id.txt'), 'r') as fr:
        num_o = int(fr.readline())
        count = 0
        line = fr.readline().split('\t')
        while line and line != ['']:
            if line[0][:4] == 'user':
                num_o -= 1
                line = fr.readline().split('\t')
                continue
            else:
                e_map[int(line[1])] = count
                count += 1
                line = fr.readline().split('\t')

    r_map = {}
    r_list = []
    with open(os.path.join(dirpath, 'relation2id.txt'), 'r') as fr:
        num_r = int(fr.readline())
        count = 0
        line = fr.readline().split('\t')
        while line and line != ['']:
            if line[0] == 'rating':
                num_r -= 1
                line = fr.readline().split('\t')
                continue
            else:
                r_map[int(line[1])] = count
                count += 1
                line = fr.readline().split('\t')
    with open(os.path.join(dirpath, 'train2id.txt'), 'r') as fr:
        num_t = int(fr.readline())
        line = fr.readline().split('\t')
        while line and line != ['']:
            if int(line[2]) == 0:
                num_t -= 1
                line = fr.readline().split('\t')
                continue
            else:
                 h = e_map[int(line[0])]
                 t = e_map[int(line[1])]
                 r = r_map[int(line[2])]
                 r_list.append((h, r, t))
                 line = fr.readline().split('\t')
    random.shuffle(r_list)
    train_list = random.sample(r_list, int(num_t*0.8))
    valid_list = random.sample(r_list, int(num_t*0.1))
    test_list = random.sample(r_list, int(num_t*0.1))

    with open(os.path.join(dirpath, 'e_map.dat'), 'w') as fw:
        for o in e_map:
            fw.write('{}\t{}\n'.format(e_map[o], o))
    with open(os.path.join(dirpath, 'r_map.dat'), 'w') as fw:
        for o in r_map:
            fw.write('{}\t{}\n'.format(r_map[o], o))
    with open(os.path.join(dirpath, 'train.dat'), 'w') as fw:
        for t in train_list:
            fw.write('{}\t{}\t{}\n'.format(t[0], t[2], t[1]))
    with open(os.path.join(dirpath, 'valid.dat'), 'w') as fw:
        for t in valid_list:
            fw.write('{}\t{}\t{}\n'.format(t[0], t[2], t[1]))
    with open(os.path.join(dirpath, 'test.dat'), 'w') as fw:
        for t in test_list:
            fw.write('{}\t{}\t{}\n'.format(t[0], t[2], t[1]))



if __name__ == '__main__':
    to_format_rcf('../data/books_10_2021/',294543)
    # to_format_kg_jkr('./ml100k_imdb_sep2/')
    # to_format_pmf('./ml100k_imdb_sep3/')
