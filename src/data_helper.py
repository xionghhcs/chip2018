import pandas as pd
import numpy as np


def load_embed(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        lines = [l.strip('\n') for l in lines]
    word_nums = len(lines)
    embed_size = len(lines[0].split()) - 1

    vocabulary = dict()
    vocabulary['<padding>'] = 0
    vocabulary['<unk>'] = 1

    embedding = np.zeros(shape=(word_nums + len(vocabulary), embed_size), dtype='float')
    embedding[0] = np.zeros(embed_size, dtype='float')
    embedding[1] = np.random.random(embed_size)

    idx = len(vocabulary)
    for line in lines:
        line = line.split()
        word = line[0]
        vector = np.array(line[1:])
        embedding[idx] = vector
        vocabulary[word] = idx

        idx += 1
    return vocabulary, embedding


def word2ids(q_list, vocabulary):
    result = []
    for q in q_list:
        new_q = []
        for w in q:
            if w in vocabulary:
                new_q.append(vocabulary[w])
            else:
                new_q.append(vocabulary['<unk>'])
        result.append(new_q)
    return result


def change_format(in_file, out_file):
    """
    将官方给的词向量文件中的\t全都替换为空格，方便gensim加载。
    :param in_file:
    :param out_file:
    :return:
    """
    with open(in_file, 'r') as f:
        lines = f.readlines()
        lines = [l.strip('\n') for l in lines]
        lines = [l.replace('\t', ' ') for l in lines]
    with open(out_file, 'w') as f:
        f.write('{} {}\n'.format(len(lines), 300))
        f.write('\n'.join(lines))


if __name__ == '__main__':
    # change_format(in_file='../data/char_embedding.txt', out_file='../data/char_embed_v1.txt')
    # change_format(in_file='../data/word_embedding.txt', out_file='../data/word_embed_v1.txt')
    pass
