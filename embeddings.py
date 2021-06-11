import numpy as np
import matplotlib.pyplot as plt

def plot_with_labels(low_dim_embs, colors, labels, filename):   # 绘制词向量图
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(20, 20))  # in inches
    colors = ['red', 'blue', 'green', 'black']
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y, c=colors[i])
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)

def read(embedding_name, threshold=0, dtype='float'):
    embedding_file = open(embedding_name, 'r', encoding='utf-8', errors='surrogateescape')
    print('读取词向量文件中......')
    header = embedding_file.readline().split(' ')
    count = int(header[0]) if threshold <= 0 else min(threshold, int(header[0]))
    dim = int(header[1])
    words = []
    matrix = np.empty((count, dim), dtype=dtype)
    for i in range(count):
        word, vec = embedding_file.readline().split(' ', 1)
        words.append(word)
        matrix[i] = np.fromstring(vec, sep=' ', dtype=dtype)
    embedding_file.close()
    return (words, matrix)
