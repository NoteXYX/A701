import jieba
import numpy as np
from sklearn.cluster import KMeans
import operator
from embeddings import read
import argparse
from pylab import mpl
from sklearn.manifold import TSNE
from TSNE import plot_with_labels
import json
import pickle


class patent_ZH:
    def __init__(self, content, doc_num, ipc):
        self.label = -1
        self.content = content
        self.doc_num = doc_num
        self.docvec = None
        self.ipc = ipc

def get_kmeans_clusters(vectors,labels, dim=100):    # 根据Birch聚类后的标签labels整理各类的向量，存放在字典clusters
    clusters = dict()
    for i in range(len(labels)):
        if int(labels[i]) not in clusters:
            clusters[int(labels[i])] = vectors[i].reshape(1, dim).tolist()
        elif int(labels[i]) in clusters:
            cur_vec = vectors[i].reshape(dim).tolist()
            cur_cluster = clusters[int(labels[i])]
            cur_cluster.append(cur_vec)
            clusters[int(labels[i])] = cur_cluster
    clusters = dict(sorted(clusters.items(), key=operator.itemgetter(0)))
    return clusters

def get_centers(label_vecs, dim=100):  # 获得各个类的中心点(噪音类除外)
    centers = np.zeros((len(list(label_vecs.keys())), dim))
    for label in list(label_vecs.keys()):
        if label == -1:  # 如果是噪音类
            continue
        else:
            cur_vectors = np.array(label_vecs[label])
            cur_center = np.mean(cur_vectors, axis=0).reshape(1, dim)
            centers[label] = cur_center
    return centers

def get_label(patent_list,cluster):
    f_num = 0
    for label in cluster:
        cur_file = patent_list[f_num]
        cur_file.label = label
        f_num += 1
    return patent_list

def get_patent_ipc(patent_list):
    ipc_dict = dict()
    for patent in patent_list:
        if patent.label not in ipc_dict:
            ipc_dict[patent.label] = [patent.ipc]
        else:
            ipc_dict[patent.label].append(patent.ipc)
    ipc_dict = dict(sorted(ipc_dict.items(), key=operator.itemgetter(0)))
    return ipc_dict

def get_class_num(labels):
    class_num = dict()
    for label in labels:
        if label not in class_num:
            class_num[label] = 1
        else:
            class_num[label] += 1
    class_num = dict(sorted(class_num.items(), key=operator.itemgetter(0)))
    return class_num

def get_index2vectors(word2ind, wordvecs, line_words):    # 获得测试文本中所有词的词向量
    ind2vec = dict()
    for word in line_words:
        cur_index = word2ind[word]
        cur_vec = wordvecs[cur_index]
        ind2vec[cur_index] = cur_vec
    return ind2vec

def get_distance(cur_vector, cur_center, method):   # 获得与中心点的距离(余弦相似度 or 欧式距离)
    if method == 'cos':
        num = float(np.dot(cur_vector, cur_center.T))
        vec_norm = np.linalg.norm(cur_vector) * np.linalg.norm(cur_center)
        cos = num / vec_norm
        sim = 0.5 + 0.5 * cos   # 归一化
        return sim
    elif method == 'ED':
        dist = np.linalg.norm(cur_vector - cur_center)
        return dist

def distance_sort(ind2vec, cur_center, method):     # 获得根据与中心点距离大小排序后的{词向量序号：与中心点的距离}
    index_distance = dict()
    for index in ind2vec:
        distance = get_distance(ind2vec[index], cur_center, method)
        index_distance[index] = distance
    if method == 'cos':
        sorted_distance = sorted(index_distance.items(), key=operator.itemgetter(1), reverse=True)
    else:
        sorted_distance = sorted(index_distance.items(), key=operator.itemgetter(1))
    sorted_index_distance = dict(sorted_distance)
    return sorted_index_distance

def get_stopwords(fname):
    stop_file = open(fname, 'r', encoding='utf-8')
    stopwords = list()
    for line in stop_file.readlines():
        stopwords.append(line.strip())
    stop_file.close()
    return stopwords

def write_cluster_result(fname, class_num, my_ipc):
    with open(fname, 'w', encoding='utf-8') as result_f:
        result_f.write('聚类结果为：\n')
        for label in class_num:
            result_f.write(str(label) + ':' + str(class_num[label]) + '\n')
        for label in my_ipc:
            result_f.write('类标签为:' + str(label) + ':' + '\n')
            result_f.write(str(class_num[label]) + '条专利' + '\n')
            for ipc in my_ipc[label]:
                result_f.write(str(label) + ':  ' + ipc + '\n')

def get_most_label(line_vecs, birch_model, dim=100):    #词向量加和
    line_matrix = np.zeros((1, dim))
    for vec in line_vecs:
        line_matrix = np.row_stack((line_matrix, vec))
    line_matrix = np.delete(line_matrix, 0, 0)
    line_AVG = np.mean(line_matrix, axis=0).reshape(1, dim)
    most_label = birch_model.predict(line_AVG)
    return most_label

def myCluster(words, wordvecs, birch_train_name, TSNE_name, clusterNum):       # 词向量加和平均
    dim = 100
    stopwords = get_stopwords('data/stopwords/stopwords_new.txt')
    word2ind = {word: i for i, word in enumerate(words)}
    test_vecs = np.zeros((1, dim))
    ipc_list = list()
    with open(birch_train_name, 'r', encoding='utf-8') as test_file:
        num = 0
        for test_line in test_file.readlines()[:200]:           #############################
            num += 1
            line_split = test_line.split(' ::  ')
            if len(line_split) == 2:
                ipc_list.append(line_split[0][:6])
                content = line_split[1].strip()
                test_line_words = list(jieba.cut(content))
                line_words = [word for word in test_line_words if word not in stopwords]
                line_wordvecs = np.zeros((1, dim))
                for i in range(len(line_words)):
                    if line_words[i] in word2ind:
                        cur_wordindex = word2ind[line_words[i]]
                        cur_wordvec = wordvecs[cur_wordindex].reshape(1, dim)
                        line_wordvecs = np.row_stack((line_wordvecs, cur_wordvec))
                line_wordvecs = np.delete(line_wordvecs, 0, 0)
                if line_wordvecs.all() == 0 or line_wordvecs.shape[0] == 0:
                    continue
                else:
                    cur_linevec = np.mean(line_wordvecs, axis=0).reshape(1, dim)
                    test_vecs = np.row_stack((test_vecs, cur_linevec))
                    print('%d:处理%s人物词条......' % (num, line_split[0]))
        test_vecs = np.delete(test_vecs, 0 , 0)
    print(test_vecs.shape)
    # model = Birch(threshold=birchThreshold, branching_factor=50, n_clusters=10).fit(test_vecs)
    model = KMeans(n_clusters=clusterNum, init='k-means++', n_init=clusterNum).fit(test_vecs)
    cluster = model.labels_
    labels_unique = np.unique(cluster)
    n_clusters_ = len(labels_unique)
    print('聚类的类别数目：%d' % n_clusters_)
    class_num = get_class_num(cluster)
    print('聚类结果为：')
    for label in class_num:
        print(str(label) + ':' + str(class_num[label]))
    label_vecs = get_kmeans_clusters(test_vecs, cluster, dim=100)
    centers = get_centers(label_vecs, dim=100)
    # PCA降维
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    low_dim_embs = tsne.fit_transform(test_vecs)
    label_low_vecs = get_kmeans_clusters(low_dim_embs, cluster, dim=2)
    low_centers = get_centers(label_low_vecs, dim=2)
    clusterJson = {}
    clusterJson['clusters'] = label_low_vecs
    clusterJson['centers'] = {}
    for i in range(n_clusters_):
        clusterJson['centers'][i] = low_centers[i].tolist()
    with open('data/cluster/cluster.json', 'w', encoding='utf-8') as jsonFile:
        json.dump(clusterJson, jsonFile, indent=4)
    ########## 分成两个程序的话要保存下列变量 ##############################
    # with open('data/cluster/model.pkl', 'wb') as modelFile:
    #     pickle.dump(model, modelFile)
    # with open('data/cluster/centers.pkl', 'wb') as centersFile:
    #     pickle.dump(centers, centersFile)
    ########## 分成两个程序的话要保存上述变量 ##############################
    ######################################### 画图 ##########################################
    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    plot_with_labels(clusterJson, TSNE_name)
    ######################################### 画图 ##########################################
    return model, centers

def keyword_extraction_JSON(logJsonName, test_name, words, wordvecs, kmeansModel, centers, dim=100, topn=20):
    logJsonFile = open(logJsonName, 'w', encoding='utf-8')
    testRes = []
    stopwords = get_stopwords('data/stopwords/stopwords_new.txt')
    keywordstop = get_stopwords('data/stopwords/mystop.txt')
    word2ind = {word: i for i, word in enumerate(words)}
    with open(test_name, 'r', encoding='utf-8') as test_file:
        num = 0
        for test_line in test_file.readlines():
            curTestRes = {}
            line_split = test_line.split(' ::  ')
            if len(line_split) == 2:
                content = line_split[1].strip()
                print('%s人物介绍：' % line_split[0].strip())
                print(content)
                curTestRes['name'] = line_split[0].strip()
                curTestRes['content'] = content
                test_line_words = list(jieba.cut(content))
                line_words = list()
                line_vecs = list()
                for word in test_line_words:
                    if word not in stopwords and word not in keywordstop and word in word2ind and len(word)>1 and not word.isdigit():
                        line_words.append(word)
                        cur_wordvec = wordvecs[word2ind[word]].reshape(1, dim)
                        line_vecs.append(cur_wordvec)
                assert len(line_words) == len(line_vecs)
                if len(line_vecs) < 1:
                    continue
                ind2vec = get_index2vectors(word2ind, wordvecs, line_words)
                most_label = get_most_label(line_vecs, kmeansModel)
                curTestRes['clusterLabel'] = int(most_label[0])
                center = centers[most_label]
                sorted_index_distance = distance_sort(ind2vec, center, 'cos')
                keyword_num = 0
                curKws = []
                for our_item in list(sorted_index_distance.items()):
                    our_word = words[our_item[0]]
                    our_dis = our_item[1]
                    curKws.append(our_word)
                    print(our_word + '%f' % our_dis)
                    print(curKws)
                    keyword_num += 1
                    if keyword_num >= topn:
                        break
                curTestRes['keywords'] = curKws
                print('------------------------------------------------------------------')
                num += 1
            testRes.append(curTestRes)
    json.dump(testRes, logJsonFile, ensure_ascii=False, indent=4)
    logJsonFile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UPKEM')
    parser.add_argument('--embedding_file', '-e', help='词嵌入模型',
                        default=r'data/word2vec/jsonVec.vec')
    parser.add_argument('--kmeans_train_name', '-k', help='聚类训练文件',
                        default=r'data/cluster/kmeansTrain.txt')
    parser.add_argument('--logJsonName', '-l', help='关键词提取JSON文件名',
                        default=r'data/log/logJson.json')
    parser.add_argument('--test_name', '-t', help='关键词提取测试文本',
                        default=r'data/test/keywordTest.txt')
    parser.add_argument('--clusterNum', '-c', help='Kmeans聚类簇个数',
                        default=3)
    parser.add_argument('--TSNE_name', '-n', help='聚类结果图位置',
                        default=r'data/figs/JSONcluster.png')
    args = parser.parse_args()
    embedding_name = args.embedding_file
    kmeans_train_name = args.kmeans_train_name
    logJsonName = args.logJsonName
    test_name = args.test_name
    clusterNum = args.clusterNum
    TSNE_name = args.TSNE_name
    words, wordvecs = read(embedding_name, dtype=float)
    ########## 分成两个程序的话要保存下列变量 ##############################
    # with open('data/cluster/words.pkl', 'wb') as wordsFile:
    #     pickle.dump(words, wordsFile)
    # with open('data/cluster/wordvecs.pkl', 'wb') as wordvecsFile:
    #     pickle.dump(wordvecs, wordvecsFile)
    ########## 分成两个程序的话要保存上述变量 ##############################
    kmeans_model, centers = myCluster(words, wordvecs, kmeans_train_name, TSNE_name, clusterNum)
    keyword_extraction_JSON(logJsonName, test_name, words, wordvecs, kmeans_model, centers)
