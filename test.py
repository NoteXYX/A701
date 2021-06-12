import jieba
from cluster import *
import json
import pickle


def keyword_extraction_JSON(log_file_name, test_name, words, wordvecs, birch_model, centers, dim=100, topn=20):
    log_file = open(log_file_name, 'w', encoding='utf-8')
    stopwords = get_stopwords('data/stopwords/stopwords_new.txt')
    keywordstop = get_stopwords('data/stopwords/mystop.txt')
    word2ind = {word: i for i, word in enumerate(words)}
    with open(test_name, 'r', encoding='utf-8') as test_file:
        num = 0
        for test_line in test_file.readlines()[:200]:       ####################
            line_split = test_line.split(' ::  ')
            if len(line_split) == 2:
                content = line_split[1].strip()
                print('%s人物介绍：' % line_split[0].strip())
                print(content)
                log_file.write('%s人物介绍：\n' % line_split[0].strip())
                log_file.write('%s\n' % content)
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
                most_label = get_most_label(line_vecs, birch_model)
                center = centers[most_label]
                sorted_index_distance = distance_sort(ind2vec, center, 'cos')
                keyword_num = 0
                for our_item in list(sorted_index_distance.items()):
                    our_word = words[our_item[0]]
                    our_dis = our_item[1]
                    log_file.write('%s\n' % our_word)
                    print(our_word + '%f' % our_dis)
                    keyword_num += 1
                    if keyword_num >= topn:
                        break
                print('------------------------------------------------------------------')
                log_file.write('------------------------------------------------------------------\n')
                num += 1
    log_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--log_file_name', '-l', help='日志文件名',
                        default=r'data/log/jsonTestLog.txt')
    parser.add_argument('--test_name', '-t', help='关键词提取测试文本',
                        default=r'data/test/jsonTest.txt')
    parser.add_argument('--topn', '-n', help='关键词提取个数',
                        default=20)
    args = parser.parse_args()
    log_file_name = args.log_file_name
    test_name = args.test_name
    topn = args.topn
    load_words = open('data/cluster/words.pkl', 'rb')
    words = pickle.load(load_words)
    load_words.close()
    load_wordvecs = open('data/cluster/wordvecs.pkl', 'rb')
    wordvecs = pickle.load(load_wordvecs)
    load_wordvecs.close()
    modelFile = open('data/cluster/model.pkl', 'rb')
    kmeans_model = pickle.load(modelFile)
    modelFile.close()
    centersFile = open('data/cluster/centers.pkl', 'rb')
    centers = pickle.load(centersFile)
    centersFile.close()
    keyword_extraction_JSON(log_file_name, test_name, words, wordvecs, kmeans_model, centers, dim=100, topn=topn)