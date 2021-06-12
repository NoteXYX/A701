1、安装必要的python包

- `pip install jieba`
- `pip install numpy`
- `pip install scipy`
- `pip install matplotlib`
- `pip install scikit-learn`
- `pip install gensim`

2、将训练词向量的JSON文件如："data14-500xN.json"放入data/word2vec/中，随后执行`python3 genWord2VecTrain.py`，即可在data/word2vec/下生成训练词向量所需的语料库"jsonVecTrain.txt"

3、运行`python3 train_word2vec_model.py data/word2vec/jsonVecTrain.txt data/word2vec/jsonVec.model data/word2vec/jsonVec.vec`即可在data/word2vec/
下生成词向量文件"jsonVec.vec"

4、将聚类所需JSON文件如："resultJson.json"放入data/cluster/中，将测试关键词所需JSON文件如："testJson.json"放入data/test/中

5、运行`python3 dataPre.py`，在data/cluster/下生成K-means聚类训练文本"kmeansTrain.txt"；在data/test/下生成关键词提取测试文本"keywordTest.txt"

6、运行`python3 cluster.py`，在data/figs/下生成聚类结果图"JSONcluster.png"；在data/cluster/下生成聚类结果的JSON文件"cluster.json"
其中包含每个簇中每个点的x,y坐标(clusters)以及每个簇的簇心x,y坐标(centers)；在data/log/下生成关键词提取结果JSON文件"logJson.json"，其中包含每个
测试人物的人名(name)、描述文本(content)、预测簇标签(clusterLabel)、描述文本中提取出的关键词(keywords)

