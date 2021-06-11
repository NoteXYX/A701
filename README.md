1、将词向量文件all_rm_abstract_100_mincount1.vec放入data/word2vec文件夹中

2、将聚类所需JSON文件放入data/cluster中，将测试关键词所需JSON文件放入data/test文件夹中

3、运行dataPre.py，在data/cluster/下生成K-means聚类训练文本jsonKmeansTrain.txt；在data/test下生成关键词提取测试文本jsonTest.txt

4、运行cluster.py，在data/figs/下生成聚类结果图JSONcluster.png；在data/log下生成关键词提取结果文件jsonTestLog.txt

