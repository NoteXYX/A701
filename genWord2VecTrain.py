import jieba
import json

def fenci(jsonName, vecTrainTxtName):
    vecTrainTxt = open(vecTrainTxtName, 'w', encoding='utf-8')
    with open(jsonName, 'r', encoding='utf-8') as load_f:
        filesList = json.load(load_f)
        for file in filesList:
            vecTrainTxt.write(file['name'] + ' ')
            seg_list = jieba.cut(file['text'])
            vecTrainTxt.write(" ".join(seg_list) + '\n')
    vecTrainTxt.close()

if __name__ == '__main__':
    vecTrainJson = 'data/word2vec/data14-500xN.json'
    vecTrainTxtName = 'data/word2vec/jsonVecTrain.txt'
    fenci(vecTrainJson, vecTrainTxtName)