
import os
import json
import math
from n_gram import *

class GetTFIDF:
    def __init__(self):
        self.data_path = os.path.join(os.path.abspath('.'), 'data/corpus.json') #获取当前的绝对路径
        self.ngram = N_Gram()

    # 读取输入文本，将其拆分成三种n-gram，同时计算每种ngram的doc_frequency和total frequency
    def get_data(self):
        with open(self.data_path, 'r') as f:
            datas = json.load(f)
            # 所有字典n-gram的累加，用于统计total frequency
            total_unigram_fre = {}
            total_bigram_fre = {}
            total_trigram_fre = {}
            # 所有字典n-gram的累加，用于统计doc frequency
            total_unigram_apr = {}
            total_bigram_apr = {}
            total_trigram_apr = {}
            for data in datas:
                pageTitle = data['pageTitle']
                body = data['body']
                if 'subTitles' in data:
                    subTitles = ' '.join(data['subTitles'])
                text = pageTitle + " " + subTitles + " " + body

                # n_gram_fre统计本字典的frequency，n_gram_apr统计是否在本字典中出现过
                # 并利用字典合并的方式将结果累加，以计算doc frequency和total frequency
                uni_gram_fre, uni_gram_apr = self.ngram.create_ngram(text, 1)
                self.merge_dict(uni_gram_fre, total_unigram_fre)
                self.merge_dict(uni_gram_apr, total_unigram_apr)
                bi_gram_fre, bi_gram_apr = self.ngram.create_ngram(text, 2)
                self.merge_dict(bi_gram_fre, total_bigram_fre)
                self.merge_dict(bi_gram_apr, total_bigram_apr)
                tri_gram_fre, tri_gram_apr = self.ngram.create_ngram(text, 3)
                self.merge_dict(tri_gram_fre, total_trigram_fre)
                self.merge_dict(tri_gram_apr, total_trigram_apr)

            # 存储三种n-gram的doc_frequency
            self.total_uni_doc = total_unigram_apr
            self.total_bi_doc = total_bigram_apr
            self.total_tri_doc = total_trigram_apr

            # 存储三种n-gram的total frequency
            self.total_uni_fre = total_unigram_fre
            self.total_bi_fre = total_bigram_fre
            self.total_tri_fre = total_trigram_fre

            # 存储三种n-gram的IDF值
            self.unigram_idf = self.caculate_IDF(total_unigram_apr)
            self.bigram_idf = self.caculate_IDF(total_bigram_apr)
            self.trigram_idf = self.caculate_IDF(total_trigram_apr)
            # 存储三种n-gram的TF-IDF值
            self.unigram_tfidf = self.caculate_TF_IDF(total_unigram_fre, total_unigram_apr)
            self.bigram_tfidf = self.caculate_TF_IDF(total_bigram_fre, total_bigram_apr)
            self.trigram_tfidf = self.caculate_TF_IDF(total_trigram_fre, total_trigram_apr)

    # 字典合并，有相同的key则对应的value相加，若没有则直接添加过来
    def merge_dict(self, dict1, dict2):
        for key, value in dict1.items():
            if key in dict2.keys():
                dict2[key] += value
            else:
                dict2[key] = value

    # 计算每个token的IDF值
    def caculate_IDF(self, doc_fre):
        doc_num = 1000
        word_idf = {}  # 存储每个词的IDF值
        for key, value in doc_fre.items():
            word_idf[key] = math.log(doc_num / (value + 1))
        return word_idf

    # 计算每个token的TF-IDF值
    def caculate_TF_IDF(self, total_fre, doc_fre):
        word_tf = {}  # 存储每个词的TF值
        for key, value in total_fre.items():
            word_tf[key] = value / len(total_fre)

        doc_num = 1000
        word_idf = {}  # 存储每个词的IDF值
        for key, value in doc_fre.items():
            word_idf[key] = math.log(doc_num / (value + 1))

        word_tf_idf = {}  # 存储每个词的TF-IDF值
        for key in total_fre.keys():
            word_tf_idf[key] = word_tf[key] * word_idf[key]
        return word_tf_idf

    # 生成task 1的global_vocab.json文件
    def output_global(self):
        gloabal_vocab = []
        # unigram的数据
        for key in self.total_uni_doc.keys():
            data = {}
            data["token"] = key
            data["ngram"] = 1
            data["doc_frequency"] = self.total_uni_doc[key]
            data["frequency"] = self.total_uni_fre[key]
            data["idf"] = self.unigram_idf[key]
            data["tfidf"] = self.unigram_tfidf[key]
            gloabal_vocab.append(data)

        # bigram的数据
        for key in self.total_bi_doc.keys():
            data = {}
            data["token"] = key
            data["ngram"] = 2
            data["doc_frequency"] = self.total_bi_doc[key]
            data["frequency"] = self.total_bi_fre[key]
            data["idf"] = self.bigram_idf[key]
            data["tfidf"] = self.bigram_tfidf[key]
            gloabal_vocab.append(data)

        # trigram的数据
        for key in self.total_tri_doc.keys():
            data = {}
            data["token"] = key
            data["ngram"] = 3
            data["doc_frequency"] = self.total_tri_doc[key]
            data["frequency"] = self.total_tri_fre[key]
            data["idf"] = self.trigram_idf[key]
            data["tfidf"] = self.trigram_tfidf[key]
            gloabal_vocab.append(data)

        gloabal_vocab.sort(key=lambda x: x["doc_frequency"], reverse=True)  # 将结果按照doc_frequency降序排序
        filename = os.path.join(os.path.abspath('.'), 'data/global_vocab.json')
        f = open(filename, 'w')
        json.dump(gloabal_vocab, f, sort_keys=False, indent=2)  # 将结果保存到json文件中

    # 生成task 1的corpus_with_tokens.json文件
    def output_corpus(self):
        with open(self.data_path, 'r') as f:
            datas = json.load(f)
            corpus_tokens = []
            for data in datas:
                # 一个corpus中有多个document
                doc = {}
                id = data['id']
                pageTitle = data['pageTitle']
                body = data['body']
                doc["id"] = id
                doc["pageTitle"] = pageTitle
                if 'subTitles' in data:
                    doc["subTitles"] = data['subTitles']
                    subTitles = ' '.join(data['subTitles'])
                doc["body"] = body
                text = pageTitle + " " + subTitles + " " + body

                #保存一个document中的多个token
                doc_tokens = []
                # 统计每个token在当前document中的出现次数frequency
                uni_gram_fre, uni_gram_apr = self.ngram.create_ngram(text, 1)
                for key, value in uni_gram_fre.items():
                    token = {}
                    token["token"] = key
                    token["ngram"] = 1
                    token["frequency"] = value
                    token["tfidf"] = self.unigram_tfidf[key]
                    doc_tokens.append(token)

                bi_gram_fre, bi_gram_apr = self.ngram.create_ngram(text, 2)
                for key, value in bi_gram_fre.items():
                    token = {}
                    token["token"] = key
                    token["ngram"] = 2
                    token["frequency"] = value
                    token["tfidf"] = self.bigram_tfidf[key]
                    doc_tokens.append(token)

                tri_gram_fre, tri_gram_apr = self.ngram.create_ngram(text, 3)
                for key, value in tri_gram_fre.items():
                    token = {}
                    token["token"] = key
                    token["ngram"] = 3
                    token["frequency"] = value
                    token["tfidf"] = self.trigram_tfidf[key]
                    doc_tokens.append(token)

                doc["tokens"] = doc_tokens
                corpus_tokens.append(doc)

            filename = os.path.join(os.path.abspath('.'), 'data/corpus_with_tokens.json')
            f = open(filename, 'w')
            json.dump(corpus_tokens, f, sort_keys=False, indent=2)  # 将结果保存到json文件中


if __name__ == '__main__':
    handler = GetTFIDF()
    handler.get_data()
    handler.output_global()
    handler.output_corpus()
