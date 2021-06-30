
import re

class N_Gram:
    # 文本清洗
    def cleanText(self, input):
        input = input.lower()  # 统一为小写

        # 英文缩写的还原
        input = re.sub("can\'t", "can not", input)
        input = re.sub("n\'t", " not", input)
        input = re.sub("what\'s", "what is", input)
        input = re.sub("\'ve ", " have ", input)
        input = re.sub(r"i\'m", "i am ", input)
        input = re.sub(r"\'re", " are ", input)
        input = re.sub(r"\'d", " would ", input)
        input = re.sub(r"\'ll", " will ", input)

        input = re.compile("[^a-z^0-9]").sub(" ", input)  # 只保留字母和数字
        input = re.sub('\n+', " ", input).lower()  # 匹配换行,用空格替换换行符
        input = re.sub(' +', " ", input)  # 把连续多个空格替换成一个空格
        return input.split(' ')  # 以空格为分隔符，返回列表

    # 构造n-gram模型，并统计词频
    def create_ngram(self, input, n):
        input = self.cleanText(input)
        frequency = {}  # 构造字典，统计ngram在该字典中的出现次数
        appear = {}  # 构造字典，统计ngram在该字典是否出现过

        for i in range(len(input) - n):
            ngramTemp = " ".join(input[i:i + n])
            if ngramTemp not in frequency:
                frequency[ngramTemp] = 0
            frequency[ngramTemp] += 1
            appear[ngramTemp] = 1
        return frequency, appear


