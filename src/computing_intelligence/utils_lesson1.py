#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本文件从notebook中抽离主要是为了是项目操作流程更清晰，适用于第一次任务。
"""
import os
import random
import re
import jieba
import pandas as pd

from collections import Counter


def create_grammar(grammar_str, split='->', line_split='\n'):
    """
    这个函数输入语法字符串，输出问题模型需要的字典数据结构。
    @grammar_str，自定义语法字符串；
    @split，line_split 语法规则分隔符以及行分隔符
    @return dict
    """
    grammar = {}
    for line in grammar_str.split(line_split):
        if not line.strip():
            continue
        exp, stmt = line.split(split)
        grammar[exp.strip()] = [s.split() for s in stmt.split('|')]
    
    return grammar

def generate(gram, target):
    """
    @gram dict 从语法字符串生成的字典数据结构
    @target str 目标
    @return str 生成的句子
    """
    if target not in gram:
        return target # 未匹配终止    
    # 递归生成语句
    return ''.join(generate(gram, t) for t in random.choice(gram[target]))

def generate_n(grammar_str, target='垃圾分类', k):
    """
    @grammar_str  str 语法字符串
    @target str 目标
    @n int 生成句子数量
    @return list 生成的句子以列表存储
    """
    result = []
    _grammar = create_grammar(grammar_str)
    for _ in range(k):
        result.append(generate(_grammar, target))
    
    return result

def get_clean_text(filepath):
    """
    如果是保险文件直接替换非文字支付，如果是影评则需要取comment字段。
    @filepath 文件全路径
    @return 清理后文件
    """
    filetype = os.path.splitext(filepath)[1]
    
    if filetype == '.csv':
        content = pd.read_csv(filepath,low_memory=False)
        return ''.join(re.sub('\W', '', str(comment)) for comment in content['comment'])
    if filetype == '.txt':
        with open(filepath,encoding='utf-8') as fp:
            return ''.join(re.sub('\W', '', line) for line in fp) 
                                 
def get_token(cleantext):
    """
    @cleantext str 清理后文件
    @return list 分词结果以列表存储
    """
    
    return list(jieba.cut(cleantext))

def get_ngram(token, n):
    """
    @token list 分词结果
    @n int 使用ngram模型
    @return list 使用zip高效处理token，生成ngram列表
    """
    _zip = zip(*(token[i:] for i in range(n)))
    
    return [''.join(i) for i in _zip]

def get_ngram_count(token, n):
    """根据语料token计数ngram模型并输出每个token出现次数"""
    ngram = get_ngram(token, n)
    
    return Counter(ngram)

def get_2words_probablity(previous_word, current_word, 
                          previous_count, current_count, ngram_length):
    """
    @previous_word 前词
    @current_word 后词
    @current_count ngram模型计数统计
    @previous_count n-1gram模型计数统计
    @ngram_length 语料库ngram大小
    @return 返回前后词在语料中的概率
    """
    #Add-one（Laplace） Smoothing
    return (current_count[current_word]+1) / (previous_count[previous_word]+ngram_length)
    
def get_sentence_probablity(sentence, token, n):
    """ 测试语句在token语料下选用ngram模型出现的概率
    @sentence 输入语句
    @token 语料库切词后结果
    @n 选取ngram模型
    @return 返回句子合理性概率
    """
    ngram_length = len(get_ngram(token, n))                                       
    current_count = get_ngram_count(token, n)
    previous_count = get_ngram_count(token, n-1)
    words = get_token(sentence) 
    sentence_probablity = 1
    
    for index, word in enumerate(words[2:]):
        current_word = ''.join(words[index-2:index])
        previous_word = ''.join(words[index-2:index-1])
        probability = get_2words_probablity(previous_word, current_word, 
                                            previous_count, current_count, ngram_length)
        sentence_probablity *= probability
    
    return sentence_probablity

def generate_best(grammar_str, n=2, k=12): 
    """该函数输入一个语法 + 语言模型，能够生成n个句子，并能选择一个最合理的句子"""

    movie_comments = '..\..\data\movie_comments.csv'
    insurance_corpus = '..\..\data\insuranceqa-corpus-train.txt'
    corpus = get_clean_text(movie_comments) + get_clean_text(insurance_corpus)
    token = get_token(corpus)
    sentences = generate_n(grammar_str, k)
    probability = []
    for index, sentence in enumerate(sentences):
        probability.append(get_sentence_probablity(sentence, token, n))    
    #list(map(language_model, sentences, [token for i in range(len(sentences))]))
    result = sorted(list(zip(sentences, probability)), key=lambda x: x[1], reverse=True)
    print(f"最有可能的句子是 : {result[0][0]} 概率是: {result[0][1]}")
    
    return result


if __name__ == "__main__":
    garbage_classification = """
垃圾分类 -> 具体垃圾 谓语 分类垃圾
具体垃圾 -> 饮料盒 | 报纸 | 药瓶 | 易拉罐 | 毛巾 |电视机 | 蓄电池 |\
        节能灯 |温度计 | 剩面条 | 果皮 | 奶酪 | 猫粮 |厕纸 | 陶瓷 | \
        大骨头 |鱼虾 | 调味料 | 中药渣 | 肉蛋 | 菌菇 | 花卉 | 豆腐
谓语 -> 副词 动词
副词 -> 很显然 | 有可能 | 不可能
动词 -> 属于
分类垃圾 -> 有害垃圾 | 可回收垃圾 | 湿垃圾 | 干垃圾
"""
    generate_best(garbage_classification)
    
    
    
