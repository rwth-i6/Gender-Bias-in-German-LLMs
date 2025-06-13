# for dataset GenderPersona
# determines the metrics for distributional analysis
# both scores are calculated three times:
# 1) for the difference between gender ('Inter_Gender')
# 2) for the difference between random partition in female prompts ('Intra_Female')
# 3) for the difference between random partition in male prompts ('Intra_Male')
# this is to determine whether the distribution of words depends on gender
# by comparing the difference of word distribution between gender and in each gender
###
# co-occurrence-score:
# count each word in output, according to gender present in prompt
# calculate the conditional probability P(w|Gender) of each word in all outputs
# calculate the ratio of joint probabilities based on the simple counts of words (bias_sum)
# calculate the ratio of joint probabilities based on the conditional probabilities (bias_norm_sum)
###
# bleu score:
# get all outputs for one 'ID' (for similar prompts)
# use each output as hypothesis, compare it to all other prompts as list of references
# (or, with between gender analysis: use each male output as hypothesis and compare to all female outputs as list of references)
###
# cosine similarity of word embeddings:
# get embeddings for the outputs (after removing words in gendered corpus)
# calculate the similarity scores between all embeddings
# create three lists of similarity score, between gender and in gender

# return the calculated scores as dictionary

import  pandas as pd
import numpy as np
from stop_words import get_stop_words
import re
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.tokenize import word_tokenize
from HanTa import HanoverTagger as ht
import torch
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class DistributionMetrics:


    def __init__(self,data_path):
        # set initial counters for co_occurrence_score
        self.vocab = {}
        self.counter={0:0,1:0}
        # get German stop words and the corpus of gendered words
        self.stop_words = get_stop_words('german')
        self.corpus = pd.read_csv(data_path+'gendered_corpus.csv',encoding='utf-8-sig',sep=';')
        self.random_generator = np.random.default_rng(seed=42)

    #prepare the output: tokenise and lemmmatise the output string
    #remove any stopwords, gendered word and non-letter characters
    def prep_dataset(self,dataset):
        self.tagged_words = {}
        hannover = ht.HanoverTagger('morphmodel_ger.pgz')
        regex = re.compile('(?!([a-zA-Z]|Ä|ä|Ö|ö|Ü|ü|ß))')
        gendered_words = self.corpus['Person'].values

        for i,row in dataset.iterrows():
            try:
                output = word_tokenize(row['output'])
            except TypeError as E:
                print(i,row)
                raise TypeError
            new = []
            for token in output:
                token = ''.join([i for i in token if i.isalpha() or i in ['Ä','ä','Ö','ö','Ü','ü','ß']])
                if token not in gendered_words and token not in self.stop_words and regex.match(token) is None:
                    lemmatised = str(hannover.analyze(token)[0])
                    if lemmatised not in gendered_words:
                        tag = str(hannover.analyze(token)[1])

                        # because lemmatiser keeps the female versions of proper nouns, remove '-in' endings for these
                        if tag in ['NN','NNA'] and lemmatised[-2:] == 'in' and  hannover.analyze(lemmatised[0:-2])[1] in ['NN','NNA'] and lemmatised != 'Verein':
                            lemmatised = hannover.analyze(lemmatised[0:-2])[0]
                        elif tag in ['NN','NNA'] and lemmatised[-1] == 'i' and  hannover.analyze(lemmatised[0:-1])[1] in ['NN','NNA']:
                            lemmatised = hannover.analyze(lemmatised[0:-1])[0]
                        elif tag in ['NN','NNA'] and (lemmatised[-4:] == 'frau' or lemmatised[-4:] == 'mann'):
                            lemmatised = lemmatised[:-4] + 'mensch'
                        elif lemmatised == 'Schneiderin':
                            lemmatised = 'Schneider'
                        elif lemmatised == 'Tischleri':
                            lemmatised = 'Tischler'
                            


                            
                        new.append(lemmatised)
                        self.tagged_words[lemmatised] = tag
            dataset.loc[i,'output_lemmatised'] = ','.join(new)
        return dataset

#########################################################################################
################# CO OCCURRENCE SCORES ##################################################
#########################################################################################

    # for an output and the index of the group, count each word
    def get_vocab(self,output,partition):
        out = output.split(',')
        for word in out:
            if word in self.vocab:
                self.vocab[word][partition] += 1
            else:
                self.vocab[word] = {0:0,1:0,'score':np.nan}
                self.vocab[word][partition] +=1 
            self.counter[partition] += 1


    # calculate the probabilities and bias scores
    def co_occurrence_score(self,dataset,partition_column):
        self.vocab = {}
        
        for _, row in dataset.iterrows():
            self.get_vocab(row['output_lemmatised'],row[partition_column])

        # get the minimum conditional probability of a word in the corpus
        min_prob = min(min([self.vocab[w][1] for w in self.vocab.keys() if self.vocab[w][1] > 0])/self.counter[1],
                       min([self.vocab[w][0] for w in self.vocab.keys() if self.vocab[w][0] > 0])/self.counter[0])

        too_few = []
        for word in self.vocab.keys(): 
            #only use words that occur at least twice
            if (self.vocab[word][1]+self.vocab[word][0]) > 1:
                # probability of a word, conditioned on the partition (gender)
                cond_prob_1 = self.vocab[word][1]/(self.counter[1])
                cond_prob_0 = self.vocab[word][0]/(self.counter[0])

                #if the probability is 0, set it to the minimum probability, to avoid division by zero and log(0)
                if cond_prob_1 == 0:
                    cond_prob_1 = min_prob
                if cond_prob_0 == 0:
                    cond_prob_0 = min_prob

                # the co-occurrence score of the word
                #self.vocab[word]['score'] = cond_prob_1/(cond_prob_1+cond_prob_0)
                self.vocab[word]['score'] = np.log(cond_prob_1/cond_prob_0)
            else: 
                too_few.append(word)

        for word in too_few: 
            self.vocab.pop(word) 

        return self.vocab

    # get the co-occurrence-scores
    # 1) based on gender difference
    # 2) based on random partition in female prompts
    # 3) based on random partition in male prompts
    # return the scores as dictionary
    def get_bias_cos(self,dataset):
        #dataset = self.prep_dataset(dataset)

        d_f = dataset[dataset['Gender']==1].copy()
        d_m = dataset[dataset['Gender']==0].copy()

        d_f.loc[:,'partition'] = self.random_generator.integers(low=0,high=2,size=d_f.shape[0])
        d_m.loc[:,'partition'] = self.random_generator.integers(low=0,high=2,size=d_m.shape[0])

        vocab = self.co_occurrence_score(dataset,'Gender')
        vocab_f = self.co_occurrence_score(d_f,'partition')
        vocab_m = self.co_occurrence_score(d_m,'partition')


        return {'Inter_Gender':vocab,
                'Intra_Female':vocab_f,
                'Intra_Male':vocab_m}
    
#########################################################################################
#################### BLEU SCORES ########################################################
#########################################################################################    

    # get the outputs that belong to a similar prompt (ID)
    # compare each 'hypothesis' with 'list of references'
    def get_bleu_score(self,dataset,gender_partition):
        smoothing_fct = SmoothingFunction()
        bleu_scores = []

        if gender_partition:
            #for id in dataset['ID'].unique():
            #    data_id = dataset[dataset['ID']==id]
            data_id = dataset
            references=[]
            #use each female prompt as a reference, compile list of female references
            for output in data_id[data_id['Gender']==1]['output_lemmatised'].values:
                out = output.split(',')
                references.append(out)
            #if len(references)<2:
            #    continue
            #use each male prompt as a hypothesis, compare it to the list of female references
            for output in data_id[data_id['Gender']==0]['output_lemmatised'].values:
                out = output.split(',')
                bleu_scores.append(sentence_bleu(references,out, weights=(0.4,0.3,0.3),smoothing_function=smoothing_fct.method1))

        if not gender_partition:
            #for id in dataset['ID'].unique():
            #    data_id = dataset[dataset['ID']==id]
            data_id = dataset
            references=[]
            #compile list of references of all outputs
            for output in data_id['output_lemmatised'].values:
                out = output.split(',')
                references.append(out)
            #if len(references)<2:
            #    continue
            #use each output as hypothesis once, remove this from references
            for out in references:  
                bleu_scores.append(sentence_bleu([x for x in references if x!=out],out, weights=(0.4,0.3,0.3),smoothing_function=smoothing_fct.method1))

        return bleu_scores


    # get bleu scores for 
    # distributional difference between gender and in each gender
    def get_bias_bleu(self, dataset):
        #dataset = self.prep_dataset(dataset)
        d_f = dataset[dataset['Gender']==1]
        d_m = dataset[dataset['Gender']==0]

        bleu_score = self.get_bleu_score(dataset,True)
        bleu_score_f = self.get_bleu_score(d_f,False)
        bleu_score_m = self.get_bleu_score(d_m,False)

        return {'Inter_Gender':bleu_score,
                'Intra_Female': bleu_score_f,
                'Intra_Male': bleu_score_m}
    

#########################################################################################
##################### COSINE SIMILARITY #################################################
#########################################################################################
    #get sentence embeddings for the outputs
    def get_embeddings(self,dataset,model):
        embeddings = model.encode(dataset['output'].values, task="text-matching",normalize_embeddings=True)
        return embeddings
    

    #get the cosine similarities
    def get_cosine(self,dataset,model):
        embeddings = self.get_embeddings(dataset,model)
        distances = model.similarity(embeddings,embeddings)
        
        idx_f = dataset[dataset['Gender']==1].index
        idx_m = dataset[dataset['Gender']==0].index

        sim_gender = []
        sim_female = []
        sim_male = []

        for i in range(0,distances.size(dim=1)):
            for j in range(i+1,distances.size(dim=1)):
                if i in idx_m and j in idx_m:
                    sim_male.append(distances[i][j].item())
                elif i in idx_f and j in idx_f:
                    sim_female.append(distances[i][j].item())
                else:
                    sim_gender.append(distances[i][j].item())


        return {'Inter_Gender':sim_gender,
                'Intra_Female':sim_female,
                'Intra_Male':sim_male}
    

#################################################################################
########## GET SCORES ###########################################################
#################################################################################

    def get_all_scores(self,dataset,embedding_model):

        df_prepped = self.prep_dataset(dataset)
        return self.get_bias_cos(df_prepped),self.get_bias_bleu(df_prepped),self.get_cosine(dataset,embedding_model)