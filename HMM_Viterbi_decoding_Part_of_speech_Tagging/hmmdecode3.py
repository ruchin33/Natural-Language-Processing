#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 20:26:55 2018

@author: ruchinpatel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 22:28:18 2018

@author: ruchinpatel
"""


import numpy as np
import itertools
import math
import time
import sys

word_tag = dict() #a dictionary with key as a tuple (word,tag) and the value as the count when word|tag occurs 
tagprev_tagcurr = dict()#a dictionary with key as a tuple (tag(i-1),tag(i)) and the value as the count when tag(i)|tag(i-1) occurs
tag_count = dict() #a dictionary with key as a tag and the value as the total count when tag occurs
word_count = dict() #a dictionary with key as a word and value as a the total count of the words
tag_end_count = dict() #a dictionary with key as a tag and the value as the count when tag occurs at the end of the sentence
transition_prob = dict() #a dictionary with key as a tuple (tag(i-1),tag(i)) and the value as the prob of transitioning from state(i-1) to state(i)
emission_prob = dict() #a dictionary with key as a tuple (word,tag) and the value as the prob of occurance of a word given a tag
tag_set = list()
possible_pos = 0
vocabulary_counts = 0
unique_tags_word = dict()

        
        

fhand = open('hmmmodel.txt')
count = 0
tran_prob_size = 0
emi_prob_size = 0
tag_count_size = 0
tag_end_count_size = 0
stage_trans = 0
stage_emi = 0
stage_tag_count = 0
stage_tag_end = 0

param_size_list = list()


for line in fhand:
    
    if(count == 7):
       stage_trans = 7 #Starts at 6
       stage_emi = stage_trans + int(param_size_list[3]) #starts at 18
       stage_tag_count = stage_emi + int(param_size_list[4]) #starts at 56
       
       stage_tag_end = stage_tag_count + int(param_size_list[5]) #starts at 60 
    if(count < 7):
        line = line.rstrip()
        line_list = line.split()
        individual_param_size_list = line_list[0].rsplit(':')
        individual_param_size = individual_param_size_list[1] 
        param_size_list.append(individual_param_size)
       
    elif((count>=stage_trans) and (count < stage_emi)):
        line = line.rstrip()
        line_list = line.split()
        #print(line_list)
        #print(':    transition Probability')
        trans_and_prob = line_list[0].rsplit(':',1)
        transition = trans_and_prob[0]
        prev_and_curr_tag = transition.rsplit('->',1)
        previous_tag = prev_and_curr_tag[0]
        current_tag = prev_and_curr_tag[1]
        prob = float(trans_and_prob[1])
        transition_prob[(previous_tag,current_tag)] = prob
        
        
    elif((count>=stage_emi) and (count < stage_tag_count)):
        line = line.rstrip()
        line_list = line.split()
        word_and_prob = line_list[0].rsplit(':',1)
        observation = word_and_prob[0]
        word_and_curr_tag = observation.rsplit('->',1)
        word_curr = word_and_curr_tag[0]
        curr_tag = word_and_curr_tag[1]
        state_prob = float(word_and_prob[1])
        emission_prob[(word_curr,curr_tag)] = state_prob
        #print(line_list)
        #print(':    emission Probability')
        
    elif((count>=stage_tag_count) and (count < stage_tag_end)):
        line = line.rstrip()
        line_list = line.split()
        tag_and_count = line_list[0].rsplit(':',1)
        tag_any = tag_and_count[0]
        count_any = float(tag_and_count[1])
        tag_count[tag_any] = count_any
        #print(line_list)
        #print(':    all Tag counts')
        
    else:
        line = line.rstrip()
        line_list = line.split()
        tagend_and_countend = line_list[0].rsplit(':',1)
        tag_end = tagend_and_countend[0]
        count_end = float(tagend_and_countend[1])
        tag_end_count[tag_end] = count_end
        #print(line_list)
        #print(':    end Tag counts')
    
    
    count = count+1
    
    
for key in emission_prob:
    unique_tags_word[key[0]] = unique_tags_word.get(key[0],[]) + [key[1]]

tag_count_size = int(param_size_list[0])
vocab_size = int(param_size_list[1])
total_sentences = int(param_size_list[2])

#for keys is emission_prob()




###############################Vuterbi Algorithm starts##############################
class ViterbiBlock:
    block_number = tuple()
    Viterbi_log_prob = 0.0
    block_word_j = ''
    block_POS_i = ''
    backtrace = dict()
    
    def __init__(self,w,t,list_num,col):
        self.block_number = (list_num,col)
        self.Viterbi_log_prob = 0.0
        self.block_word_j = w
        self.block_POS_i = t
        self.backtrace[self.block_number] = None
        
    def setViterbi_log_prob(self,max_vit_calc):
        #print('Inside class: '+str(trans_prob))
        #print(emi_prob)
        self.Viterbi_log_prob = max_vit_calc 
        
    def setBackPointer(self,sel_list,sel_col):
        if((sel_list == None) or (sel_col == None)):
            self.backtrace[self.block_number] = None
        else:
            self.backtrace[self.block_number] = (sel_list,sel_col)
        
    
    def getBlockNum(self):
        return self.block_number
    
    def getViterbi_Log_Prob(self):
        return self.Viterbi_log_prob
    
    def getBlock_word_j(self):
        return self.block_word_j
    
    def getBlock_POS_i(self):
        return self.block_POS_i
    
    def getBacktrace(self):
        return self.backtrace.get(self.block_number)
    
def generateTaggedSentence(final_state_obj):
    # print(final_state_obj.getBlockNum())
    current_block = final_state_obj.getBlockNum()
    list_num = 0
    col = 0
    word = ''
    tag = ''
    s_list = list()
    while (current_block is not None):
        list_num = current_block[0]
        col = current_block[1]
        word = Viterbi_block_list[list_num][col].getBlock_word_j()
        tag = Viterbi_block_list[list_num][col].getBlock_POS_i()
        current_block = Viterbi_block_list[list_num][col].getBacktrace()
        tagged_word = word+'/'+tag
        s_list = [tagged_word] + s_list
        
    return s_list
    

fhand = open(sys.argv[1])
#fhand = open('en_dev_raw.txt')
count = 0
tag_set = list(tag_count.keys())
fout = open('hmmoutput.txt','w') 

for line in fhand:
    start_time_line = time.time()
    line = line.rstrip()
    words = line.split()
    
    N = len(tag_set)
    T = len(words)
    
#    start_time = time.time()    
    
    Viterbi_block_list = [[] for i in range(T)]
    
    for i in range(len(Viterbi_block_list)):
        current_list = list()
        current_list = unique_tags_word.get(words[i])
        if(current_list == None):
            for j in range(len(tag_set)):
                Viterbi_block_list[i].append(ViterbiBlock(words[i],tag_set[j],i,j))
        else:
            for j in range(len(current_list)):
                tag_of_word = unique_tags_word.get(words[i])[j]
                Viterbi_block_list[i].append(ViterbiBlock(words[i],tag_of_word,i,j))
            
    #initialize the first column
    
    for s in range(len(Viterbi_block_list[0])):
        current_word = Viterbi_block_list[0][s].getBlock_word_j()
        current_tag =  Viterbi_block_list[0][s].getBlock_POS_i()
        current_trans_prob = transition_prob.get(('None',current_tag))
        current_emi_prob = emission_prob.get((current_word,current_tag))
        prev_vit_prob = 1
        #print('Outside class: '+str(current_trans_prob))
        #print(current_emi_prob)
        if(current_trans_prob == None): #here do laplace smoothing for the start state only
            current_trans_prob = 1/(total_sentences+tag_count_size)
            
        if(current_emi_prob == None):
            current_emi_prob = 1/(tag_count.get(current_tag) + vocab_size)
        
        curr_vit_calc = math.log(current_trans_prob)+math.log(current_emi_prob)+math.log(prev_vit_prob)
        Viterbi_block_list[0][s].setViterbi_log_prob(curr_vit_calc)
        Viterbi_block_list[0][s].setBackPointer(None,None)
#    print("initialize the first column took ", time.time() - start_time, " to run")
        
    for i in range(1,T):
        
        for s in range(len(Viterbi_block_list[i])):
            
            v_list = list()
            b_list= dict()
            
            for m in range(len(Viterbi_block_list[i-1])):
                
                prev_tag = Viterbi_block_list[i-1][m].getBlock_POS_i()
                current_tag = Viterbi_block_list[i][s].getBlock_POS_i()
                current_word = Viterbi_block_list[i][s].getBlock_word_j()
                prev_blocknum = Viterbi_block_list[i-1][m].getBlockNum()
                prev_vit_prob = Viterbi_block_list[i-1][m].getViterbi_Log_Prob()
                current_trans_prob = transition_prob.get((prev_tag,current_tag))
                current_emi_prob = emission_prob.get((current_word,current_tag))
                
                
                
                if(current_trans_prob == None):
                    if(tag_end_count.get(prev_tag) == None):
                        current_trans_prob = 1/(tag_count.get(prev_tag) + tag_count_size)
                    else:
                        current_trans_prob = 1/((tag_count.get(prev_tag)-tag_end_count.get(prev_tag))+ tag_count_size)
                    
                if(current_emi_prob == None):
                    current_emi_prob = 1/(tag_count.get(current_tag) + vocab_size)
                    
                    
                curr_prob_product = prev_vit_prob+math.log(current_trans_prob)+math.log(current_emi_prob)
                
                v_list.append(curr_prob_product)
                
#                print(prev_blocknum)
                
                b_list[prev_blocknum] = prev_vit_prob+math.log(current_trans_prob)
                
            Viterbi_block_list[i][s].setViterbi_log_prob(max(v_list))  
            maxValKey = max(b_list, key=b_list.get)
            Viterbi_block_list[i][s].setBackPointer(maxValKey[0],maxValKey[1])
           
        if(i ==(T-1)):
            
#            print(Viterbi_block_list[i])
            b_list= dict()
            for final_state in range(len(Viterbi_block_list[i])):
#                print(Viterbi_block_list[i][final_state].getViterbi_Log_Prob())
#                print(Viterbi_block_list[i][final_state].getBlockNum())
#                print(Viterbi_block_list[i][final_state].getBlock_word_j())
#                print(Viterbi_block_list[i][final_state].getBlock_POS_i())
#                print(Viterbi_block_list[i][final_state].getBacktrace())
#                print('final_state',final_state)
#                print('i',i)
                b_list[(i,final_state)] = Viterbi_block_list[i][final_state].getViterbi_Log_Prob()
            
            maxValKey = max(b_list, key=b_list.get)
#            print('max_val_key',maxValKey)
            sentence_list = list()
            sentence_list = generateTaggedSentence(Viterbi_block_list[maxValKey[0]][maxValKey[1]])
            
    for w in range(len(sentence_list)):
        if(w == (len(sentence_list)-1)):
            fout.write(sentence_list[w]+'\n')
        else:
            fout.write(sentence_list[w]+' ')
    
#    print(count)
#    print("Printing sentence to file took ", time.time() - start_time, "to run")
    
    count += 1
    
#    print("Tagging entire sentence took ", time.time() - start_time_line, "to run")
    
fout.close()

##########################Testing Accuracy#####################################################






