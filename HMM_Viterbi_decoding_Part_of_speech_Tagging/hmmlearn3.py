#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 20:28:04 2018

@author: ruchinpatel
"""


import numpy as np
import itertools
import sys

word_tag = dict() #a dictionary with key as a tuple (word,tag) and the value as the count when word|tag occurs 
tagprev_tagcurr = dict()#a dictionary with key as a tuple (tag(i-1),tag(i)) and the value as the count when tag(i)|tag(i-1) occurs
tag_count = dict() #a dictionary with key as a tag and the value as the total count when tag occurs
word_count = dict() #a dictionary with key as a word and value as a the total count of the words
tag_end_count = dict() #a dictionary with key as a tag and the value as the count when tag occurs at the end of the sentence
transition_prob = dict() #a dictionary with key as a tuple (tag(i-1),tag(i)) and the value as the prob of transitioning from state(i-1) to state(i)
emission_prob = dict() #a dictionary with key as a tuple (word,tag) and the value as the prob of occurance of a word given a tag
tag_set = list()
unique_tags_word = dict()
possible_pos = 0
vocabulary_counts = 0

class ParameterGenerator:
    
    def CalculateCounts(self,word_list):
        l1 = word_list[:]
        for i in range(len(l1)):
            curr_word = l1[i]
            word_split = curr_word.rsplit('/',1)
            
            #Calculates Word|Tag counts
            word_tag[(word_split[0],word_split[1])] = word_tag.get((word_split[0],word_split[1]),0) + 1
            #Generates a list of Unique tags for a word 
            #Calculates Tag counts
            tag_count[word_split[1]]=tag_count.get(word_split[1],0) + 1
            #Calculates counts of Every word in the Vocabulary
            word_count[word_split[0]]=word_count.get(word_split[0],0) + 1
            #Calculates counts of Tags occuring at the end
            if( i== (len(l1)-1)):
                tag_end_count[word_split[1]]=tag_end_count.get(word_split[1],0) + 1
            #Calculates the counts of tag(i)|tag(i-1)
            if(i == 0):
                tagprev_tagcurr[(None,word_split[1])] = tagprev_tagcurr.get((None,word_split[1]),0) + 1
            else:
                prev_word = l1[i-1]
                prev_word_split = prev_word.rsplit('/',1)    
                tagprev_tagcurr[(prev_word_split[1],word_split[1])] = tagprev_tagcurr.get((prev_word_split[1],word_split[1]),0) + 1
            
        #Generate a Tag-Set
        
    def CalculateParam(self,num_sentences):
        tag_set = list(tag_count.keys())
        possible_pos = len(tag_set)
        vocabulary_counts = len(word_count)
        
        for prev_curr_tag in list(tagprev_tagcurr.keys()):
            prev_tag = prev_curr_tag[0] 
            
            
            if(prev_tag == None):
                #A_i_j = (tagprev_tagcurr.get(prev_curr_tag) + 1)/(num_sentences + possible_pos)
                A_i_j = (tagprev_tagcurr.get(prev_curr_tag)+1)/(num_sentences + possible_pos)
                transition_prob[prev_curr_tag] = A_i_j
            else:
                #A_i_j = (tagprev_tagcurr.get(prev_curr_tag) + 1)/(tag_count.get(prev_tag) + possible_pos)
                if(tag_end_count.get(prev_tag) == None):
                    A_i_j = (tagprev_tagcurr.get(prev_curr_tag) + 1)/((tag_count.get(prev_tag)) + possible_pos)
                    transition_prob[prev_curr_tag] = A_i_j
                else:
                    A_i_j = (tagprev_tagcurr.get(prev_curr_tag) + 1)/(((tag_count.get(prev_tag))-tag_end_count.get(prev_tag)) + possible_pos)
                    transition_prob[prev_curr_tag] = A_i_j
        
        for word_given_tag in list(word_tag.keys()):
            tag_given = word_given_tag[1]
            
            #B_j = (word_tag.get(word_given_tag) + 1)/(tag_count.get(tag_given) + vocabulary_counts)
            B_j = (word_tag.get(word_given_tag) + 1)/(tag_count.get(tag_given) + vocabulary_counts)
            emission_prob[word_given_tag] = B_j
            

fhand = open(sys.argv[1])
count = 0
an = ParameterGenerator()
for line in fhand:
    line = line.rstrip()
    line_list = line.split()
    an.CalculateCounts(line_list)
    count = count+1
    
    
an.CalculateParam(count)
#print(line_list)

fout = open('hmmmodel.txt','w')
fout.write('POS_tag_count:'+str(len(tag_count.keys()))+'\n')
fout.write('Vocabulary_size:'+str(len(word_count))+'\n')
fout.write('Total_number_of_sentences:'+str(count)+'\n')
fout.write('Transition_Prob_size:'+str(len(transition_prob))+'\n')
fout.write('Emission_Prob_size:'+str(len(emission_prob))+'\n')
fout.write('Individual_tag_count:'+str(len(tag_count))+'\n')
fout.write('Individual_tag_count_at_end:'+str(len(tag_end_count))+'\n')
for previous_tag,current_tag in transition_prob.keys():
    if(previous_tag == None):
        fout.write('None'+'->'+current_tag+':'+str(transition_prob.get((previous_tag,current_tag)))+'\n')
    else:
        fout.write(previous_tag+'->'+current_tag+':'+str(transition_prob.get((previous_tag,current_tag)))+'\n')
        
for word_j,tag_j in emission_prob.keys():
    fout.write(word_j+'->'+tag_j+':'+str(emission_prob.get((word_j,tag_j)))+'\n')
        
for tag_any in tag_count.keys():
    fout.write(tag_any+':'+str(tag_count.get(tag_any))+'\n')
    
for tag_last in tag_end_count.keys():
    fout.write(tag_last+':'+str(tag_end_count.get(tag_last))+'\n')

fout.close()








