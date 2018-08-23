#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 18:05:22 2018

@author: ruchinpatel
"""

import numpy as np
import sys
import string

Vocabulary = set()
Log_Prior_pos = 0
Log_Prior_neg = 0
Log_Prior_true = 0
Log_Prior_fake = 0
Log_word_given_class = dict()

fhand = open('nbmodel.txt')

stop_w = ['a','about','above','after','again','against','all','am','an','and','any','are',"aren't",'as','at','be','because','been','before','being','below','between','both','but','by',"can't",'cannot','could',"couldn't",'did',"didn't",'do','does',"doesn't",'doing',"don't",'down','during','each','few','for','from','further','had',"hadn't",'has',"hasn't",'have',"haven't",'having','he',"he'd","he'll","he's",'her','here',"here's",'hers','herself','him','himself','his','how',"how's",'i',"i'd","i'll","i'm","i've",'if','in','into','is',"isn't",'it',"it's",'its','itself',"let's",'me','more','most',"mustn't",'my','myself','no','nor','not','of','off','on','once','only','or','other','ought','our','ours','ourselves','out','over','own','same',"shan't",'she',"she'd","she'll","she's",'should',"shouldn't",'so','some','such','than','that',"that's",'the','their','theirs','them','themselves','then','there',"there's",'these','they',"they'd","they'll","they're","they've",'this','those','through','to','too','under','until','up','very','was',"wasn't",'we',"we'd","we'll","we're","we've",'were',"weren't",'what',"what's",'when',"when's",'where',"where's",'which','while','who',"who's",'whom','why',"why's",'with',"won't",'would',"wouldn't",'you',"you'd","you'll","you're","you've",'your','yours','yourself','yourselves']
stop_words_new = ['however','can','why','found','whom','between','ten','further','than','being','thereby','where','already','within','done','indeed','ie','twelve','became','call','she','after','here','find','my','without','me','us','whole','last','his','through','very','whereupon','whatever','hasnt','in','it','most','via','amongst','your','somewhere','anything','bottom','see','yourself','except','perhaps','other','but','how','whence','before','themselves','eg','must','too','hundred','empty','latterly','almost','ever','into','somehow','upon','formerly','and','could','should','de','by','nevertheless','fifty','him','forty','go','now','only','thus','again','himself','nowhere','what','on','neither','each','co','beforehand','one','everywhere','four','six','bill','sometimes','their','third','while','ltd','full','then','were','system','etc','hence','though','yours','seeming','have','cant','might','re','everything','whether','rather','give','or','serious','from','do','three','towards','whose','also','something','they','if','thereupon','you','latter','is','i','off','no','an','least','same','twenty','below','not','first','nothing','mill','inc','be','around','them','a','hereby','every','so','those','still','well','made','whereby','had','herself','he','will','under','beside','fill','meanwhile','been','together','move','toward','therein','whereas','much','throughout','our','although','down','former','interest','put','would','which','its','per','with','due','nobody','who','mostly','whoever','namely','fifteen','am','ours','own','about','such','that','alone','to','eight','either','name','up','thence','anyone','was','anywhere','keep','the','yourselves','at','becomes','fire','whither','during','besides','may','nine','several','these','thin','when','otherwise','whereafter','because','seemed','some','less','has','others','ourselves','seem','top','side','amoungst','any','sincere','seems','anyway','thru','along','all','more','thick','her','please','behind','of','else','becoming','next','noone','against','wherein','hers','cry','whenever','among','cannot','describe','detail','few','hereafter','nor','there','thereafter','above','afterwards','couldnt','another','for','front','get','as','many','moreover','both','are','sixty','yet','since','everyone','mine','two','enough','amount','five','take','beyond','back','therefore','un','elsewhere','herein','show','over','eleven','often','someone','this','always','con','hereupon','once','sometime','wherever','part','anyhow','we','itself','until','none','myself','never','onto','become','even','across','out']

stop_words_set = set()
for word in stop_w:
    stop_words_set.add(word)
    
for word in stop_words_new:
    stop_words_set.add(word)
    
final_stop_words_list = list(stop_words_set)


stop_words = np.array(final_stop_words_list)
Corpus = []

count = 0
for line in fhand:
    if(count == 0):
#        print(count)
        temp = 0
        for word in line.split():
            if(temp==1):
                total_prior = int(word)
            temp = temp+1
            
    if(count == 1):
#        print(count)
        temp = 0
        for word in line.split():
            if(temp==1):
                Vocab_len = int(word)
            temp = temp+1
    if(count == 2):
#        print(count)
        temp = 0
        for word in line.split():
            if(temp==1):
                MLE_count = int(word)
            temp = temp+1
            
    if(count == 3):
#       print(count)
       Log_Prior_pos = float(line)
       next
#       print(count)
       Log_Prior_neg = float(line)
       next
#       print(count)
       Log_Prior_true = float(line)
       next
#       print(count)
       Log_Prior_fake = float(line)
      
    if(count == 7):
#        print(count)
        for word in line.split():
            Vocabulary.add(word)
    if(count>7):
        temp = 0
        for word in line.split():
            if(temp==0):
                w = word
            if(temp==1):
                c = word
            if(temp==2):
                p = float(word)
            
            temp = temp+1
            
        Log_word_given_class[(w,c)] = p   

    count = count+1

fhand = open(sys.argv[1])
fout = open('nboutput.txt','w')

#############Converting to lower case and then Removing stop words ###########

for line in fhand:
    first_word = line[0:7]
    doc_1 = line[7:len(line)]
    doc_1 = doc_1.lower()
    doc = first_word+" "+doc_1
#    print(doc)
    updated_doc = ""
    for word in doc.split():
#        print(word)
        if(word not in stop_words):
            updated_doc = updated_doc + word + " "
            
#    print(updated_doc)
    
    Corpus.append(updated_doc)
    
###########Remove punctuations################################################

table = str.maketrans({key: None for key in string.punctuation})
for i in range(0,len(Corpus)):
    doc = Corpus[i]
    Corpus[i] = doc.translate(table)
    
#print(Corpus)
#print()
#print()
#print()

for i in range(0,len(Corpus)):
    temp = 0
    log_pos_sum = Log_Prior_pos
    log_neg_sum = Log_Prior_neg
    log_true_sum = Log_Prior_true
    log_fake_sum = Log_Prior_fake
    
    for word in Corpus[i].split():
        if(temp==0):
            alpha_numeric = word
            temp = temp+1
        log_pos_sum = log_pos_sum + Log_word_given_class.get((word,"pos"),0)
        log_neg_sum = log_neg_sum + Log_word_given_class.get((word,"neg"),0)
        log_true_sum = log_true_sum + Log_word_given_class.get((word,"true"),0)
        log_fake_sum = log_fake_sum + Log_word_given_class.get((word,"fake"),0)
    
    fout.write(alpha_numeric+" ")
    if(log_true_sum>=log_fake_sum):
        fout.write("True"+" ")
    else:
        fout.write("Fake"+" ")
    
    if(log_pos_sum>=log_neg_sum):
        fout.write("Pos"+"\n")
    else:
        fout.write("Neg"+"\n")
        
        
fout.close()
    

        
