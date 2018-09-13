#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 23:38:42 2018

@author: ruchinpatel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 23:04:52 2018

@author: ruchinpatel
"""

import sys
import numpy as np
import string

fhand = open(sys.argv[1])

bias1 = 0
bias2 = 0
c1_w_length = 0
c2_w_length = 0
V_l = 0
class1_weight = []
class2_weight = []
word_Vocab_index = dict()

count = 0
for line in fhand:
    if(count == 0):
#        print(count)
        temp = 0
        for word in line.split():
            if(temp==1):
                bias1 = int(word)
            temp = temp+1
            
    elif(count == 1):
#        print(count)
        temp = 0
        for word in line.split():
            if(temp==1):
                bias2 = int(word)
            temp = temp+1
    elif(count == 2):
#        print(count)
        temp = 0
        for word in line.split():
            if(temp==1):
                c1_w_length = int(word)
            temp = temp+1
            
    elif(count == 3):
#        print(count)
        temp = 0
        for word in line.split():
            if(temp==1):
                c2_w_length = int(word)
            temp = temp+1
            
    elif(count == 4):
#        print(count)
        temp = 0
        for word in line.split():
            if(temp==1):
                V_l = int(word)
            temp = temp+1
            
    elif(count == 5):
        bias1 = float(line.split()[0])
    
    elif(count == 6):
#        print(count)
        bias2 = float(line.split()[0])
        
    elif(count == 7):
#        print(count)
        for word in line.split():
            class1_weight.append(float(word))
    
    elif(count == 8):
#        print(count)
        for word in line.split():
            class2_weight.append(float(word))
        
    else:
        temp = 0
        key = ''
        idx = 0
        for word in line.split():
            
            if(temp == 0):
                key = word
            if(temp==1):
                idx = int(word)
            temp = temp+1
        
        word_Vocab_index[key] = idx

    count = count+1
    
class1_weight = np.array(class1_weight)
class2_weight = np.array(class2_weight) 

stop_w = ['a','about','above','after','again','against','all','am','an','and','any','are',"aren't",'as','at','be','because','been','before','being','below','between','both','but','by',"can't",'cannot','could',"couldn't",'did',"didn't",'do','does',"doesn't",'doing',"don't",'down','during','each','few','for','from','further','had',"hadn't",'has',"hasn't",'have',"haven't",'having','he',"he'd","he'll","he's",'her','here',"here's",'hers','herself','him','himself','his','how',"how's",'i',"i'd","i'll","i'm","i've",'if','in','into','is',"isn't",'it',"it's",'its','itself',"let's",'me','more','most',"mustn't",'my','myself','no','nor','not','of','off','on','once','only','or','other','ought','our','ours','ourselves','out','over','own','same',"shan't",'she',"she'd","she'll","she's",'should',"shouldn't",'so','some','such','than','that',"that's",'the','their','theirs','them','themselves','then','there',"there's",'these','they',"they'd","they'll","they're","they've",'this','those','through','to','too','under','until','up','very','was',"wasn't",'we',"we'd","we'll","we're","we've",'were',"weren't",'what',"what's",'when',"when's",'where',"where's",'which','while','who',"who's",'whom','why',"why's",'with',"won't",'would',"wouldn't",'you',"you'd","you'll","you're","you've",'your','yours','yourself','yourselves']
punctuation = string.punctuation


stop_words = np.array(stop_w)
word_freq = dict()
N_doc = 0
Corpus = []
fhand = open(sys.argv[2])

#############Converting to lower case and then Removing stop words ###########
tag_list=[]
for line in fhand:
    tag_list.append(line.split()[0])
    doc = line
    doc = doc.lower()
#    print(doc)
    updated_doc = ""
    for word in doc.split():
#        print(word)
        if(word not in stop_words):
            updated_doc = updated_doc + word + " "
            
#    print(updated_doc)
    N_doc = N_doc + 1
    Corpus.append(updated_doc)
    
#print("Total docs are: ",N_doc)
    
###########Remove punctuations################################################

table = str.maketrans({key: None for key in string.punctuation})
for i in range(0,len(Corpus)):
    doc = Corpus[i]
    Corpus[i] = doc.translate(table)
    
#print(Corpus)
#print()
#print()
#print()
    
which_index = 0
Corpus_dictionary = dict()
for i in range(len(Corpus)):
    count = 0
    tag = ''
    sentence_word_count = dict()
    for word in Corpus[i].split():

        if(count == 0):
            tag = word
        else:
            if(word_Vocab_index.get(word) != None):
                
                sentence_word_count[word] = sentence_word_count.get(word,0) + 1
            
            
        count = count + 1
    
    Corpus_dictionary[tag_list[i]] = sentence_word_count
    
    
data_dev = np.zeros((len(Corpus),len(word_Vocab_index)))   
    
for i in range(len(tag_list)):
    d1 = Corpus_dictionary[tag_list[i]]
    for keys in d1:
        word = keys
        if(word_Vocab_index.get(word) != None):
            index = word_Vocab_index.get(word)
            data_dev[i,index] = d1.get(keys)
    
class1_predicted = (np.dot(data_dev,class1_weight)) + bias1 
class2_predicted = (np.dot(data_dev,class2_weight)) + bias2

fout = open('percepoutput.txt','w')

for i in range(len(class1_predicted)):
    
    label1 = ''
    label2 = ''
    
    if(class1_predicted[i] > 0):
        label1 = 'True'
    else:
        label1 = 'Fake'
        
    if(class2_predicted[i] > 0):
        label2 = 'Pos'
    else:
        label2 = 'Neg'
        
    fout.write(tag_list[i]+' '+label1+' '+label2+'\n')
    
fout.close()
    
    
    

