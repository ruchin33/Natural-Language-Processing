#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 00:03:37 2018

@author: ruchinpatel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 23:37:59 2018

@author: ruchinpatel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 21:21:40 2018

@author: ruchinpatel
"""

import sys
import numpy as np
import string


def VanillaPerceptron(data,class1_labels,class2_labels,max_iter):
    
    w_1 = np.zeros((data.shape[1],))
    w_2 = np.zeros((data.shape[1],))
    b1 = 0
    b2 = 0
    w1_T_y = 0
    w2_T_y = 0
    
#---------------For class 1------------------------------
    for itr in range(0,max_iter):
        a1 = 0
        for i in range(0,data.shape[0]):
            
            w1_T_y = np.dot(data[i,:],w_1)
            a1 = w1_T_y + b1
            if((a1*class1_labels[i]) <= 0):
                w_1 = w_1 + (class1_labels[i]*data[i,:])
                b1 = b1 + class1_labels[i]
                
#---------------For class 2------------------------------
    for itr in range(0,max_iter):
        a2 = 0
        for i in range(0,data.shape[0]):
            
            w2_T_y = np.dot(data[i,:],w_2)
            a2 = w2_T_y + b2
            if((a2*class2_labels[i]) <= 0):
                w_2 = w_2 + (class2_labels[i]*data[i,:])
                b2 = b2 + class2_labels[i]
                
    return {'class1':w_1,'bias1':b1,'class2':w_2,'bias2':b2}



def AveragedPerceptron(data,class1_labels,class2_labels,max_iter):
    
    w_1 = np.zeros((data.shape[1],))
    w_2 = np.zeros((data.shape[1],))
    u1 = np.zeros((data.shape[1],))
    u2 = np.zeros((data.shape[1],))
    c1 = 1
    c2 = 1
    b1 = 0
    b2 = 0
    Beta1 = 0
    Beta2 = 0
    w1_T_y = 0
    w2_T_y = 0
    
#---------------For class 1------------------------------
    for itr in range(0,max_iter):
        a1 = 0
        for i in range(0,data.shape[0]):
            
            w1_T_y = np.dot(data[i,:],w_1)
            a1 = w1_T_y + b1
            if((a1*class1_labels[i]) <= 0):
                w_1 = w_1 + (class1_labels[i]*data[i,:])
                b1 = b1 + class1_labels[i]
                u1 = u1 + (class1_labels[i]*c1*data[i,:]) 
                Beta1 = Beta1 + (class1_labels[i]*c1)
                
            c1 = c1+1
            
    w_1 = w_1 -(u1/c1)
    b1 = b1 = (Beta1/c1)
                
#---------------For class 2------------------------------
    for itr in range(0,max_iter):
        a2 = 0
        for i in range(0,data.shape[0]):
            
            w2_T_y = np.dot(data[i,:],w_2)
            a2 = w2_T_y + b2
            if((a2*class2_labels[i]) <= 0):
                w_2 = w_2 + (class2_labels[i]*data[i,:])
                b2 = b2 + class2_labels[i]
                u2 = u2 + (class2_labels[i]*c2*data[i,:]) 
                Beta2 = Beta2 + (class2_labels[i]*c2)
                
            c2 = c2+1
    
    w_2 = w_2 - (u2/c2)
    b2 = b2 - (Beta2/c2)
            
    
    return {'class1':w_1,'bias1':b1,'class2':w_2,'bias2':b2}
                
    


stop_w = ['a','about','above','after','again','against','all','am','an','and','any','are',"aren't",'as','at','be','because','been','before','being','below','between','both','but','by',"can't",'cannot','could',"couldn't",'did',"didn't",'do','does',"doesn't",'doing',"don't",'down','during','each','few','for','from','further','had',"hadn't",'has',"hasn't",'have',"haven't",'having','he',"he'd","he'll","he's",'her','here',"here's",'hers','herself','him','himself','his','how',"how's",'i',"i'd","i'll","i'm","i've",'if','in','into','is',"isn't",'it',"it's",'its','itself',"let's",'me','more','most',"mustn't",'my','myself','no','nor','not','of','off','on','once','only','or','other','ought','our','ours','ourselves','out','over','own','same',"shan't",'she',"she'd","she'll","she's",'should',"shouldn't",'so','some','such','than','that',"that's",'the','their','theirs','them','themselves','then','there',"there's",'these','they',"they'd","they'll","they're","they've",'this','those','through','to','too','under','until','up','very','was',"wasn't",'we',"we'd","we'll","we're","we've",'were',"weren't",'what',"what's",'when',"when's",'where',"where's",'which','while','who',"who's",'whom','why',"why's",'with',"won't",'would',"wouldn't",'you',"you'd","you'll","you're","you've",'your','yours','yourself','yourselves']
punctuation = string.punctuation


stop_words = np.array(stop_w)
word_freq = dict()
N_doc = 0
Corpus = []
fhand = open(sys.argv[1])


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
    
################Counting frequencies of each word and making an array of all words that occur less than a certain times############################
#    
#for i in range(0,len(Corpus)):
#    count = 0
#    for word in Corpus[i].split():
#        if(count>=3):
#            word_freq[word] = word_freq.get(word,0) + 1
#        
#        count = count+1
#        
#
#low_freq_w = []
#
#for key in word_freq:
#    if(word_freq.get(key) <= 1):
#        if((key != "pos") and (key != "neg") and (key != "true") and (key != "fake")):
#            low_freq_w.append(key)
#        
#low_freq_words = np.array(low_freq_w)
#
#for i in range(0,len(Corpus)):
#    
#    doc = Corpus[i]
#    updated_doc = ""
#    for word in doc.split():
#        if(word not in low_freq_words):
#            updated_doc = updated_doc + word + " "
#        
#    Corpus[i] = updated_doc
#    
    
#------------------------Creating Data-----------------------------------------
    
#Shuffle the data--------------------------------------------------------------



class1_labels = []
class2_labels = []
which_index = 0
word_Vocab_index = dict()
Corpus_dictionary = dict()
for i in range(len(Corpus)):
    count = 0
    tag = ''
    sentence_word_count = dict()
    for word in Corpus[i].split():

        if(count == 0):
            tag = word
        elif(count == 1):
            if(word == 'fake'):
                class1_labels.append(-1)
            elif(word == 'true'):
                class1_labels.append(1)
        elif(count == 2):
            if(word == 'neg'):
                class2_labels.append(-1)
            elif(word == 'pos'):
                class2_labels.append(1)    
        else:
            sentence_word_count[word] = sentence_word_count.get(word,0) + 1
            if(word_Vocab_index.get(word) == None):
                word_Vocab_index[word] = which_index
                which_index = which_index+1
            
            
        count = count + 1
    
    Corpus_dictionary[tag_list[i]] = sentence_word_count
 
data = np.zeros((len(Corpus),len(word_Vocab_index)))

for i in range(len(tag_list)):
    
    d1 = Corpus_dictionary[tag_list[i]]
    for keys in d1:
        word = keys
        index = word_Vocab_index.get(word)
        data[i,index] = d1.get(keys)
        
        
vanilla_weight_vector = VanillaPerceptron(data,class1_labels,class2_labels,50)
Averaged_weight_vector = AveragedPerceptron(data,class1_labels,class2_labels,50)

fout1 = open('vanillamodel.txt','w')
#fout2 = open('averagedmodel.txt','w')
fout1.write('bias1: '+str(1)+'\n')
fout1.write('bias2: '+str(1)+'\n')
fout1.write('vanilla_weight_vector_class1_length: '+str(len(vanilla_weight_vector['class1']))+'\n')
fout1.write('vanilla_weight_vector_class2_length: '+str(len(vanilla_weight_vector['class2']))+'\n')
fout1.write('word_Vocab_index: '+str(len(word_Vocab_index))+'\n')

fout1.write(str(vanilla_weight_vector['bias1'])+'\n')
fout1.write(str(vanilla_weight_vector['bias2'])+'\n')
for i in range(len(vanilla_weight_vector['class1'])):
    fout1.write(str(vanilla_weight_vector['class1'][i])+' ')
    
fout1.write('\n')
for i in range(len(vanilla_weight_vector['class2'])):
    fout1.write(str(vanilla_weight_vector['class2'][i])+' ')

fout1.write('\n')

for keys in word_Vocab_index:
   fout1.write(keys+' '+str(word_Vocab_index[keys])+'\n')
    
fout1.close()



fout2 = open('averagedmodel.txt','w')
fout2.write('bias1: '+str(1)+'\n')
fout2.write('bias2: '+str(1)+'\n')
fout2.write('Averaged_weight_vector_class1_length: '+str(len(Averaged_weight_vector['class1']))+'\n')
fout2.write('Averaged_weight_vector_class2_length: '+str(len(Averaged_weight_vector['class2']))+'\n')
fout2.write('word_Vocab_index: '+str(len(word_Vocab_index))+'\n')

fout2.write(str(Averaged_weight_vector['bias1'])+'\n')
fout2.write(str(Averaged_weight_vector['bias2'])+'\n')
for i in range(len(Averaged_weight_vector['class1'])):
    fout2.write(str(Averaged_weight_vector['class1'][i])+' ')
    
fout2.write('\n')
for i in range(len(Averaged_weight_vector['class2'])):
    fout2.write(str(Averaged_weight_vector['class2'][i])+' ')
    
fout2.write('\n')
for keys in word_Vocab_index:
   fout2.write(keys+' '+str(word_Vocab_index[keys])+'\n')

fout2.close()
        
        
    
    

