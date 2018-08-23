import sys
import numpy as np
import string

stop_w = ['a','about','above','after','again','against','all','am','an','and','any','are',"aren't",'as','at','be','because','been','before','being','below','between','both','but','by',"can't",'cannot','could',"couldn't",'did',"didn't",'do','does',"doesn't",'doing',"don't",'down','during','each','few','for','from','further','had',"hadn't",'has',"hasn't",'have',"haven't",'having','he',"he'd","he'll","he's",'her','here',"here's",'hers','herself','him','himself','his','how',"how's",'i',"i'd","i'll","i'm","i've",'if','in','into','is',"isn't",'it',"it's",'its','itself',"let's",'me','more','most',"mustn't",'my','myself','no','nor','not','of','off','on','once','only','or','other','ought','our','ours','ourselves','out','over','own','same',"shan't",'she',"she'd","she'll","she's",'should',"shouldn't",'so','some','such','than','that',"that's",'the','their','theirs','them','themselves','then','there',"there's",'these','they',"they'd","they'll","they're","they've",'this','those','through','to','too','under','until','up','very','was',"wasn't",'we',"we'd","we'll","we're","we've",'were',"weren't",'what',"what's",'when',"when's",'where',"where's",'which','while','who',"who's",'whom','why',"why's",'with',"won't",'would',"wouldn't",'you',"you'd","you'll","you're","you've",'your','yours','yourself','yourselves']
punctuation = string.punctuation


stop_words = np.array(stop_w)
word_freq = dict()
N_doc = 0
Corpus = []
fhand = open('train-labeled.txt')

#############Converting to lower case and then Removing stop words ###########

for line in fhand:
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
    
for i in range(0,len(Corpus)):
    count = 0
    for word in Corpus[i].split():
        if(count>=3):
            word_freq[word] = word_freq.get(word,0) + 1
        
        count = count+1
        

low_freq_w = []

for key in word_freq:
    if(word_freq.get(key) <= 1):
        if((key != "pos") and (key != "neg") and (key != "true") and (key != "fake")):
            low_freq_w.append(key)
        
low_freq_words = np.array(low_freq_w)

for i in range(0,len(Corpus)):
    
    doc = Corpus[i]
    updated_doc = ""
    for word in doc.split():
        if(word not in low_freq_words):
            updated_doc = updated_doc + word + " "
        
    Corpus[i] = updated_doc
    
#print(Corpus)
#print()
#print()
#print()

###################Creating big Doc(That is all docs of class C together)

bigdoc_pos = dict()
bigdoc_neg = dict()
bigdoc_true = dict()
bigdoc_fake = dict()
Corpus_pos = []
Corpus_neg = []
Corpus_true = []
Corpus_fake = []

for i in range(0,len(Corpus)):
    count = 0
    for word in Corpus[i].split():
        if(count<3):
            if(count == 1):
                class1 = word
            elif(count == 2):
                class2 = word
        count = count + 1
        if(count == 3):
            break
        
    
    if(class1=="true"):
        Corpus_true.append(Corpus[i][16:len(Corpus[i])])
    elif(class1=="fake"):
        Corpus_fake.append(Corpus[i][16:len(Corpus[i])])
        
    if(class2=="pos"):
        Corpus_pos.append(Corpus[i][16:len(Corpus[i])])
    elif(class2=="neg"):
        Corpus_neg.append(Corpus[i][16:len(Corpus[i])])
        
bigdoc_pos["pos"] = Corpus_pos
bigdoc_neg["neg"] = Corpus_neg
bigdoc_true["true"] = Corpus_true
bigdoc_fake["fake"] = Corpus_fake
        
        
##############Counting words in class C and also the Vocabulary length

pos_word_count = 0
neg_word_count = 0
true_word_count = 0
fake_word_count = 0  
word_given_pos_count = dict()
word_given_neg_count = dict()
word_given_true_count = dict()
word_given_fake_count = dict() 
Vocabulary = set()     
#-------------------Total pos word count-----------------------------

for i in range(0,len(Corpus_pos)):
    
    for word in Corpus_pos[i].split():
        pos_word_count = pos_word_count + 1
        word_given_pos_count[word] = word_given_pos_count.get(word,0) + 1
        Vocabulary.add(word)
        
for i in range(0,len(Corpus_neg)):
    
    for word in Corpus_neg[i].split():
        neg_word_count = neg_word_count + 1
        word_given_neg_count[word] = word_given_neg_count.get(word,0) + 1
        Vocabulary.add(word)
        
for i in range(0,len(Corpus_true)):
    
    for word in Corpus_true[i].split():
        true_word_count = true_word_count + 1
        word_given_true_count[word] = word_given_true_count.get(word,0) + 1
        Vocabulary.add(word)
        
for i in range(0,len(Corpus_fake)):
    
    for word in Corpus_fake[i].split():
        fake_word_count = fake_word_count + 1
        word_given_fake_count[word] = word_given_fake_count.get(word,0) + 1
        Vocabulary.add(word)


##############################Calculate parameters###########################

if(len(Corpus_pos) != 0):
    Log_Prior_pos = np.log10(len(Corpus_pos)/N_doc)
else:
    Log_Prior_pos = 0

if(len(Corpus_neg) != 0):
    Log_Prior_neg = np.log10(len(Corpus_neg)/N_doc)
else:
    Log_Prior_neg = 0
    
if(len(Corpus_true) != 0):
    Log_Prior_true = np.log10(len(Corpus_true)/N_doc)
else:
    Log_Prior_true = 0
    
if(len(Corpus_fake) != 0):
    Log_Prior_fake = np.log10(len(Corpus_fake)/N_doc)
else:
    Log_Prior_fake = 0
##print(len(Corpus_pos)," ",N_doc)
#Log_Prior_neg = np.log10(len(Corpus_neg)/N_doc)
##print(len(Corpus_neg)," ",N_doc)
#Log_Prior_true = np.log10(len(Corpus_true)/N_doc)
##print(len(Corpus_true)," ",N_doc)
#Log_Prior_fake = np.log10(len(Corpus_fake)/N_doc)
##print(len(Corpus_fake)," ",N_doc)

Log_word_given_class = dict()
        
for word in Vocabulary:
    Log_word_given_class[(word,"pos")] = np.log10((word_given_pos_count.get(word,0)+1)/(pos_word_count + len(Vocabulary)))
    Log_word_given_class[(word,"neg")] = np.log10((word_given_neg_count.get(word,0)+1)/(neg_word_count + len(Vocabulary)))
    Log_word_given_class[(word,"true")] = np.log10((word_given_true_count.get(word,0)+1)/(true_word_count + len(Vocabulary)))   
    Log_word_given_class[(word,"fake")] = np.log10((word_given_fake_count.get(word,0)+1)/(fake_word_count + len(Vocabulary)))
    
            
fout = open('nbmodel.txt','w')
fout.write('Log_Prior_length: '+str(4)+'\n')
fout.write('Vocabulary_length: '+str(len(Vocabulary))+'\n')
fout.write('Word_given_class_prob_length: '+str(len(Log_word_given_class))+'\n')
fout.write(str(Log_Prior_pos)+'\n')
fout.write(str(Log_Prior_neg)+'\n')
fout.write(str(Log_Prior_true)+'\n')
fout.write(str(Log_Prior_fake)+'\n')

v = ""
for word in Vocabulary:
    v = v+word+" "
    
fout.write(v+'\n')
    
for key in Log_word_given_class:
    fout.write(key[0]+" "+key[1]+" "+str(Log_word_given_class.get(key))+'\n')

fout.close()
           
                
                
                




    
                
            
            
        
    
    



