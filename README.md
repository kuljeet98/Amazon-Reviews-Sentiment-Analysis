# Amazon-Reviews-Sentiment-Analysis


Importing  <<<In Progress>>

## Dataset

![data](https://user-images.githubusercontent.com/43536129/46906312-10f18680-cf1f-11e8-9b54-d60835da25df.PNG)
 * Dataset taken from Kaggel ([View](https://www.kaggle.com/snap/amazon-fine-food-reviews))
  #### Data Cleaning
 - Duplicate enteries removal
 - Incorrect cases removal (Example: Helpfullness numerator > Helpfullness denominator)
 - Cleaning HTML tags from the text reviews of customers
 
 Checking for HTML tags
 ``` python
import re
i=0
for sentence in final_data['Text'].values:
    if (len(re.findall('<.*?>', sentence))):
        print("index : ",i)
        print(sentence)
        break;
    i = i+1
```

 Removing HTML tags
 ``` python
# html tags cleaner
def cleanhtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext
# punctuation cleaner
def cleanpunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]',r' ',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return  cleaned
```

- Snowball stemmer used for stemming (Was observered to be better than Porter stemmer for the task)
``` python
# Computation

i=0
posi_words = []
negi_words = []
str1=' '
final_string=[]
for sent in final_data['Text'].values:
    filtered_sent = []
    sent = cleanhtml(sent)
    for w in sent.split():
        for cleaned_word in cleanpunc(w).split():
            if (cleaned_word.isalpha() and len(cleaned_word)>2):
                if(cleaned_word.lower() not in stop):
                    s = (sno.stem(cleaned_word.lower())).encode('utf8')
                    filtered_sent.append(s)
                    if(final_data['Score'].values[i] == 'positive'):
                        posi_words.append(s)
                    elif(final_data['Score'].values[i] == 'negative'):
                        negi_words.append(s)
                else:
                    continue
            else:
                continue
    str1 = b" ".join(filtered_sent)
    final_string.append(str1)
    i+=1
```

### Analysis

For the task of data analysis, TSNE and PCA techniques were considered. However, TSNE rendered better plots.
(Random sampling was done to reduce number of points to a smaller number due RAM constraints for processing 500k data points)
Some samples are shown below but for complete TSNE and PCA analysis with and without Truncates SVD - [Here](https://github.com/prabhnoor0212/Amazon-Reviews-Sentiment-Analysis/blob/master/analysis.pdf)
#### Some key observations from these plots
- All the plots showed high overlaping between positive and negative points.
- TruncatedSVD plots allowed more data points to be considered but at the cost of relatively poor plot quality
- BOW representation showed two clusters : one being heavily populated by positive and negative points and the second was fairly spaced out with almost equal population among both the classes
- Although, none of the plots showed signs of a simple plane seperation but, word2vec plots showed the best results among BOW, tf-idf and word2vec techniques


### ML modelling and Sample Results / Discussion
For processing the text, 4 techniques (Bag of Words, tf-idf, average word to vec, tfidf weighted word2vec)
(refer ipython notebooks for complete analysis with various hyper parameters and more algorithms)

Note: As the dataset is imbalanced (1:8 negative to positive reviews), therefore, accuracy might be quite high but it is not the best metric for evaluation. Confusion Matrix & TPR, FPR, TNR, FNR, Precision values provide a much better picture.

#### Bag of words sample observations

TSNE

![bag of words](https://user-images.githubusercontent.com/43536129/46907878-473b0000-cf37-11e8-8ad3-0491fe39d5ce.PNG)

Train Accuracy : 88.13%

Test Accuracy : 85%

Confusion Matrix


![cmatbow](https://user-images.githubusercontent.com/43536129/46908099-3096a800-cf3b-11e8-9daa-5b65447e90c1.PNG)

#### tf-idf sample observations

TSNE

![tfidftsne](https://user-images.githubusercontent.com/43536129/46907944-8584ef00-cf38-11e8-8466-32fac4ee5d3e.PNG)


Train Accuracy : 88.6%

Test Accuracy : 86%

Confusion Matrix


![cmattfidf](https://user-images.githubusercontent.com/43536129/46908140-d34f2680-cf3b-11e8-9417-ca7efaaa7440.PNG)

#### Average Word2Vec sample observations

TSNE

![avgw2v](https://user-images.githubusercontent.com/43536129/46907963-dbf22d80-cf38-11e8-9223-7a3f43f19bab.PNG)

Train Accuracy : 89.94%

Test Accuracy : 88.56%

Confusion Matrix


![cmatavgw2v](https://user-images.githubusercontent.com/43536129/46915736-3f2d9f80-cfcd-11e8-840b-6be43033d3b6.PNG)


Thus, Word2Vec technique worked much better than the BOW and tfidf for the imbalanced dataset.

#### tfidf weighted Word2Vec sample observations

TSNE

![tfidfweightedw2v](https://user-images.githubusercontent.com/43536129/46907991-43a87880-cf39-11e8-90dd-162e2cfe6af0.PNG)


Note: Due to imbalanced datasets the False Positive Rate is quite high. To counter this problem many options were tried - (under sampling, oversampling, weighted effects, etc.). As we had a heavy dataset of 500k points, undersampling proved viable.

DISCUSSION: Task of building of vectorizers should be strictly done on the training data. The test data shouldn't be used for fitting the vectorizers and should be used at the time of testing only to avoid data leakage probelm.

FULL RESULTS: Refer ipython notebooks
