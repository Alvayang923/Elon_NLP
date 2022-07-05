# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 23:38:04 2022

@author: alana
"""

import nltk
global dictionary
dictionary = nltk.corpus.words.words("en")
dictionary = [word.lower() for word in dictionary]

def fast_dictionary(df_in, col_name):
    # word dictionaries for each topic/label
    from collections import Counter
    word_freq = dict()
    for topic in df_in.label.unique():
        word_freq[topic] = Counter(
            df_in[col_name][
                df_in.label == topic].str.cat().split())
    return word_freq

def pre_process_text_new(tmp_f):
    tmp_f = clean_txt(tmp_f)
    tmp_f = [word_t.lower() for word_t in tmp_f.split(
        ) if word_t in dictionary]
    tmp_f = ' '.join(tmp_f)
    return tmp_f

def pre_process_text(tmp_f, sw):
    tmp_f = clean_txt(tmp_f)
    if sw == "check_words":
        tmp_f = [word_t.lower() for word_t in tmp_f.split(
            ) if word_t in dictionary]
        tmp_f = ' '.join(tmp_f)
    return tmp_f

def file_seeker(path_in, sw):
    import os
    import pandas as pd
    tmp_pd = pd.DataFrame()
    for dirName, subdirList, fileList in os.walk(path_in):
        tmp_dir_name = dirName.split("/")[-1::][0]
        try:
            for word in fileList:
                #print(word)
                tmp = open(dirName+"/"+word, "r", encoding="ISO-8859-1")
                tmp_f = tmp.read()
                tmp.close()
                if sw == "check_words":
                    tmp_f = [word_t.lower() for word_t in tmp_f.split(
                        ) if word_t in dictionary]
                    tmp_f = ' '.join(tmp_f)
                cln_txt = clean_txt(tmp_f)
                if len(cln_txt) != 0:
                    tmp_pd = tmp_pd.append({'label': tmp_dir_name,                                    
                                            #'path': dirName+"/"+word}, 
                                            'body': tmp_f,
                                            'body_cleaned': cln_txt},
                                            ignore_index = True)
        except Exception as e:
            print (e)
            pass
    return tmp_pd

def clean_txt(txt_in):
    import re
    clean_str = re.sub("[^A-Za-z]+", " ", txt_in).strip().lower()
    return clean_str

def word_count(var_i):
    tmp = len(var_i.split())
    return tmp 

def word_unique_count(var_i):
    tmp = len(set(var_i.split()))
    return tmp

def open_pickle(path_in, file_name):
    import pickle
    tmp = pickle.load(open(path_in + file_name, "rb"))
    return tmp

def write_pickle(path_in, file_name, var_in):
    import pickle
    pickle.dump(var_in, open(path_in + file_name, "wb"))
    
def my_stem(var_in):
    from nltk.stem.porter import PorterStemmer
    my_stem = PorterStemmer()
    tmp = [my_stem.stem(word) for word in var_in.split()]
    tmp = ' '.join(tmp)
    return tmp

def word_predict(corpus_in, word_in):
    count_t = corpus_in.split().count(word_in)
    total_words = len(corpus_in.split())
    the_o = None
    try:
        the_o = round((count_t / total_words)*100, 2)
    except:
        pass
    return the_o

def my_stop_words(var_in):
    from nltk.corpus import stopwords
    sw = stopwords.words('english')
    tmp = [word for word in var_in.split() if word not in sw]
    tmp = ' '.join(tmp)
    return tmp

def extract_embeddings_pre(df_in, num_vec_in, path_in, filename):
    from gensim.models import Word2Vec
    import pandas as pd
    from gensim.models import KeyedVectors
    import pickle
    my_model = KeyedVectors.load_word2vec_format(filename, binary=True) 
    my_model = Word2Vec(df_in.str.split(),
                        min_count=1, vector_size=num_vec_in)#size=300)
    word_dict = my_model.wv.key_to_index
    #my_model.similarity("trout", "fish")
    def get_score(var):
        try:
            import numpy as np
            tmp_arr = list()
            for word in var:
                tmp_arr.append(list(my_model.wv[word]))
        except:
            pass
        return np.mean(np.array(tmp_arr), axis=0)
    tmp_out = df_in.str.split().apply(get_score)
    tmp_data = tmp_out.apply(pd.Series).fillna(0)
    pickle.dump(my_model, open(path_in + "embeddings.pkl", "wb"))
    pickle.dump(tmp_data, open(path_in + "embeddings_df.pkl", "wb" ))
    return tmp_data

def extract_embeddings_domain(df_in, num_vec_in, path_in):
    #domain specific, train out own model specific to our domains
    from gensim.models import Word2Vec
    import pandas as pd
    import numpy as np
    import pickle
    model = Word2Vec(
        df_in.str.split(), min_count=1,
        vector_size=num_vec_in, workers=3, window=5, sg=0)
    #wrd_dict = model.wv.key_to_index
    #model.wv.most_similar('machine', topn=10)
    def get_score(var):
        try:
            tmp_arr = list()
            for word in var:
                tmp_arr.append(list(model.wv[word]))
        except:
            pass
        return np.mean(np.array(tmp_arr), axis=0)
    tmp_out = df_in.str.split().apply(get_score)
    tmp_data = tmp_out.apply(pd.Series).fillna(0)
    #model.wv.save_word2vec_format(base_path + "embeddings_domain.pkl")
    pickle.dump(model, open(path_in + "embeddings_domain_model.pkl", "wb"))
    pickle.dump(tmp_data, open(path_in + "embeddings_df_domain.pkl", "wb" ))
    return tmp_data

def cosine_fun(df_a, df_b, label_in): 
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    cos_matrix_t = pd.DataFrame(cosine_similarity(df_a, df_b))
    cos_matrix_t.index = label_in
    cos_matrix_t.columns = label_in
    return cos_matrix_t

def my_cos_sim(x_in, y_in):
    import numpy as np
    num_t = sum(x_in*y_in)
    den_t = np.linalg.norm(x_in)*np.linalg.norm(y_in)
    return num_t / den_t

def my_bow(df_in, path_in, gram_m, gram_n, sw, name_in):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd
    if sw == "tf-idf":
        my_cv = TfidfVectorizer(ngram_range=(gram_m, gram_n))
    else:
        my_cv = CountVectorizer(ngram_range=(gram_m, gram_n))
    my_cv_data = pd.DataFrame(my_cv.fit_transform(df_in).toarray())
    col_names = list(my_cv.vocabulary_.keys())
    my_cv_data.columns = col_names
    write_pickle(path_in, name_in, my_cv)
    return my_cv_data

def my_pca(df_in, path_in, file_name, exp_var_in):
    from sklearn.decomposition import PCA
    my_pca = PCA(n_components=exp_var_in)#, svd_solver='full')
    my_pca_data = my_pca.fit_transform(df_in)
    exp_var = sum(my_pca.explained_variance_ratio_)
    print ("Explained variance is:", exp_var)
    write_pickle(path_in, file_name, my_pca)
    return my_pca_data

def fetch_bi_grams(var):
    import numpy as np
    from gensim.models import Phrases
    from gensim.models.phrases import Phraser
    sentence_stream = np.array(var)
    bigram = Phrases(sentence_stream, min_count=5, threshold=10, delimiter=",")
    trigram = Phrases(bigram[sentence_stream], min_count=5, threshold=10)
    bigram_phraser = Phraser(bigram)
    trigram_phraser = Phraser(trigram)
    bi_grams = list()
    tri_grams = list()
    for sent in sentence_stream:
        bi_grams.append(bigram_phraser[sent])
        tri_grams.append(trigram_phraser[sent])
    return bi_grams, tri_grams

def lda_fun(path_in, the_data_t):
    import gensim
    from gensim.corpora.dictionary import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
    import gensim.corpora as corpora
    import matplotlib.pyplot as plt
    from kneed import KneeLocator
    
    bi, tri = fetch_bi_grams(the_data_t.str.split())

    the_data = tri

    dictionary = Dictionary(the_data)
    id2word = corpora.Dictionary(the_data)
    
    corpus = [id2word.doc2bow(text) for text in the_data]
    
    n_topics = 8
    ldamodel = gensim.models.ldamodel.LdaModel(
        corpus, num_topics=n_topics, id2word=id2word, iterations=50, passes=15,
        random_state=123)
    ldamodel.save('model5.gensim')
    topics = ldamodel.print_topics(num_words=4)
    for topic in topics:
        print(topic)
        
    #compute Coherence Score using c_v
    coherence_model_lda = CoherenceModel(
        model=ldamodel, texts=the_data,
        dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
    
    c_scores = list()
    for word in range(1, 8):
        print (word)
        ldamodel = gensim.models.ldamodel.LdaModel(
            corpus, num_topics=word, id2word=id2word, iterations=10, passes=5,
            random_state=123)
        coherence_model_lda = CoherenceModel(model=ldamodel, texts=the_data,
                                              dictionary=dictionary,
                                              coherence='c_v')
        c_scores.append(coherence_model_lda.get_coherence())
    
    x = range(1, 8)
    #https://pypi.org/project/kneed/
    kn = KneeLocator(x, c_scores,
                     curve='convex', direction='increasing')
    opt_topics = kn.knee
    print ("Optimal topics is", opt_topics)
    plt.plot(x, c_scores)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    return 0

def my_model_fun(parameters, df_in, label_in, path_in):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_fscore_support
    my_model = RandomForestClassifier()
    
    my_grid_search = GridSearchCV(
        my_model, param_grid=parameters, cv=5)
    my_grid_search.fit(df_in, label_in)
    print ("Best Score:", my_grid_search.best_score_,
           "Best Params:",my_grid_search.best_params_)
    
    my_model = RandomForestClassifier(**my_grid_search.best_params_)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df_in, label_in, test_size=0.20, random_state=42)
    my_model.fit(X_train, y_train)
    y_pred = my_model.predict(X_test)
    prec, recall, acc, y = precision_recall_fscore_support(
        y_test, y_pred, average='weighted')
    print ("precision:",prec,"recall:",recall)
    write_pickle(path_in, "model.pkl", my_model)
    return my_model

#pip install 'gensim==4.1.2'
#pip install kneed