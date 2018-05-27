# -*- coding: utf-8 -*-
#!/usr/bin/env python
import string
import collections
from sklearn import metrics
from  requests import ConnectionError
from pylab import *

import xml
import json
from xml.dom import minidom
from urllib2 import HTTPError
import requests
import gensim
from heapq import nlargest
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

url='http://ws.clarin-pl.eu/nlprest2/base/process'
tool="any2txt|wcrft2"
mail="epsonik@o2.pl"
headers = {'content-type': 'application/json'}

class Osf_Initialize():
    processed_data=pd.DataFrame
    data = pd.DataFrame
    data_sekcja = pd.DataFrame
    data_clean = pd.DataFrame
    data_model = None
    sekcja_opis_projektu_cel_naukowy = pd.DataFrame()
    sekcja_opis_projektu_efekt = pd.DataFrame()
    sekcja_opis_projektu_literatura = pd.DataFrame()
    sekcja_opis_projektu_metodyka_badan = pd.DataFrame()
    sekcja_opis_projektu_stan_wiedzy = pd.DataFrame()
    sekcja_opis_projektu_znaczenie_projektu = pd.DataFrame()

    opis_sekcje=["opisProjektuCelNaukowy", "opisProjektuEfekt", "opisProjektuLiteratura",
    "opisProjektuMetodykaBadan", "opisProjektuStanWiedzy","opisProjektuZnaczenieProjektu"]

    frame_structure = ["ID", "NAZWASEKCJI", "WARTOSC",
                       "DATA_WPLYWU", "WI_KONKURS_ID", "TYTUL", "SLOWA_KLUCZE",
                       "WI_STATUSWNIOSKU_KOD"]
    def initialize(self,csv_link):
        self.data = pd.read_csv(csv_link, encoding='utf-8', header=0)
        self.data_model = pd.read_csv(csv_link, encoding='utf-8', header=0)
        self.processed_data=self.data
    def initialize_sekcja(self,sekcja_link,tup,nr_rows=120,create_structure=False):
        if create_structure:
            self.create_structure(tup)
        if sekcja_link:
            self.data_sekcja = pd.read_csv(sekcja_link, encoding='utf-8', header=0)
            self.processed_data=self.data_sekcja.head(nr_rows)
    def create_structure(self,tup):

            def create_structure_row(row):
                def parse_opis():
                    col_structure = [row[x] for x in self.frame_structure]
                    s = pd.DataFrame([list(col_structure)], columns=self.frame_structure)
                    return s

                if row["NAZWASEKCJI"] == "opisProjektuCelNaukowy":
                    self.sekcja_opis_projektu_cel_naukowy = self.sekcja_opis_projektu_cel_naukowy.append(parse_opis(),
                                                                                                         ignore_index=True)
                    return row
                if row["NAZWASEKCJI"] == "opisProjektuEfekt":
                    self.sekcja_opis_projektu_efekt = self.sekcja_opis_projektu_efekt.append(parse_opis(),
                                                                                             ignore_index=True)
                    return row
                if row["NAZWASEKCJI"] == "opisProjektuLiteratura":
                    self.sekcja_opis_projektu_literatura = self.sekcja_opis_projektu_literatura.append(parse_opis(),
                                                                                                       ignore_index=True)
                    return row
                if row["NAZWASEKCJI"] == "opisProjektuMetodykaBadan":
                    self.sekcja_opis_projektu_metodyka_badan = self.sekcja_opis_projektu_metodyka_badan.append(
                        parse_opis(),
                        ignore_index=True)
                    return row
                if row["NAZWASEKCJI"] == "opisProjektuStanWiedzy":
                    self.sekcja_opis_projektu_stan_wiedzy = self.sekcja_opis_projektu_stan_wiedzy.append(parse_opis(),
                                                                                                         ignore_index=True)
                    return row
                if row["NAZWASEKCJI"] == "opisProjektuZnaczenieProjektu":
                    self.sekcja_opis_projektu_znaczenie_projektu = self.sekcja_opis_projektu_znaczenie_projektu.append(
                        parse_opis(),
                        ignore_index=True)
                    return row

            self.processed_data = self.processed_data.apply(create_structure_row, axis=1)
            data.sekcja_opis_projektu_znaczenie_projektu.to_csv('opisProjektuZnaczenieProjektu' +tup+ ".csv",
                                                                encoding='utf8')
            data.sekcja_opis_projektu_stan_wiedzy.to_csv('opisProjektuStanWiedzy' +tup+ ".csv", encoding='utf8')
            data.sekcja_opis_projektu_metodyka_badan.to_csv('opisProjektuMetodykaBadan' +tup+ ".csv", encoding='utf8')
            data.sekcja_opis_projektu_literatura.to_csv('opisProjektuLiteratura' +tup+ ".csv", encoding='utf8')
            data.sekcja_opis_projektu_efekt.to_csv('opisProjektuEfekt' +tup+ ".csv", encoding='utf8')
            data.sekcja_opis_projektu_cel_naukowy.to_csv('opisProjektuCelNaukowy' +tup+ ".csv", encoding='utf8')
class Cleansing(Osf_Initialize):

    def __init__(self, previous):
        self.processed_data = previous.processed_data

    def clean(self):
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'<[^>]+>', '')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&nbsp;', '')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&oacute;', u'Ã³')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&ndash;', u'-')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&ldquo;', u'"')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&bdquo;', u'"')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&quot;', u'"')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&rdquo;', u'"')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&amp;', u'&')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&ouml;', u'o')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&uuml;', u'u')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&auml;', u'a')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&bull;', u'â¢')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&middot;', u'â¢')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&deg;', '')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&rsquo;', u'\'')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace('\\', '')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'/', '')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&shy;', '')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&eacute;', u'e')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&uacute;', u'u')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&aacute;', u'a')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&gt;', u'>')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&agrave;', u'a')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&alpha;', u'alpha')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&mdash;', u'-')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&beta;', u'beta')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&mu;', u'mikro')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&gamma;', u'gamma')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&omega;', u'omega')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&tau;', u'tau')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&sum', '')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&lt', u'<')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&reg', '')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&times;', u'x')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&not;', '')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&nbs;', '')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&nbs;', '')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.strip()
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.lower()
        return
    def clean_not_printable(self):
        polish = u"ąęćżźńłóś"
        def clean_not_printable_row(row):
            if type(row.WARTOSC) != float and row.WARTOSC:
                string.printable = string.printable + polish
                row.WARTOSC = filter(lambda x: x in string.printable, row.WARTOSC)
                return row
        self.processed_data = self.processed_data.apply(clean_not_printable_row, axis=1)

class OsfData_Lemmatization(Cleansing):

    def __init__(self, previous):
        self.processed_data = previous.processed_data
    def lemmatize(self, tup):
        chunk = "chunk"
        sentence = "sentence"
        tok = "tok"
        lex = "lex"
        base = "base"
        ctag = "ctag"
        leksem_noun = {'subst', 'depr'}
        leksem_verb = {'fin', 'bedzie', 'aglt', 'praet', 'impt', 'imps',
                       'inf', 'pcon', 'pant', 'ger', 'pact', 'ppas'}
        leksem_adjective = {'adj', 'adja', 'adjp'}
        leksem_others = {'winien', 'qub'}
        set = leksem_noun.union(leksem_verb, leksem_adjective, leksem_others)
        def createURL(text):
            data = {"lpmn": tool,
                    "text": text,
                    "user": mail
                    }
            return data
        def lemmatize_row(row):
            if type(row.WARTOSC) != float and row.WARTOSC:
                try:
                    r = requests.post(url, data=json.dumps(createURL(row.WARTOSC.encode('utf8'))),
                                      headers=headers)
                    if r.status_code == 200:
                        row["LEMMATIZED"] =parseXML(r.text,row)
                        return row
                except (HTTPError, ConnectionError):
                    print 'Failed to open %s' % (row.WARTOSC)
        def parseXML(resp,row):
            try:
                parsed = minidom.parseString(resp.encode('utf-8'))
                cNodes = parsed.childNodes
                description=list()
                for i in cNodes[1].getElementsByTagName(chunk):
                    for k in i.getElementsByTagName(sentence)[0].getElementsByTagName(tok):
                        path = k.getElementsByTagName(lex)[0]
                        path_base = path.getElementsByTagName(base)[0].firstChild.toxml()
                        path_ctag = path.getElementsByTagName(ctag)[0].firstChild.toxml()
                        if any(x in path_ctag for x in set):
                            description.append(path_base)
                b=" ".join(description)
                return b
            except xml.parsers.expat.ExpatError, e:
                print "ERROR "
                print e
                print row.ROWNUM
                print resp

        self.processed_data = self.processed_data.apply(lemmatize_row, axis=1)
        data.processed_data.to_csv('opisProjektuEfektExportClean' +tup+ ".csv", encoding='utf8')
class OsfData_Wordlist():
    dictionary=[]
    corpus = []
    tf_idf = []
    sims = []
    mtx= pd.DataFrame()
    k = list()
    a = pd.DataFrame
    lemmatized= "LEMMATIZED"
    lemmatized2= "LEMMATIZED2"
    type = "TYPE"
    stat = "STAT"
    klasa_obiektu = "KLASA_OBIEKTU"
    def create_mtx(self):
        def read(x):
            csv_link = "opisProjektuEfektExportClean%s.csv" % x
            a = pd.read_csv(csv_link, encoding='utf-8', header=0)
            a[self.type] = pd.Series(len(a) * [x])
            return a
        self.processed_data=pd.concat([read(x) for x in range(1,4)],ignore_index=True)
        self.processed_data = self.processed_data.reindex(np.random.permutation(self.processed_data.index))
    def build_wordlist(self):
        from nltk.tokenize import word_tokenize
        def build_wordlist_row(row):
            if type(row[self.lemmatized]) != float and row[self.lemmatized]:
                try:
                    row[self.lemmatized2] = word_tokenize(row[self.lemmatized])
                    return row
                except TypeError:
                    print row[self.lemmatized]
                    print 1
            else:
                print "jolo"
        self.processed_data = self.processed_data.apply(build_wordlist_row, axis=1)
    def create_dictionary(self):
        gen_docs=self.processed_data[self.lemmatized2].tolist()
        try:
            self.dictionary = gensim.corpora.Dictionary(gen_docs)
            self.corpus = [self.dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
        except (TypeError,AttributeError) as e:
            print e
            print 2

        self.tf_idf = gensim.models.TfidfModel(self.corpus)
        self.sims = gensim.similarities.Similarity('/home/mbarto/PycharmProjects/osf',
                                                   self.tf_idf[self.corpus],
                                                   num_features=len(self.dictionary))
    def find_sims_row(self,row):#find sims of row to every in set
        gen_doc =row[self.lemmatized2]
        try:
            query_doc_bow = self.dictionary.doc2bow(gen_doc)
            query_doc_tf_idf = self.tf_idf[query_doc_bow]
            a=self.sims[query_doc_tf_idf]
            return a
        except AttributeError as e:
            print e
    def find_sims(self):
        def find_sims_row(row):
            gen_doc =row[self.lemmatized2]
            try:
                query_doc_bow = self.dictionary.doc2bow(gen_doc)
                query_doc_tf_idf = self.tf_idf[query_doc_bow]
                a=self.sims[query_doc_tf_idf]
                self.k.append(a)
                return row
            except AttributeError as e:
                print e
                print 3
        self.processed_data = self.processed_data.apply(find_sims_row, axis=1)
    def find_n_max(self,idx):
        b = self.find_sims_row(self.processed_data.iloc[idx])
        m = nlargest(2, xrange(len(b)), key=b.__getitem__)
        return self.processed_data.iloc[m[1]]

    def prepare_sp_mtx(self):
        lenm=len(self.processed_data.index)
        self.mtx = pd.DataFrame(columns=range(lenm))
        for i in range(lenm):
            df2 = pd.DataFrame([pd.Series(np.zeros(lenm))], columns=range(lenm))
            df3 = self.mtx.append(df2,ignore_index=True)
            self.mtx=df3
    def calculate_sp_mtx(self, n=7):

        lenm=len(self.processed_data.index)
        t=0
        for index, row in self.processed_data.iterrows():
            print t
            vec = list()
            row=self.processed_data.iloc[t]
            b = self.find_sims_row(row)
            df2 = pd.DataFrame([b], columns=range(lenm))
            m = nlargest(n, xrange(len(b)), key=b.__getitem__)
            for i in range(n):
                vec.append(self.processed_data.iloc[m[i]][self.type])
            df2[self.stat]=str(collections.Counter(vec))
            df2[self.klasa_obiektu] = row[self.type]
            df3= self.mtx.append(df2,ignore_index=True)
            self.mtx=df3
            t+=1

class OsfClassify():
    def create_mtx(self):
        def get_book_df(tup):
            file_path = "opisProjektuEfektExportClean%s.csv" % tup
            clean_data = pd.read_csv(file_path, encoding='utf-8', header=0)
            a = pd.DataFrame({
                'type': pd.Series(len(clean_data) * [tup]),
                'txt': pd.Series(clean_data["LEMMATIZED"]),
            })
            return a
        self.processed_data=pd.concat([get_book_df(x) for x in range(1,4)],ignore_index=True)
    def statistics(self):
        print self.processed_data.groupby('type').count()
        self.processed_data['words'] = self.processed_data['txt'].apply(
            lambda x: len(x.split())
        )
        print self.processed_data.groupby('type')['words'].describe()
        print self.processed_data.groupby('type')['words'].quantile(0.98)
    def vektorize(self):
        train_df, test_df = train_test_split(
            self.processed_data,
            test_size=0.1,
            stratify=self.processed_data['type'],
        )
        print test_df
        vectorizer = CountVectorizer()
        print vectorizer.fit(train_df['txt'])
        import sys
        reload(sys)
        sys.setdefaultencoding('utf8')
        X_train = vectorizer.transform(train_df['txt'])
        print X_train

        X_test = vectorizer.transform(test_df['txt'])
        model = LogisticRegression(class_weight='balanced', dual=True)
        model.fit(X_train, train_df['type'])
        print
        print model.score(X_test, test_df['type'])
        print X_test
        target = test_df['type']
        predicted = model.predict(X_test)
        print (metrics.classification_report(target, predicted, digits=4))
# for x in range(1,4):
#     data = Osf_Initialize()
#     file_path="exportTyp%sS.csv" % x
#     data.initialize(file_path)
#     data.initialize_sekcja(file_path,str(x),create_structure=True)
#     print "done%s" % x
# data = Osf_Initialize()
# x=2
# file_path="opisProjektuEfekt%s.csv" % x
# data.initialize(file_path)
# data = Cleansing(data)
# data.clean()
# data.clean_not_printable()
# data = OsfData_Lemmatization(data)
# data.lemmatize(str(x))
# print "typ"+str(x)
# print "done%s" % x

# data = Osf_Initialize()
# x=1
# file_path="opisProjektuEfekt%s.csv" % x
# data.initialize(file_path)
# data = Cleansing(data)
# data.clean()
# data.clean_not_printable()
# data = OsfData_Lemmatization(data)
# data.lemmatize(str(x))
# print "typ"+str(x)
# print "done%s" % x

data = OsfData_Wordlist()
data.create_mtx()
print "ok"
data.build_wordlist()
print "ok2"
data.create_dictionary()
print "ok3"
data.calculate_sp_mtx()
print "ok4"
print data.mtx

# data = OsfClassify()
# data.create_mtx()
# data.statistics()
# data.vektorize()


