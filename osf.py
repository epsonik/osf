# -*- coding: UTF-8 -*-
#!/usr/bin/env python
import pandas as pd
import string
import xml
import json
from xml.dom import minidom
from urllib2 import HTTPError
import requests
import gensim
from nltk.tokenize import word_tokenize
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
    def initialize_sekcja(self,sekcja_link,nr_rows=120,create_structure=False):
        if create_structure:
            self.create_structure()
        if sekcja_link:
            self.data_sekcja = pd.read_csv(sekcja_link, encoding='utf-8', header=0)
            self.processed_data=self.data_sekcja.head(nr_rows)
    def create_structure(self):

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
            data.sekcja_opis_projektu_znaczenie_projektu.to_csv('opisProjektuZnaczenieProjektu' + ".csv",
                                                                encoding='utf8')
            data.sekcja_opis_projektu_stan_wiedzy.to_csv('opisProjektuStanWiedzy' + ".csv", encoding='utf8')
            data.sekcja_opis_projektu_metodyka_badan.to_csv('opisProjektuMetodykaBadan' + ".csv", encoding='utf8')
            data.sekcja_opis_projektu_literatura.to_csv('opisProjektuLiteratura' + ".csv", encoding='utf8')
            data.sekcja_opis_projektu_efekt.to_csv('opisProjektuEfekt' + ".csv", encoding='utf8')
            data.sekcja_opis_projektu_cel_naukowy.to_csv('opisProjektuCelNaukowy' + ".csv", encoding='utf8')
class Cleansing(Osf_Initialize):

    def __init__(self, previous):
        self.processed_data = previous.processed_data

    def clean(self):
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'<[^>]+>', '')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&nbsp;', '')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&oacute;', u'ó')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&ndash;', u'-')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&ldquo;', u'"')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&bdquo;', u'"')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&quot;', u'"')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&rdquo;', u'"')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&amp;', u'&')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&ouml;', u'o')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&uuml;', u'u')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&auml;', u'a')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&bull;', u'•')
        self.processed_data.WARTOSC = self.processed_data.WARTOSC.str.replace(r'&middot;', u'•')
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
        polish = u"ąęłóźżćń"
        def clean_not_printable_row(row):
            if type(row.WARTOSC) != float and row.WARTOSC:
                string.printable = string.printable + polish
                row.WARTOSC = filter(lambda x: x in string.printable, row.WARTOSC)
                return row
        self.processed_data = self.processed_data.apply(clean_not_printable_row, axis=1)

class OsfData_Lemmatization(Cleansing):

    def __init__(self, previous):
        self.processed_data = previous.processed_data
    def lemmatize(self, ):
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
                        row["LEMMATIZED"] = parseXML(r.text,row)
                        return row
                except HTTPError:
                    print 'Failed to open %d -%s' % (row.ROWNUM,row)
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
                return description
            except xml.parsers.expat.ExpatError, e:
                print "ERROR "
                print e
                print row.ROWNUM
                print resp

        self.processed_data = self.processed_data.apply(lemmatize_row, axis=1)
        data.processed_data.to_csv('exportClean' + ".csv", encoding='utf8')
class OsfData_Wordlist():
    dictionary=[]
    corpus = []
    tf_idf = []
    sims = []
    def __init__(self):
        csv_link = "exportClean.csv"
        self.clean_data= pd.read_csv(csv_link, encoding='utf-8', header=0)
        self.processed_data = self.clean_data
    def build_wordlist(self):
        def sortFreqDict(freqdict):
            aux = [(freqdict[key], key) for key in freqdict]
            aux.sort()
            aux.reverse()
            return aux

        def wordListToFreqDict(wordlist):
            wordfreq = [wordlist.count(p) for p in wordlist]
            return dict(zip(wordlist, wordfreq))
        def build_wordlist_row(row):
            row["DICT"]=sortFreqDict(wordListToFreqDict(row["LEMMATIZED"]))
            return row
        self.processed_data = self.processed_data.apply(build_wordlist_row, axis=1)

    def create_dictionary(self):
        self.dictionary = gensim.corpora.Dictionary(self.processed_data["LEMMATIZED"])
        self.corpus = [self.dictionary.doc2bow(gen_doc) for gen_doc in self.processed_data["LEMMATIZED"]]
        self.tf_idf = gensim.models.TfidfModel(self.corpus)
        self.sims = gensim.similarities.Similarity('/home/mbarto/PycharmProjects/osf',
                                              self.tf_idf[self.corpus],
                                                   num_features=len(self.dictionary))
    def find_sims(self):
        def find_sims_row(row):
            query_doc_bow = self.dictionary.doc2bow(row["LEMMATIZED"])
            query_doc_tf_idf = self.tf_idf[query_doc_bow]
            print row["LEMMATIZED"]
            print self.sims[query_doc_tf_idf]
            return row
        self.processed_data = self.processed_data.apply(find_sims_row, axis=1)


# data = Osf_Initialize()
# file_path="export195.csv"
# data.initialize_sekcja("opisProjektuStanWiedzy.csv",60)
# data = Cleansing(data)
# data.clean()
# data.clean_not_printable()
# data = OsfData_Lemmatization(data)
# data.lemmatize()
data = OsfData_Wordlist()
print data.processed_data["LEMMATIZED"]
data.create_dictionary()
data.find_sims()

