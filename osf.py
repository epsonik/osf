# -*- coding: UTF-8 -*-
#!/usr/bin/env python
import pandas as pd
import validators
import urllib2
import string
import xml
from xml.dom import minidom
from urllib2 import quote
from urllib2 import HTTPError
from httplib import BadStatusLine
url='http://ws.clarin-pl.eu/nlprest/syn/wcrft2/'
class Osf_Initialize():
    processed_data=pd.DataFrame
    data = pd.DataFrame
    data_model = None
    def initialize(self,csv_link):
        self.data = pd.read_csv(csv_link, encoding='utf-8', header=0)
        self.data_model = pd.read_csv(csv_link, encoding='utf-8', header=0)
        self.processed_data=self.data.ix[1010:]
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
class Lemmatization(Cleansing):

    def __init__(self, previous):
        self.processed_data = previous.processed_data
    def lemmatize(self):
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
        def lemmatize_row(row):
            if type(row.WARTOSC) != float and row.WARTOSC:
                data = quote(row.WARTOSC.encode('utf8'))
                validators.url(url+data)
                if validators.url(url+data) ==True:
                    try:
                        adress=url+data
                        request = urllib2.Request(adress)
                        resp = urllib2.urlopen(request).read()
                        row["LEMMATIZED"] = parseXML(resp, row,adress)
                        return row
                    except HTTPError, e:
                        print 'Failed to open %d -%s' % (row.ROWNUM, adress)
        def parseXML(resp,row,adress):
            try:
                parsed = minidom.parseString(resp)
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
                print adress
                print resp
        self.processed_data = self.processed_data.apply(lemmatize_row, axis=1)
data = Osf_Initialize()
file_path="export195.csv"
data.initialize(file_path)
data = Cleansing(data)
data.clean()
data.clean_not_printable()
data = Lemmatization(data)
data.lemmatize()
print data.processed_data.WARTOSC
print data.processed_data["LEMMATIZED"]