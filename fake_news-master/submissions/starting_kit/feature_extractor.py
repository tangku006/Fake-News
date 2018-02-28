import pandas as pd
import scipy
import string
from scipy.sparse import hstack
from sklearn import preprocessing
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import ast
import numpy as np
 
def document_preprocessor(doc):
    # TODO: is there a way to avoid these encode/decode calls?
    try:
        doc = unicode(doc, 'utf-8')
    except NameError:  # unicode is a default on python 3
        pass
    doc = unicodedata.normalize('NFD', doc)
    doc = doc.encode('ascii', 'ignore')
    doc = doc.decode("utf-8")
    return str(doc)
 
def token_processor(tokens):
    """ A custom token processor
    
    This function can be edited to add some additional
    transformation on the extracted tokens (e.g. stemming)
    """
 
    stemmer = SnowballStemmer('english')
    for token in tokens:
        yield stemmer.stem(token)
 
class FeatureExtractor(TfidfVectorizer):
    """Convert a collection of raw docs to a matrix of TF-IDF features. """
    def __init__(self):
        nltk_stop_words = set(stopwords.words('english'))
        sklearn_stop_words = set(stop_words.ENGLISH_STOP_WORDS)
        another_stop_words = set(['a', 'able', 'about', 'above', 'abroad', 'according', 'accordingly', 'across', 'actually', 'adj', 'after', 'afterwards', 'again', 'against', 'ago', 'ahead', 'ain\'t', 'all', 'allow', 'allows', 'almost', 'alone', 'along', 'alongside', 'already', 'also', 'although', 'always', 'am', 'amid', 'amidst', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'apart', 'appear', 'appreciate', 'appropriate', 'are', 'aren\'t', 'around', 'as', 'a\'s', 'aside', 'ask', 'asking', 'associated', 'at', 'available', 'away', 'awfully', 'b', 'back', 'backward', 'backwards', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'begin', 'behind', 'being', 'believe', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'both', 'brief', 'but', 'by', 'c', 'came', 'can', 'cannot', 'cant', 'can\'t', 'caption', 'cause', 'causes', 'certain', 'certainly', 'changes', 'clearly', 'c\'mon', 'co', 'co.', 'com', 'come', 'comes', 'concerning', 'consequently', 'consider', 'considering', 'contain', 'containing', 'contains', 'corresponding', 'could', 'couldn\'t', 'course', 'c\'s', 'currently', 'd', 'dare', 'daren\'t', 'definitely', 'described', 'despite', 'did', 'didn\'t', 'different', 'directly', 'do', 'does', 'doesn\'t', 'doing', 'done', 'don\'t', 'down', 'downwards', 'during', 'e', 'each', 'edu', 'eg', 'eight', 'eighty', 'either', 'else', 'elsewhere', 'end', 'ending', 'enough', 'entirely', 'especially', 'et', 'etc', 'even', 'ever', 'evermore', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'f', 'fairly', 'far', 'farther', 'few', 'fewer', 'fifth', 'first', 'five', 'followed', 'following', 'follows', 'for', 'forever', 'former', 'formerly', 'forth', 'forward', 'found', 'four', 'from', 'further', 'furthermore', 'g', 'get', 'gets', 'getting', 'given', 'gives', 'go', 'goes', 'going', 'gone', 'got', 'gotten', 'greetings', 'h', 'had', 'hadn\'t', 'half', 'happens', 'hardly', 'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'hello', 'help', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'here\'s', 'hereupon', 'hers', 'herself', 'he\'s', 'hi', 'him', 'himself', 'his', 'hither', 'hopefully', 'how', 'howbeit', 'however', 'hundred', 'i', 'i\'d', 'ie', 'if', 'ignored', 'i\'ll', 'i\'m', 'immediate', 'in', 'inasmuch', 'inc', 'inc.', 'indeed', 'indicate', 'indicated', 'indicates', 'inner', 'inside', 'insofar', 'instead', 'into', 'inward', 'is', 'isn\'t', 'it', 'it\'d', 'it\'ll', 'its', 'it\'s', 'itself', 'i\'ve', 'j', 'just', 'k', 'keep', 'keeps', 'kept', 'know', 'known', 'knows', 'l', 'last', 'lately', 'later', 'latter', 'latterly', 'least', 'less', 'lest', 'let', 'let\'s', 'like', 'liked', 'likely', 'likewise', 'little', 'll', 'look', 'looking', 'looks', 'low', 'lower', 'ltd', 'm', 'made', 'mainly', 'make', 'makes', 'many', 'may', 'maybe', 'mayn\'t', 'me', 'mean', 'meantime', 'meanwhile', 'merely', 'might', 'mightn\'t', 'mine', 'minus', 'miss', 'more', 'moreover', 'most', 'mostly', 'mr', 'mrs', 'much', 'must', 'mustn\'t', 'my', 'myself', 'n', 'name', 'namely', 'nd', 'near', 'nearly', 'necessary', 'need', 'needn\'t', 'needs', 'neither', 'never', 'neverf', 'neverless', 'nevertheless', 'new', 'next', 'nine', 'ninety', 'no', 'nobody', 'non', 'none', 'nonetheless', 'noone', 'no-one', 'nor', 'normally', 'not', 'nothing', 'notwithstanding', 'novel', 'now', 'nowhere', 'o', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'on', 'once', 'one', 'ones', 'one\'s', 'only', 'onto', 'opposite', 'or', 'other', 'others', 'otherwise', 'ought', 'oughtn\'t', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'own', 'p', 'particular', 'particularly', 'past', 'per', 'perhaps', 'placed', 'please', 'plus', 'possible', 'presumably', 'probably', 'provided', 'provides', 'q', 'que', 'quite', 'qv', 'r', 'rather', 'rd', 're', 'really', 'reasonably', 'recent', 'recently', 'regarding', 'regardless', 'regards', 'relatively', 'respectively', 'right', 'round', 's', 'said', 'same', 'saw', 'say', 'saying', 'says', 'second', 'secondly', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'several', 'shall', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 'shouldn\'t', 'since', 'six', 'so', 'some', 'somebody', 'someday', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specified', 'specify', 'specifying', 'still', 'sub', 'such', 'sup', 'sure', 't', 'take', 'taken', 'taking', 'tell', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', 'that\'ll', 'thats', 'that\'s', 'that\'ve', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'there\'d', 'therefore', 'therein', 'there\'ll', 'there\'re', 'theres', 'there\'s', 'thereupon', 'there\'ve', 'these', 'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'thing', 'things', 'think', 'third', 'thirty', 'this', 'thorough', 'thoroughly', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'till', 'to', 'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 't\'s', 'twice', 'two', 'u', 'un', 'under', 'underneath', 'undoing', 'unfortunately', 'unless', 'unlike', 'unlikely', 'until', 'unto', 'up', 'upon', 'upwards', 'us', 'use', 'used', 'useful', 'uses', 'using', 'usually', 'uucp', 'v', 'value', 'various', 've', 'versus', 'very', 'via', 'viz', 'vs', 'w', 'want', 'wants', 'was', 'wasn\'t', 'way', 'we', 'we\'d', 'welcome', 'well', 'we\'ll', 'went', 'were', 'we\'re', 'weren\'t', 'we\'ve', 'what', 'whatever', 'what\'ll', 'what\'s', 'what\'ve', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'where\'s', 'whereupon', 'wherever', 'whether', 'which', 'whichever', 'while', 'whilst', 'whither', 'who', 'who\'d', 'whoever', 'whole', 'who\'ll', 'whom', 'whomever', 'who\'s', 'whose', 'why', 'will', 'willing', 'wish', 'with', 'within', 'without', 'wonder', 'won\'t', 'would', 'wouldn\'t', 'x', 'y', 'yes', 'yet', 'you', 'you\'d', 'you\'ll', 'your', 'you\'re', 'yours', 'yourself', 'yourselves', 'you\'ve', 'z', 'zero'])
        self.all_stop_words = list(nltk_stop_words | sklearn_stop_words | another_stop_words)
        usefull_stopwords = ['very', 'too', 'such', 'only', 'against', 'not', 'nor', 'no', 'most', 'more' ]
        self.all_stop_words = set([w for w in self.all_stop_words if w not in usefull_stopwords])
        
        super(FeatureExtractor, self).__init__(input='content', encoding='utf-8',
            decode_error='strict', strip_accents=None, lowercase=True,
            preprocessor=None, tokenizer=None, analyzer='word',
            stop_words=self.all_stop_words,
            ngram_range=(1, 1), max_df=0.7, min_df=0.01,
            max_features=None, vocabulary=None, binary=False,
            dtype=np.int64, norm='l2', use_idf=True, smooth_idf=False,
            sublinear_tf=False)        
 
    def fit(self, X_df, y=None):
        """Learn a vocabulary dictionary of all tokens in the raw documents.
 
        Parameters
        ----------
        X_df : pandas.DataFrame
            a DataFrame, where the text data is stored in the ``statement``
            column.
        """
        super(FeatureExtractor, self).fit(X_df.statement, y)
        return self
 
    def build_tokenizer(self):
        """
        Internal function, needed to plug-in the token processor, cf.
        http://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes
        """
 
        tokenize = super(FeatureExtractor, self).build_tokenizer()
        return lambda doc: list(token_processor(tokenize(doc)))
 
    def transform(self, X_df):
        self.df = X_df[['job', 'source', 'state', 'subjects']]
        self.df.reset_index(drop=True, inplace=True)        
        subjects = ['Candidate Biography', 'Crime', 'Economy', 'Education', 'Elections', 'Energy', 'Environment', 'Federal Budget', 'Health Care', 'Immigration', 'Jobs', 'Message Machine 2010', 'Message Machine 2012', 'Other subject', 'State Budget', 'Taxes'] 
        job = ['Democrat', 'Other job', 'Republican']
        source = ['Barack Obama', 'Chain email', 'Chris Christie', 'Hillary Clinton', 'Joe Biden', 'John Boehner', 'John McCain', 'Marco Rubio', 'Michele Bachmann', 'Mitt Romney', 'Newt Gingrich', 'Other source', 'Rick Perry', 'Rick Scott', 'Sarah Palin', 'Scott Walker'] 
        state = ['Arizona', 'Florida', 'Georgia', 'Illinois', 'Massachusetts', 'New Jersey', 'New York', 'Ohio', 'Oregon', 'Other state', 'Rhode Island', 'Texas', 'Virginia', 'Wisconsin']
        self.df = self.df.join(pd.DataFrame(columns=subjects)).join(pd.DataFrame(columns=job)).join(pd.DataFrame(columns=source)).join(pd.DataFrame(columns=state))
        self.df.fillna(0, inplace=True)
        for i in range(self.df.shape[0]):
            subjects_row = ast.literal_eval(self.df['subjects'][i])
            for j in subjects_row:
                if j in subjects:
                    self.df.at[i, j] = 1
                else:
                    self.df.at[1, 'Other subject'] = 1
            job_ = self.df.at[i, 'job']
            if str(job_) in job:
                self.df.at[i, job_] = 1
            else:
                self.df.at[i, 'Other job'] = 1
            source_ = self.df.at[i, 'source']
            if source_ in source:
                self.df.at[i, source_] = 1
            else:
                self.df.at[i, 'Other source'] = 1
            state_ = self.df.at[i, 'state']
            if state_ in state:
                self.df.at[i, state_] = 1
            else:
                self.df.at[i, 'Other state'] = 1
        self.df.drop('subjects', axis=1, inplace=True)
        self.df.drop('job', axis=1, inplace=True)
        self.df.drop('source', axis=1, inplace=True)
        self.df.drop('state', axis=1, inplace=True)
        eng_stopwords = set(self.all_stop_words)
        # Number of words in the statement
        X_df["num_words"] = X_df["statement"].apply(lambda x: len(x.split()))
        ## Number of unique words in the statement
        X_df["num_unique_words"] = X_df["statement"].apply(lambda x: len(set(x.split())))
        ## Number of characters in the statement
        X_df["num_chars"] = X_df["statement"].apply(lambda x: len(x))
        ## Number of stopwords in the statement
        X_df["num_stopwords"] = X_df["statement"].apply(lambda x: len([w for w in x.lower().split() if w in eng_stopwords]))
        ## Number of punctuations in the statement
        punctuation = set(string.punctuation)
        punctuation.update(["``", "`", "..."])
        X_df["num_punctuations"] =X_df['statement'].apply(lambda x: len([c for c in x if c in punctuation]) )
        ## Number of title case words in the statement
        X_df["num_words_upper"] = X_df["statement"].apply(lambda x: len([w for w in x.split() if w.isupper()]))
        ## Number of title case words in the statement
        X_df["num_words_title"] = X_df["statement"].apply(lambda x: len([w for w in x.split() if w.istitle()]))
        # Average length of the words in the statement
        X_df["mean_word_len"] = X_df["statement"].apply(lambda x: np.mean([len(w) for w in x.split()])) 
        X_df_others = X_df.drop(['statement','date','edited_by','job','researched_by','source','state','subjects'],axis=1)
        from sklearn import preprocessing
        X = hstack([scipy.sparse.csr_matrix(preprocessing.scale(X_df_others.values)), super(FeatureExtractor, self).transform(X_df.statement), scipy.sparse.csr_matrix(self.df.values)]).toarray()
        return X
    def fit_transform(self, X_df, y=None):
        return self.fit(X_df, y).transform(X_df)