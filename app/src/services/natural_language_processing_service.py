import math
import os

from app.src.domain.link import Link
from app.src.services.db_service import DBService
from app.src.services.logging_service import LoggingService

import spacy
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
nltk.download('wordnet')

from dotenv import load_dotenv
load_dotenv()
IMIS = os.getenv('IMIS')

class NaturalLanguageProcessingService:
    def __init__(self, db_service: DBService, logging_service: LoggingService):
        self.db_service = db_service
        self.logging_service = logging_service
        self.nlp = spacy.load("en_coreference_web_trf")


    def get_unprocessed_ids(self):
        where = {"relevance_score": None}
        what = {"_id": 1}
        self.db_service.set_collection("search_results")
        unprocessed_ids = self.db_service.select_what_where(what, where)
        return unprocessed_ids

    def get_text(self, search_result_id):
        where = {"_id": search_result_id}
        what = {"text": 1, "_id": 0}
        self.db_service.set_collection("search_results")
        text_cursor = self.db_service.select_what_where(what, where)
        text = text_cursor.next()
        text_cursor.close()
        return text['text']

    def tokenize(self, text: str):
        token_text = word_tokenize(text)
        return token_text

    def pos_tagging(self, token_text):
        pos_tag_text = nltk.pos_tag(token_text)
        return pos_tag_text

    def remove_stopwords_en(self,text: str):
        stop_words_en = set(stopwords.words('english')) #majority of publication will be in english
        punctuations = "?:!.,;<>/\+-"
        # turn the string into a list of words based on separators (blank, comma, etc.)
        word_tokens = word_tokenize(text.lower())
        # create a list of all words that are neither stopwords nor punctuations
        result = [x for x in word_tokens if x not in stop_words_en and x not in punctuations]

        # create a new string of all remaining words
        seperator = ' '
        return seperator.join(result)

    def lemmatizing_en(self, text: str):
        word_tokens = word_tokenize(text.lower())
        seperator = ' '
        result = [lemmatizer.lemmatize(x) for x in word_tokens]
        return seperator.join(result)

    def coreference_resolution(self, text):
        doc = self.nlp(text)
        spans = doc.spans
        span_array = []
        self.logging_service.logger.debug(spans)
        for spangroup in spans.values():
            span_tuple = []
            for span in spangroup:
                self.logging_service.logger.debug(text[span.start_char:span.end_char])
                span_tuple.append(text[span.start_char:span.end_char])
            span_array.append(span_tuple)
        return span_array

    def pronoun_to_noun_mapping(self, text, coreference_array):
        for coreference_tuple in coreference_array:
            text = text.replace(coreference_tuple[1], coreference_tuple[0])
        return text

    def number_of_words(self, text: str):
        return len(text)

    def number_of_sentences(self, text: str):
        return len(sent_tokenize(text))

    def topic_count(self, topic: str, text: str):
        return text.count(topic)

    def number_of_nouns(self, text: str):
        lines = 'lines is some string of words'
        # function to test if something is a noun
        is_noun = lambda pos: pos[:2] == 'NN'
        # do the nlp stuff
        tokenized = word_tokenize(text)
        nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
        return len(nouns)

    def relevance_score(self, number_of_sentences, topic_count, number_of_nouns):
        topic_noun_ratio = topic_count / number_of_nouns
        topic_sentence_ratio = topic_count / number_of_sentences
        score = math.log10(topic_count * topic_noun_ratio * topic_sentence_ratio * pow(10, 5))
        return score

