import os

from app.src.domain.link import Link
from app.src.services.db_service import DBService
from app.src.services.logging_service import LoggingService

import spacy

from dotenv import load_dotenv
load_dotenv()
IMIS = os.getenv('IMIS')

class NaturalLanguageProcessingService:
    def __init__(self, db_service: DBService, logging_service: LoggingService):
        self.db_service = db_service
        self.logging_service = logging_service
        self.nlp = spacy.load("en_coreference_web_trf")


    def get_unprocessed_ids(self):
        where = {"nlp": None}
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

    def do_coreference_resolution(self, text):
        doc = self.nlp(text)
        spans = doc.spans
        span_results = ''
        self.logging_service.logger.debug(spans)
        for spangroup in spans.values():
            for span in spangroup:
                self.logging_service.logger.debug(text[span.start_char:span.end_char])
                span_results = span_results + text[span.start_char:span.end_char]
        return span_results
