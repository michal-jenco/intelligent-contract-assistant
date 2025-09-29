# ============================================================
#  Project:     Intelligent Contract Assistant - Sample task for Blocshop/Erste
#  Author:      Michal Jenčo
#  Created:     2025
#
#  Copyright (c) 2025 Michal Jenčo


import spacy


class NamedEntityRecognizer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    # TODO: figure out how to type the Sequence of Spans that this returns
    def get_entities(self, text: str):
        doc = self.nlp(text)

        return doc.ents
