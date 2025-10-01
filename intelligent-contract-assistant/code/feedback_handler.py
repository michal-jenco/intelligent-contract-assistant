# ============================================================
#  Project:     Intelligent Contract Assistant - Sample task for Blocshop/Erste
#  Author:      Michal Jenčo
#  Created:     2025
#
#  Copyright (c) 2025 Michal Jenčo


import hashlib
import csv
import os
import json
from datetime import datetime


_FEEDBACK_FILE = "../feedback_store/feedback_log.csv"
_CORRECTIONS_FILE = "../feedback_store/corrections.json"


class FeedbackHandler:
    """
    Handles saving and loading feedback for a single input source file
    """

    def __init__(self, filepath):
        self.hash = self._get_file_hash(filepath)

    def log_feedback(self, question, answer, sources, feedback, correction) -> None:
        """Append feedback to CSV."""

        file_exists = os.path.isfile(_FEEDBACK_FILE)

        with open(_FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(["timestamp", "question", "answer", "sources", "feedback", "correction", "file_hash"])

            writer.writerow([
                datetime.now().isoformat(),
                question,
                answer,
                "; ".join(sources),
                feedback,
                correction,
                self.hash,
            ])

    @staticmethod
    def load_corrections():
        if os.path.exists(_CORRECTIONS_FILE):
            with open(_CORRECTIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_correction(self, question, correction) -> None:
        corrections = self.load_corrections()
        key = question.strip().lower()
        corrections[key] = {"correction": correction, "file_hash": self.hash, "updated": datetime.now().isoformat()}

        with open(_CORRECTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(corrections, f, indent=2, ensure_ascii=False)

    def check_corrections(self, question):
        corrections = self.load_corrections()
        key = question.strip().lower()

        return corrections.get(key, None)

    @staticmethod
    def _get_file_hash(filepath) -> str:
        """Return SHA256 hash of a file (used to tie feedback to a specific doc)."""

        h = hashlib.sha256()

        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()
