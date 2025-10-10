import re
import string
import unicodedata

import inflect

from languages import Languages

SUPPORTED_PUNCTUATION_SET = ",.?"


class Normalizer(object):
    def __init__(self, keep_punctuation: bool, punctuation_set: str = SUPPORTED_PUNCTUATION_SET) -> None:
        self._keep_punctuation = keep_punctuation
        self._punctuation_set = punctuation_set

    def normalize(self, sentence: str, raise_error_on_invalid_sentence: bool) -> str:
        raise NotImplementedError()

    @classmethod
    def create(
        cls,
        language: Languages,
        keep_punctuation: bool,
        punctuation_set: str = SUPPORTED_PUNCTUATION_SET,
    ):
        if language == Languages.EN:
            return EnglishNormalizer(keep_punctuation, punctuation_set)
        elif language == Languages.AR:
            return ArabicNormalizer(keep_punctuation, punctuation_set)
        elif language in [
            Languages.DE,
            Languages.ES,
            Languages.FR,
            Languages.IT,
            Languages.PT_PT,
            Languages.PT_BR,
        ]:
            return DefaultNormalizer(keep_punctuation, punctuation_set)
        else:
            raise ValueError(f"Cannot create {cls.__name__} of type `{language}`")


class DefaultNormalizer(Normalizer):
    """
    Adapted from: https://github.com/openai/whisper/blob/main/whisper/normalizers/basic.py
    """

    ADDITIONAL_DIACRITICS = {
        "œ": "oe",
        "Œ": "OE",
        "ø": "o",
        "Ø": "O",
        "æ": "ae",
        "Æ": "AE",
        "ß": "ss",
        "ẞ": "SS",
        "đ": "d",
        "Đ": "D",
        "ð": "d",
        "Ð": "D",
        "þ": "th",
        "Þ": "th",
        "ł": "l",
        "Ł": "L",
    }

    def _remove_symbols_and_diacritics(self, s: str) -> str:
        return "".join(
            (
                DefaultNormalizer.ADDITIONAL_DIACRITICS[c]
                if c in DefaultNormalizer.ADDITIONAL_DIACRITICS
                else (
                    ""
                    if unicodedata.category(c) == "Mn"
                    else (
                        " "
                        if unicodedata.category(c)[0] in "MS"
                        or (unicodedata.category(c)[0] == "P" and c not in SUPPORTED_PUNCTUATION_SET)
                        else c
                    )
                )
            )
            for c in unicodedata.normalize("NFKD", s)
        )

    def normalize(self, sentence: str, raise_error_on_invalid_sentence: bool = False) -> str:
        sentence = sentence.lower()
        sentence = re.sub(r"[<\[][^>\]]*[>\]]", "", sentence)
        sentence = re.sub(r"\(([^)]+?)\)", "", sentence)
        sentence = sentence.replace("!", ".")
        sentence = sentence.replace("...", "")
        sentence = self._remove_symbols_and_diacritics(sentence).lower()

        if self._keep_punctuation:
            removable_punctuation = "".join(set(SUPPORTED_PUNCTUATION_SET) - set(self._punctuation_set))
        else:
            removable_punctuation = SUPPORTED_PUNCTUATION_SET

        for c in removable_punctuation:
            sentence = sentence.replace(c, "")

        sentence = re.sub(r"\s+", " ", sentence)

        return sentence


class EnglishNormalizer(Normalizer):
    AMERICAN_SPELLINGS = {
        "acknowledgement": "acknowledgment",
        "analogue": "analog",
        "armour": "armor",
        "ascendency": "ascendancy",
        "behaviour": "behavior",
        "behaviourist": "behaviorist",
        "cancelled": "canceled",
        "catalogue": "catalog",
        "centre": "center",
        "centres": "centers",
        "colour": "color",
        "coloured": "colored",
        "colourist": "colorist",
        "colourists": "colorists",
        "colours": "colors",
        "cosier": "cozier",
        "counselled": "counseled",
        "criticised": "criticized",
        "crystallise": "crystallize",
        "defence": "defense",
        "discoloured": "discolored",
        "dishonour": "dishonor",
        "dishonoured": "dishonored",
        "encyclopaedia": "Encyclopedia",
        "endeavour": "endeavor",
        "endeavouring": "endeavoring",
        "favour": "favor",
        "favourite": "favorite",
        "favours": "favors",
        "fibre": "fiber",
        "flamingoes": "flamingos",
        "fulfill": "fulfil",
        "grey": "gray",
        "harmonised": "harmonized",
        "honour": "honor",
        "honourable": "honorable",
        "honourably": "honorably",
        "honoured": "honored",
        "honours": "honors",
        "humour": "humor",
        "islamised": "islamized",
        "labour": "labor",
        "labourers": "laborers",
        "levelling": "leveling",
        "luis": "lewis",
        "lustre": "luster",
        "manoeuvring": "maneuvering",
        "marshall": "marshal",
        "marvellous": "marvelous",
        "merchandising": "merchandizing",
        "milicent": "millicent",
        "moustache": "mustache",
        "moustaches": "mustaches",
        "neighbour": "neighbor",
        "neighbourhood": "neighborhood",
        "neighbouring": "neighboring",
        "neighbours": "neighbors",
        "omelette": "omelet",
        "organisation": "organization",
        "organiser": "organizer",
        "practise": "practice",
        "pretence": "pretense",
        "programme": "program",
        "realise": "realize",
        "realised": "realized",
        "recognised": "recognized",
        "shrivelled": "shriveled",
        "signalling": "signaling",
        "skilfully": "skillfully",
        "smouldering": "smoldering",
        "specialised": "specialized",
        "sterilise": "sterilize",
        "sylvia": "silvia",
        "theatre": "theater",
        "theatres": "theaters",
        "travelled": "traveled",
        "travellers": "travelers",
        "travelling": "traveling",
        "vapours": "vapors",
        "wilful": "willful",
    }

    ABBREVIATIONS = {
        "junior": "jr",
        "senior": "sr",
        "okay": "ok",
        "doctor": "dr",
        "mister": "mr",
        "missus": "mrs",
        "saint": "st",
    }

    APOSTROPHE_REGEX = r"(?<!\w)\'|\'(?!\w)"  # Apostrophes that are not part of a contraction

    @staticmethod
    def to_american(sentence: str) -> str:
        return " ".join(
            [
                (EnglishNormalizer.AMERICAN_SPELLINGS[x] if x in EnglishNormalizer.AMERICAN_SPELLINGS else x)
                for x in sentence.split()
            ]
        )

    @staticmethod
    def normalize_abbreviations(sentence: str) -> str:
        return " ".join(
            [
                (EnglishNormalizer.ABBREVIATIONS[x] if x in EnglishNormalizer.ABBREVIATIONS else x)
                for x in sentence.split()
            ]
        )

    def normalize(self, sentence: str, raise_error_on_invalid_sentence: bool = False) -> str:
        p = inflect.engine()

        sentence = sentence.lower()

        for c in "-/–—":
            sentence = sentence.replace(c, " ")

        for c in '‘":;“”`()[]':
            sentence = sentence.replace(c, "")

        sentence = sentence.replace("!", ".")
        sentence = sentence.replace("...", "")

        if self._keep_punctuation:
            removable_punctuation = "".join(set(SUPPORTED_PUNCTUATION_SET) - set(self._punctuation_set))
        else:
            removable_punctuation = SUPPORTED_PUNCTUATION_SET

        for c in removable_punctuation:
            sentence = sentence.replace(c, "")

        sentence = sentence.replace("’", "'").replace("&", "and")

        sentence = re.sub(self.APOSTROPHE_REGEX, "", sentence)

        def num2txt(y):
            if any(x.isdigit() for x in y):
                ends_with_period = y[-1] == '.' and self._keep_punctuation
                if ends_with_period:
                    y = y[:-1]
                y = p.number_to_words(y).replace("-", " ").replace(",", "")
                if ends_with_period:
                    y += '.'
            return y

        sentence = " ".join(num2txt(x) for x in sentence.split())

        if raise_error_on_invalid_sentence:
            valid_characters = " '" + self._punctuation_set if self._keep_punctuation else " '"
            if not all(c in valid_characters + string.ascii_lowercase for c in sentence):
                raise RuntimeError()
            if any(x.startswith("'") for x in sentence.split()):
                raise RuntimeError()

        return sentence


class ArabicNormalizer(Normalizer):
    """
    Normalizer for Arabic text, handling common variations and diacritics
    """

    # Arabic diacritics (tashkeel) to remove
    ARABIC_DIACRITICS = [
        '\u064B',  # Tanween Fath
        '\u064C',  # Tanween Damm
        '\u064D',  # Tanween Kasr
        '\u064E',  # Fatha
        '\u064F',  # Damma
        '\u0650',  # Kasra
        '\u0651',  # Shadda
        '\u0652',  # Sukun
        '\u0653',  # Maddah
        '\u0654',  # Hamza above
        '\u0655',  # Hamza below
        '\u0656',  # Subscript alef
        '\u0670',  # Dagger alef
    ]

    # Normalize various forms of alef
    ALEF_NORMALIZATION = {
        '\u0622': '\u0627',  # Alef with madda -> Alef
        '\u0623': '\u0627',  # Alef with hamza above -> Alef
        '\u0625': '\u0627',  # Alef with hamza below -> Alef
        '\u0671': '\u0627',  # Alef wasla -> Alef
    }

    # Normalize teh marbuta
    TEH_MARBUTA_NORMALIZATION = {
        '\u0629': '\u0647',  # Teh marbuta -> Heh
    }

    def _remove_diacritics(self, text: str) -> str:
        """Remove Arabic diacritics from text"""
        for diacritic in self.ARABIC_DIACRITICS:
            text = text.replace(diacritic, '')
        return text

    def _normalize_alef(self, text: str) -> str:
        """Normalize different forms of Alef"""
        for original, normalized in self.ALEF_NORMALIZATION.items():
            text = text.replace(original, normalized)
        return text

    def _normalize_teh_marbuta(self, text: str) -> str:
        """Normalize Teh Marbuta to Heh"""
        for original, normalized in self.TEH_MARBUTA_NORMALIZATION.items():
            text = text.replace(original, normalized)
        return text

    def normalize(self, sentence: str, raise_error_on_invalid_sentence: bool = False) -> str:
        # Remove Latin script in brackets/parentheses
        sentence = re.sub(r'[<\[][^>\]]*[>\]]', '', sentence)
        sentence = re.sub(r'\([^)]+?\)', '', sentence)

        # Remove diacritics
        sentence = self._remove_diacritics(sentence)

        # Normalize alef variations
        sentence = self._normalize_alef(sentence)

        # Normalize teh marbuta
        sentence = self._normalize_teh_marbuta(sentence)

        # Remove tatweel (elongation character)
        sentence = sentence.replace('\u0640', '')

        # Handle punctuation
        if self._keep_punctuation:
            removable_punctuation = "".join(set(SUPPORTED_PUNCTUATION_SET) - set(self._punctuation_set))
        else:
            removable_punctuation = SUPPORTED_PUNCTUATION_SET

        for c in removable_punctuation:
            sentence = sentence.replace(c, '')

        # Remove English letters and special characters, keep only Arabic and punctuation
        # Arabic Unicode range: \u0600-\u06FF
        allowed_chars = r'[\u0600-\u06FF\s' + re.escape(self._punctuation_set if self._keep_punctuation else '') + r']'
        sentence = ''.join(c if re.match(allowed_chars, c) else ' ' for c in sentence)

        # Normalize whitespace
        sentence = re.sub(r'\s+', ' ', sentence)
        sentence = sentence.strip()

        return sentence


__all__ = ["Normalizer"]
