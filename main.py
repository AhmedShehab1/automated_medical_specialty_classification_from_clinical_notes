import re

import scispacy  # noqa: F401
import spacy
from confection import ConfigValidationError
from spacy.language import Language
from spacy.tokens import Doc, Span


class ClinicalTextPipeline:
    """Base pipeline for lightweight text cleaning and de-identification."""

    DEFAULT_MODEL_NAME = "en_core_sci_sm"

    _DATE_PATTERNS = (
        re.compile(r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:\d{2}|\d{4})\b"),
        re.compile(r"\b\d{4}-(?:0?[1-9]|1[0-2])-(?:0?[1-9]|[12]\d|3[01])\b"),
    )
    _PHONE_PATTERN = re.compile(
        r"(?<!\w)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)"
    )
    _SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    _MODEL_COMPATIBILITY_OVERRIDES = {
        "components.tok2vec.model.embed.include_static_vectors": False,
        "components.ner.model.tok2vec.embed.include_static_vectors": False,
    }

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        """Initialize the pipeline and load the biomedical spaCy model."""
        self.model_name = model_name
        try:
            self.nlp: Language = spacy.load(model_name)
        except ConfigValidationError:
            self.nlp = spacy.load(model_name, config=self._MODEL_COMPATIBILITY_OVERRIDES)
        except OSError as exc:
            raise RuntimeError(
                "Failed to load model '"
                f"{model_name}'. Install with: "
                "uv pip install --python .venv/bin/python --no-deps "
                "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"
            ) from exc
        self.is_ready = True

    def de_identify(self, text: str) -> str:
        """Replace common PII patterns with standard placeholders."""
        masked_text = text

        for date_pattern in self._DATE_PATTERNS:
            masked_text = date_pattern.sub("[DATE]", masked_text)

        masked_text = self._PHONE_PATTERN.sub("[PHONE]", masked_text)
        masked_text = self._SSN_PATTERN.sub("[SSN]", masked_text)
        return masked_text

    def clean_text(self, text: str) -> str:
        """Normalize whitespace and remove line breaks."""
        cleaned_text = re.sub(r"[\r\n\t]+", " ", text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        return cleaned_text.strip()

    def _merge_adjacent_entities(self, doc: Doc) -> list[Span]:
        """Join touching entities so multi-token medical concepts stay intact."""
        merged_entities: list[Span] = []

        for entity in doc.ents:
            if not merged_entities:
                merged_entities.append(entity)
                continue

            previous = merged_entities[-1]
            between_tokens = doc[previous.end : entity.start]
            should_merge = all(
                token.is_space
                or token.is_punct
                or token.text.lower() in {"of", "for", "to", "in"}
                for token in between_tokens
            )

            if should_merge:
                merged_entities[-1] = doc[previous.start : entity.end]
            else:
                merged_entities.append(entity)

        return merged_entities

    def tokenize_medical_text(self, text: str) -> list[str]:
        """Extract biomedical entities, falling back to regular tokens if needed."""
        doc = self.nlp(text)
        merged_entities = self._merge_adjacent_entities(doc)
        entity_tokens = [entity.text.strip() for entity in merged_entities if entity.text.strip()]
        prioritized_entities = [token for token in entity_tokens if len(token.split()) > 1 or any(char.isdigit() for char in token)]
        if prioritized_entities:
            # Keep output stable and avoid duplicate entity strings.
            return list(dict.fromkeys(prioritized_entities))
        if entity_tokens:
            return list(dict.fromkeys(entity_tokens))

        return [token.text for token in doc if not token.is_space and not token.is_punct]


def main() -> None:
    pipeline = ClinicalTextPipeline()
    sample_text = "Patient presents with acute myocardial infarction and Type-2 Diabetes."
    cleaned = pipeline.clean_text(sample_text)
    de_identified = pipeline.de_identify(cleaned)
    medical_tokens = pipeline.tokenize_medical_text(de_identified)
    print(medical_tokens)


if __name__ == "__main__":
    main()
