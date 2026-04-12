import re
import time
import warnings
from pathlib import Path

import pandas as pd
from negspacy.negation import Negex  # noqa: F401
import scispacy  # noqa: F401
from scispacy.linking import EntityLinker  # noqa: F401
import spacy
from confection import ConfigValidationError
from spacy.language import Language
from spacy.tokens import Doc, Span


warnings.filterwarnings("ignore", category=UserWarning, module=r"spacy(\.|$)")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"spacy(\.|$)")
warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn(\.|$)")


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

        if "scispacy_linker" not in self.nlp.pipe_names:
            self.nlp.add_pipe(
                "scispacy_linker",
                config={"resolve_abbreviations": True, "linker_name": "umls"},
                last=True,
            )

        if "negex" not in self.nlp.pipe_names:
            negex_config: dict[str, list[str]] = {"ent_types": ["ENTITY"]}
            ner_labels = list(self.nlp.pipe_labels.get("ner", ()))
            if ner_labels and "ENTITY" not in ner_labels:
                negex_config["ent_types"] = ner_labels
            self.nlp.add_pipe("negex", config=negex_config, last=True)

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

    def _merge_adjacent_entities(self, doc: Doc) -> list[list[Span]]:
        """Join touching entities so multi-token medical concepts stay intact."""
        merged_entity_groups: list[list[Span]] = []

        for entity in doc.ents:
            if not merged_entity_groups:
                merged_entity_groups.append([entity])
                continue

            previous = merged_entity_groups[-1][-1]
            between_tokens = doc[previous.end : entity.start]
            should_merge = all(
                token.is_space
                or token.is_punct
                or token.text.lower() in {"of", "for", "to", "in"}
                for token in between_tokens
            )

            if should_merge:
                merged_entity_groups[-1].append(entity)
            else:
                merged_entity_groups.append([entity])

        return merged_entity_groups

    def _get_top_group_cui(self, entity_group: list[Span]) -> str | None:
        """Return the best CUI candidate for a grouped entity span."""
        if not Span.has_extension("kb_ents"):
            return None

        ranked_candidates: list[tuple[int, float, str]] = []
        for entity in entity_group:
            kb_entities = list(entity._.kb_ents)
            if not kb_entities:
                continue

            top_cui, top_score = kb_entities[0]
            ranked_candidates.append((entity.end - entity.start, float(top_score), str(top_cui)))

        if not ranked_candidates:
            return None

        ranked_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return ranked_candidates[0][2]

    def tokenize_medical_text(self, text: str) -> list[str]:
        """Extract affirmed biomedical entities as CUIs with text fallback."""
        doc = self.nlp(text)
        merged_entity_groups = self._merge_adjacent_entities(doc)
        has_negex_extension = Span.has_extension("negex")

        entity_candidates: list[tuple[str, bool, str | None]] = []
        for group in merged_entity_groups:
            merged_span = doc[group[0].start : group[-1].end]
            is_negated = has_negex_extension and any(bool(entity._.negex) for entity in group)
            top_cui = self._get_top_group_cui(group)
            entity_candidates.append((merged_span.text.strip(), is_negated, top_cui))

        # Option A: filter negated entities from the feature space.
        # Future enhancement (Option B): keep terms and add a "_NEG" suffix.
        affirmed_entity_features = [
            top_cui if top_cui else entity_text
            for entity_text, is_negated, top_cui in entity_candidates
            if entity_text and not is_negated
        ]

        prioritized_features = [
            top_cui if top_cui else entity_text
            for entity_text, is_negated, top_cui in entity_candidates
            if entity_text
            and not is_negated
            and (len(entity_text.split()) > 1 or any(char.isdigit() for char in entity_text))
        ]
        if prioritized_features:
            # Keep output stable and avoid duplicate entity strings.
            return list(dict.fromkeys(prioritized_features))

        if affirmed_entity_features:
            return list(dict.fromkeys(affirmed_entity_features))

        if entity_candidates:
            return []

        return [token.text for token in doc if not token.is_space and not token.is_punct]


def process_dataframe(df: pd.DataFrame, pipeline: ClinicalTextPipeline | None = None) -> pd.DataFrame:
    """Clean rows and extract clinical CUI/token features from transcriptions."""
    required_columns = {"transcription", "medical_specialty"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing}")

    working_df = df.dropna(subset=["transcription", "medical_specialty"]).copy()
    working_df["medical_specialty"] = working_df["medical_specialty"].astype(str).str.strip()
    working_df = working_df[working_df["medical_specialty"] != ""].copy()

    if pipeline is None:
        pipeline = ClinicalTextPipeline()

    def _extract_features(transcription: str) -> list[str]:
        cleaned_text = pipeline.clean_text(transcription)
        de_identified_text = pipeline.de_identify(cleaned_text)
        return pipeline.tokenize_medical_text(de_identified_text)

    working_df["extracted_features"] = working_df["transcription"].astype(str).apply(_extract_features)
    return working_df


def main() -> None:
    dataset_path = Path("mtsamples.csv")
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Place mtsamples.csv in the project root."
        )

    dataset = pd.read_csv(dataset_path)
    sample_df = dataset.head(10)

    pipeline = ClinicalTextPipeline()
    smoke_features = pipeline.tokenize_medical_text(
        pipeline.de_identify(pipeline.clean_text("Patient presents with a heart attack."))
    )
    assert "C0027051" in smoke_features, "Expected UMLS CUI C0027051 for 'heart attack'."

    start_time = time.perf_counter()
    processed_sample = process_dataframe(sample_df, pipeline=pipeline)
    elapsed_seconds = time.perf_counter() - start_time

    print(f"Processed {len(processed_sample)} valid rows from head(10) in {elapsed_seconds:.2f}s.")
    print(processed_sample[["medical_specialty", "extracted_features"]].to_string(index=False))


if __name__ == "__main__":
    main()
