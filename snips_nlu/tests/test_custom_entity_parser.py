# coding=utf-8
from __future__ import unicode_literals

from pathlib import Path

from mock import patch

from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.entity_parser import CustomEntityParser
from snips_nlu.entity_parser.custom_entity_parser import \
    CustomEntityParserUsage
from snips_nlu.tests.utils import FixtureTest

DATASET = validate_and_format_dataset({
    "intents": {

    },
    "entities": {
        "dummy_entity_1": {
            "data": [
                {
                    "value": "dummy_entity_1",
                    "synonyms": ["dummy_1"]
                }
            ],
            "use_synonyms": True,
            "automatically_extensible": True,
            "parser_threshold": 1.0
        },
        "dummy_entity_2": {
            "data": [
                {
                    "value": "dummy_entity_2",
                    "synonyms": ["dummy_2"]
                }
            ],
            "use_synonyms": True,
            "automatically_extensible": True,
            "parser_threshold": 1.0
        }
    },
    "language": "en"
})


class TestCustomEntityParser(FixtureTest):
    def test_should_parse_without_stems(self):
        # Given
        parser = CustomEntityParser.build(
            DATASET, CustomEntityParserUsage.WITHOUT_STEMS)
        text = "dummy_entity_1 dummy_1 dummy_entity_2 dummy_2"

        # When
        result = parser.parse(text)

        # Then
        expected_entities = [
            {
                "value": "dummy_entity_1",
                "resolved_value": "dummy_entity_1",
                "range": {
                    "start": 0,
                    "end": 14
                },
                "entity_identifier": "dummy_entity_1"
            },
            {
                "value": "dummy_1",
                "resolved_value": "dummy_entity_1",
                "range": {
                    "start": 15,
                    "end": 22
                },
                "entity_identifier": "dummy_entity_1"
            },
            {
                "value": "dummy_entity_2",
                "resolved_value": "dummy_entity_2",
                "range": {
                    "start": 23,
                    "end": 37
                },
                "entity_identifier": "dummy_entity_2"
            },
            {
                "value": "dummy_2",
                "resolved_value": "dummy_entity_2",
                "range": {
                    "start": 38,
                    "end": 45
                },
                "entity_identifier": "dummy_entity_2"
            }
        ]
        self.assertListEqual(expected_entities, result)

    @patch("snips_nlu.entity_parser.custom_entity_parser.stem")
    def test_should_parse_with_stems(self, mocked_stem):
        # Given
        mocked_stem.side_effect = _stem
        parser = CustomEntityParser.build(
            DATASET, CustomEntityParserUsage.WITH_STEMS)
        text = "dummy_entity_ dummy_1"
        scope = ["dummy_entity_1"]

        # When
        result = parser.parse(text, scope=scope)

        # Then
        expected_entities = [
            {
                "value": "dummy_entity_",
                "resolved_value": "dummy_entity_1",
                "range": {
                    "start": 0,
                    "end": 13
                },
                "entity_identifier": "dummy_entity_1"
            }
        ]
        self.assertListEqual(expected_entities, result)

    @patch("snips_nlu.entity_parser.custom_entity_parser.stem")
    def test_should_parse_with_and_without_stems(self, mocked_stem):
        # Given
        mocked_stem.side_effect = _stem
        parser = CustomEntityParser.build(
            DATASET, CustomEntityParserUsage.WITH_AND_WITHOUT_STEMS)
        scope = ["dummy_entity_1"]
        text = "dummy_entity_ dummy_1"

        # When
        result = parser.parse(text, scope=scope)

        # Then
        expected_entities = [
            {
                "value": "dummy_entity_",
                "resolved_value": "dummy_entity_1",
                "range": {
                    "start": 0,
                    "end": 13
                },
                "entity_identifier": "dummy_entity_1"
            },
            {
                "value": "dummy_1",
                "resolved_value": "dummy_entity_1",
                "range": {
                    "start": 14,
                    "end": 21
                },
                "entity_identifier": "dummy_entity_1"
            }
        ]
        self.assertListEqual(expected_entities, result)

    def test_should_respect_scope(self):
        # Given
        parser = CustomEntityParser.build(
            DATASET, CustomEntityParserUsage.WITHOUT_STEMS)
        scope = ["dummy_entity_1"]
        text = "dummy_entity_2"

        # When
        result = parser.parse(text, scope=scope)

        # Then
        self.assertListEqual([], result)

    @patch("snips_nlu_ontology.GazetteerEntityParser.parse")
    def test_should_use_cache(self, mocked_parse):
        # Given
        mocked_parse.return_value = []
        parser = CustomEntityParser.build(
            DATASET, CustomEntityParserUsage.WITHOUT_STEMS)

        text = ""

        # When
        parser.parse(text)
        parser.parse(text)

        # Then
        self.assertEqual(1, mocked_parse.call_count)

    def test_should_be_serializable(self):
        # Given
        parser = CustomEntityParser.build(
            DATASET, CustomEntityParserUsage.WITHOUT_STEMS)
        self.tmp_file_path.mkdir()
        parser_path = self.tmp_file_path / "custom_entity_parser"
        parser.persist(parser_path)
        loaded_parser = CustomEntityParser.from_path(parser_path)

        # When
        scope = ["dummy_entity_1"]
        text = "dummy_entity_1 dummy_1"
        result = loaded_parser.parse(text, scope=scope)

        # Then
        expected_entities = [
            {
                "value": "dummy_entity_1",
                "resolved_value": "dummy_entity_1",
                "range": {
                    "start": 0,
                    "end": 14
                },
                "entity_identifier": "dummy_entity_1"
            },
            {
                "value": "dummy_1",
                "resolved_value": "dummy_entity_1",
                "range": {
                    "start": 15,
                    "end": 22
                },
                "entity_identifier": "dummy_entity_1"
            }
        ]
        self.assertListEqual(expected_entities, result)


# pylint: disable=unused-argument
def _persist_parser(path):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        f.write("nothing interesting here")


# pylint: disable=unused-argument
def _load_parser(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return f.read().strip()


# pylint: disable=unused-argument
def _stem(string, language):
    return string[:-1]
