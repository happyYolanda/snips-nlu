# coding=utf-8
from __future__ import unicode_literals

from pathlib import Path

from mock import patch

from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.entity_parser import CustomEntityParser
from snips_nlu.entity_parser.custom_entity_parser import \
    CustomEntityParserUsage
from snips_nlu.tests.utils import FixtureTest

DATASET = {
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
}
DATASET = validate_and_format_dataset(DATASET)


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


class TestCustomEntityParser(FixtureTest):

    @patch("snips_nlu_ontology.GazetteerEntityParser.persist")
    def test_should_persist_unfitted_parser(self, mocked_persist):
        # Given
        parser = CustomEntityParser(CustomEntityParserUsage.WITHOUT_STEMS)

        # When
        self.tmp_file_path.mkdir()
        parser_path = self.tmp_file_path / 'custom_entity_parser'
        parser.persist(parser_path)

        # Then
        expected_model = {
            "parser": None,
            "entities": None,
            "parser_usage": 1
        }
        expected_metadata = {"unit_name": "custom_entity_parser"}
        self.assertJsonContent(
            parser_path / "custom_entity_parser.json", expected_model)
        self.assertJsonContent(
            parser_path / "metadata.json", expected_metadata)
        mocked_persist.assert_not_called()

    @patch("snips_nlu_ontology.GazetteerEntityParser.persist")
    def test_should_persist_fitted_parser(self, mocked_persist):
        # Given
        mocked_persist.side_effect = _persist_parser
        parser = CustomEntityParser(
            CustomEntityParserUsage.WITHOUT_STEMS).fit(DATASET)

        # When
        self.tmp_file_path.mkdir()
        parser_path = self.tmp_file_path / "custom_entity_parser"
        parser.persist(parser_path)

        # Then
        parser_name = "parser"
        expected_model = {
            "parser": parser_name,
            "entities": ["dummy_entity_1", "dummy_entity_2"],
            "parser_usage": 1
        }
        expected_metadata = {"unit_name": "custom_entity_parser"}
        self.assertJsonContent(
            parser_path / "custom_entity_parser.json", expected_model)
        self.assertJsonContent(
            parser_path / "metadata.json", expected_metadata)
        _parser_path = parser_path / parser_name
        self.assertTrue(_parser_path.exists())
        with _parser_path.open("r", encoding="utf-8") as f:
            content = f.read()
        self.assertEqual("nothing interesting here", content.strip())

    @patch("snips_nlu_ontology.GazetteerEntityParser.from_path")
    def test_should_load_unfitted_parser(self, mocked_load):
        # Given
        parser_model = {
            "entities": None,
            "parser_usage": 1,
            "parser": None
        }
        metadata = {"unit_name": "custom_entity_parser"}

        self.tmp_file_path.mkdir()
        self.writeJsonContent(self.tmp_file_path / "metadata.json", metadata)
        self.writeJsonContent(
            self.tmp_file_path / "custom_entity_parser.json",
            parser_model
        )

        # When
        parser = CustomEntityParser.from_path(self.tmp_file_path)

        # Then
        self.assertEqual(CustomEntityParserUsage.WITHOUT_STEMS,
                         parser.parser_usage)
        self.assertIsNone(parser.entities)
        self.assertIsNone(parser._parser)

        mocked_load.assert_not_called()

    @patch("snips_nlu_ontology.GazetteerEntityParser.from_path")
    def test_should_load_fitted_parser(self, from_path):
        # Given
        from_path.side_effect = _load_parser
        expected_entities = {"dummy_entity_1", "dummy_entity_2"}

        _parser_name = "parser"
        _parser_path = self.tmp_file_path / _parser_name
        parser_model = {
            "entities": list(expected_entities),
            "parser_usage": 1,
            "parser": _parser_name
        }
        metadata = {"unit_name": "custom_entity_parser"}

        self.tmp_file_path.mkdir()
        self.writeJsonContent(self.tmp_file_path / "metadata.json", metadata)
        self.writeJsonContent(
            self.tmp_file_path / "custom_entity_parser.json",
            parser_model
        )

        with _parser_path.open("w", encoding="utf-8") as f:
            f.write("this is supposed to be as parser")

        # When
        parser = CustomEntityParser.from_path(self.tmp_file_path)

        # Then
        self.assertEqual(expected_entities, parser.entities)
        self.assertEqual(
            CustomEntityParserUsage.WITHOUT_STEMS, parser.parser_usage)
        self.assertEqual("this is supposed to be as parser", parser._parser)

    def test_should_fit_and_parse(self):
        # Given
        parser = CustomEntityParser(
            CustomEntityParserUsage.WITHOUT_STEMS).fit(DATASET)
        text = "dummy_entity_1 dummy_1 dummy_entity_2 dummy_2"

        # When
        result = parser.parse(text)

        # Then
        self.assertEqual(4, len(result))
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

        for ent in result:
            self.assertIn(ent, expected_entities)

    @patch("snips_nlu.entity_parser.custom_entity_parser.stem")
    def test_should_parse_with_stems(self, mocked_stem):
        # Given
        mocked_stem.side_effect = _stem
        parser = CustomEntityParser(
            CustomEntityParserUsage.WITH_STEMS).fit(DATASET)
        text = "dummy_entity_ dummy_1"
        scope = ["dummy_entity_1"]

        # When
        result = parser.parse(text, scope=scope)

        # Then
        self.assertEqual(1, len(result))
        ent = result[0]
        expected_ent = {
            "value": "dummy_entity_",
            "resolved_value": "dummy_entity_1",
            "range": {
                "start": 0,
                "end": 13
            },
            "entity_identifier": "dummy_entity_1"
        }
        self.assertDictEqual(expected_ent, ent)

    @patch("snips_nlu.entity_parser.custom_entity_parser.stem")
    def test_should_parse_with_and_without_stems(self, mocked_stem):
        # Given
        mocked_stem.side_effect = _stem
        parser = CustomEntityParser(
            CustomEntityParserUsage.WITH_AND_WITHOUT_STEMS).fit(DATASET)
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
        self.assertEqual(2, len(result))
        self.assertEqual(expected_entities, result)

    def test_should_respect_scope(self):
        # Given
        parser = CustomEntityParser(
            CustomEntityParserUsage.WITHOUT_STEMS).fit(DATASET)
        scope = ["dummy_entity_1"]
        text = "dummy_entity_2"

        # When
        result = parser.parse(text, scope=scope)

        # Then
        self.assertEqual(0, len(result))

    @patch("snips_nlu_ontology.GazetteerEntityParser.parse")
    def test_should_use_cache(self, mocked_parse):
        # Given
        mocked_parse.return_value = []
        parser = CustomEntityParser(
            CustomEntityParserUsage.WITHOUT_STEMS).fit(DATASET)

        text = ""

        # When
        parser.parse(text)
        parser.parse(text)

        # Then
        self.assertEqual(1, mocked_parse.call_count)
