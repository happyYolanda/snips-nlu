from __future__ import unicode_literals

import json
import shutil
from pathlib import Path

from snips_nlu_ontology import (BuiltinEntityParser as _BuiltinEntityParser,
                                get_all_builtin_entities,
                                get_all_gazetteer_entities,
                                get_all_grammar_entities,
                                get_builtin_entity_shortname,
                                get_supported_gazetteer_entities)

from snips_nlu.constants import DATA_PATH, ENTITIES, LANGUAGE
from snips_nlu.entity_parser.entity_parser import EntityParser
from snips_nlu.utils import json_string, temp_dir


class BuiltinEntityParser(EntityParser):
    def __init__(self, language, gazetteer_entities):
        self.language = language
        self.gazetteer_entities = gazetteer_entities
        with temp_dir() as serialization_dir:
            _build_builtin_parser_dir(
                serialization_dir, self.gazetteer_entities, self.language)
            self._parser = _BuiltinEntityParser.from_path(serialization_dir)

    @property
    def parser(self):
        return self._parser

    def persist(self, path):
        path = Path(path)
        parser_file_name = "gazetteer_entity_parser"
        metadata = {
            "language": self.language.upper(),
            "gazetteer_parser": parser_file_name
        }
        metadata_path = path / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            f.write(json_string(metadata))
        parser_path = path / parser_file_name
        self._parser.persist(parser_path)


def _build_builtin_parser_dir(target_dir, gazetteer_entities, language):
    target_dir = Path(target_dir)
    gazetteer_entity_parser = None

    if gazetteer_entities:
        parser_name = _build_gazetteer_parser(target_dir, gazetteer_entities,
                                              language)
        gazetteer_entity_parser = parser_name

    metadata = {
        "language": language.upper(),
        "gazetteer_parser": gazetteer_entity_parser
    }
    metadata_path = target_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        f.write(json_string(metadata))


def _build_gazetteer_parser(target_dir, gazetteer_entities, language):
    gazetteer_parser_name = "gazetteer_entity_parser"
    gazetteer_parser_path = target_dir / gazetteer_parser_name
    for ent in gazetteer_entities:
        if ent not in get_supported_gazetteer_entities(language):
            raise ValueError(
                "%s does not support %s as a builtin gazetteer entity"
                % (language, ent)
            )
    source_gazetteers_dir = DATA_PATH
    gazetteer_parser_metadata = []
    for ent in sorted(gazetteer_entities):
        # Fetch the compiled parser in the resources
        source_parser_path = find_gazetteer_entity_data_path(language, ent)
        short_name = get_builtin_entity_shortname(ent).lower()
        target_parser_path = gazetteer_parser_path / short_name
        parser_metadata = {
            "entity_identifier": ent,
            "entity_parser": short_name
        }
        gazetteer_parser_metadata.append(parser_metadata)
        # Copy the single entity entity parser
        shutil.copytree(source_parser_path, target_parser_path)
    # Dump the parser metadata
    gazetteer_entity_parser_metadata = {
        "parsers_metadata": gazetteer_parser_metadata
    }
    gazetteer_parser_metadata_path = gazetteer_parser_path / \
                                     "metadata.json"
    with gazetteer_parser_metadata_path.open("w", encoding="utf-8") as f:
        f.write(json_string(gazetteer_entity_parser_metadata))
    return gazetteer_parser_name


_BUILTIN_ENTITY_PARSERS = dict()


def get_builtin_entity_parser(dataset):
    language = dataset[LANGUAGE]
    gazetteer_entities = [entity for entity in dataset[ENTITIES]
                          if is_gazetteer_entity(entity)]
    return get_builtin_entity_parser_from_scope(language, gazetteer_entities)


def get_builtin_entity_parser_from_scope(language, gazetteer_entity_scope):
    global _BUILTIN_ENTITY_PARSERS
    caching_key = _get_caching_key(language, gazetteer_entity_scope)
    if caching_key not in _BUILTIN_ENTITY_PARSERS:
        for entity in gazetteer_entity_scope:
            if entity not in get_supported_gazetteer_entities(language):
                raise ValueError("Gazetteer entity '%s' is not supported in "
                                 "language '%s'" % (entity, language))
        _BUILTIN_ENTITY_PARSERS[caching_key] = BuiltinEntityParser(
            language, gazetteer_entity_scope)
    return _BUILTIN_ENTITY_PARSERS[caching_key]


def is_builtin_entity(entity_label):
    return entity_label in get_all_builtin_entities()


def is_gazetteer_entity(entity_label):
    return entity_label in get_all_gazetteer_entities()


def is_grammar_entity(entity_label):
    return entity_label in get_all_grammar_entities()


def find_gazetteer_entity_data_path(language, entity_name):
    for directory in DATA_PATH.iterdir():
        if not directory.is_dir():
            continue
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            continue
        with metadata_path.open(encoding="utf8") as f:
            metadata = json.load(f)
        if metadata.get("entity_name") == entity_name \
                and metadata.get("language") == language:
            return directory / metadata["data_directory"]
    raise FileNotFoundError(
        "No data found for the '{e}' builtin entity in language '{lang}'. "
        "You must download the corresponding resources by running "
        "'python -m snips_nlu download-entity {e} {lang}' before you can use "
        "this builtin entity.".format(e=entity_name, lang=language))


def _get_gazetteer_entity_configurations(language, gazetteer_entity_scope):
    return [{
        "builtin_entity_name": entity_name,
        "resource_path": str(find_gazetteer_entity_data_path(
            language, entity_name))
    } for entity_name in gazetteer_entity_scope]


def _get_caching_key(language, entity_scope):
    tuple_key = (language,)
    tuple_key += tuple(entity for entity in sorted(entity_scope))
    return tuple_key
