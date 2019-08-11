from typing import Dict, List
from allennlp.commands.elmo import ElmoEmbedder


class ElmoTokenEmbedder:
    __elmo_embedders__: Dict[str, ElmoEmbedder] = {}
    __aligning_files__: Dict[str, str] = {
        "en": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/en_best_mapping.pth",
        "es": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/es_best_mapping.pth",
        "fr": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/fr_best_mapping.pth",
        "it": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/it_best_mapping.pth",
        "pt": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/pt_best_mapping.pth",
        "sv": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/sv_best_mapping.pth",
        "de": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/de_best_mapping.pth"
    }
    __options_files__: Dict[str, str] = {
        "en": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/options262.json",
        "es": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/options262.json",
        "fr": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/options262.json",
        "it": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/options262.json",
        "pt": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/options262.json",
        "sv": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/options262.json",
        "de": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/options262.json"
    }
    __weight_files__: Dict[str, str] = {
        "en": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/en_weights.hdf5",
        "es": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/es_weights.hdf5",
        "fr": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/fr_weights.hdf5",
        "it": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/it_weights.hdf5",
        "pt": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/pt_weights.hdf5",
        "sv": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/sv_weights.hdf5",
        "de": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/de_weights.hdf5"
    }

    @staticmethod
    def __get_elmo_embedder__(lang: str):
        if lang not in ElmoTokenEmbedder.__elmo_embedders__:
            if lang not in ElmoTokenEmbedder.__options_files__ \
                    and lang not in ElmoTokenEmbedder.__weight_files__ \
                    and lang not in ElmoTokenEmbedder.__aligning_files__:
                raise ValueError("Cannot find Elmo cross-lingual files for language '%s'" % lang)

            embedder = ElmoEmbedder(options_file=ElmoTokenEmbedder.__options_files__.get(lang),
                                    weight_file=ElmoTokenEmbedder.__weight_files__.get(lang))
            ElmoTokenEmbedder.__elmo_embedders__.update({lang: embedder})

        return ElmoTokenEmbedder.__elmo_embedders__.get(lang)

    @staticmethod
    def get_elmo_embed(passage: List[str], lang: str):
        passage.insert(0, "<S>")
        passage.append("</S>")
        embed = ElmoTokenEmbedder.__get_elmo_embedder__(lang).embed_sentence(passage)
        return embed
