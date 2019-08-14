from typing import Dict, List
import torch
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder


class ElmoTokenEmbedder:
    __elmo_embedders__: Dict[str, ElmoEmbedder] = {}
    __aligning_files_layer__: Dict[str, str] = {
        "en_0": "/cs/labs/oabend/ofir.arviv/CrossLingualELMo/alignment_matrices/en_0.pth",
        "en_1": "/cs/labs/oabend/ofir.arviv/CrossLingualELMo/alignment_matrices/en_1.pth",
        "en_2": "/cs/labs/oabend/ofir.arviv/CrossLingualELMo/alignment_matrices/en_2.pth",
        "fr_0": "/cs/labs/oabend/ofir.arviv/CrossLingualELMo/alignment_matrices/fr_0.pth",
        "fr_1": "/cs/labs/oabend/ofir.arviv/CrossLingualELMo/alignment_matrices/fr_1.pth",
        "fr_2": "/cs/labs/oabend/ofir.arviv/CrossLingualELMo/alignment_matrices/fr_2.pth",
        "de_0": "/cs/labs/oabend/ofir.arviv/CrossLingualELMo/alignment_matrices/de_0.pth",
        "de_1": "/cs/labs/oabend/ofir.arviv/CrossLingualELMo/alignment_matrices/de_1.pth",
        "de_2": "/cs/labs/oabend/ofir.arviv/CrossLingualELMo/alignment_matrices/de_2.pth",
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
                    and lang not in ElmoTokenEmbedder.__weight_files__:
                raise ValueError("Cannot find Elmo cross-lingual model files for language '%s'" % lang)

            embedder = ElmoEmbedder(options_file=ElmoTokenEmbedder.__options_files__.get(lang),
                                    weight_file=ElmoTokenEmbedder.__weight_files__.get(lang))
            ElmoTokenEmbedder.__elmo_embedders__.update({lang: embedder})

        return ElmoTokenEmbedder.__elmo_embedders__.get(lang)

    @staticmethod
    def __get_elmo_alignment_matrix__(layer: int, lang: str):
        if lang+"_"+str(layer) not in ElmoTokenEmbedder.__aligning_files_layer__:
            raise ValueError("Cannot find ELMo cross-lingual alignment matrix for language '%s' and layer '%s'" % lang % layer)

        return ElmoTokenEmbedder.__aligning_files_layer__.get(lang+"_"+str(layer))

    @staticmethod
    def get_elmo_embed_layer_1(passage: List[str], lang: str):
        passage.insert(0, "<S>")
        passage.append("</S>")
        embeddings = ElmoTokenEmbedder.__get_elmo_embedder__(lang).embed_sentence(passage)
        embeddings = [i[1:-1] for i in embeddings]
        # alignment_matrix_0 = torch.load(ElmoTokenEmbedder.__get_elmo_alignment_matrix__(0, lang), pickle_load_args={"encoding":"utf-8"})
        alignment_matrix_1 = torch.load(ElmoTokenEmbedder.__get_elmo_alignment_matrix__(1, lang))
        # alignment_matrix_2 = torch.load(ElmoTokenEmbedder.__get_elmo_alignment_matrix__(2, lang))
        aligned_embeddings = np.matmul(embeddings, alignment_matrix_1.transpose())
        return aligned_embeddings
