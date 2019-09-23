from typing import Dict, List
import torch
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder


class ElmoCrossLingualTokenEmbedder:
    __elmo_embedders__: Dict[str, ElmoEmbedder] = {}
    __aligning_files_layer__: Dict[str, str] = {
        "en_0": "../CrossLingualELMo/alignment_matrices/en_0.pth",
        "en_1": "../CrossLingualELMo/alignment_matrices/en_1.pth",
        "en_2": "../CrossLingualELMo/alignment_matrices/en_2.pth",
        "es_0": "../CrossLingualELMo/alignment_matrices/es_0.pth",
        "es_1": "../CrossLingualELMo/alignment_matrices/es_1.pth",
        "es_2": "../CrossLingualELMo/alignment_matrices/es_2.pth",
        "fr_0": "../CrossLingualELMo/alignment_matrices/fr_0.pth",
        "fr_1": "../CrossLingualELMo/alignment_matrices/fr_1.pth",
        "fr_2": "../CrossLingualELMo/alignment_matrices/fr_2.pth",
        "it_0": "../CrossLingualELMo/alignment_matrices/it_0.pth",
        "it_1": "../CrossLingualELMo/alignment_matrices/it_1.pth",
        "it_2": "../CrossLingualELMo/alignment_matrices/it_2.pth",
        "pt_0": "../CrossLingualELMo/alignment_matrices/pt_0.pth",
        "pt_1": "../CrossLingualELMo/alignment_matrices/pt_1.pth",
        "pt_2": "../CrossLingualELMo/alignment_matrices/pt_2.pth",
        "de_0": "../CrossLingualELMo/alignment_matrices/de_0.pth",
        "de_1": "../CrossLingualELMo/alignment_matrices/de_1.pth",
        "de_2": "../CrossLingualELMo/alignment_matrices/de_2.pth",
    }
    __aligning_matrices = {
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
        if lang not in ElmoCrossLingualTokenEmbedder.__elmo_embedders__:
            if lang not in ElmoCrossLingualTokenEmbedder.__options_files__ \
                    and lang not in ElmoCrossLingualTokenEmbedder.__weight_files__:
                raise ValueError("Cannot find Elmo cross-lingual model files for language '%s'" % lang)

            embedder = ElmoEmbedder(options_file=ElmoCrossLingualTokenEmbedder.__options_files__.get(lang),
                                    weight_file=ElmoCrossLingualTokenEmbedder.__weight_files__.get(lang),
                                    cuda_device=0)
            ElmoCrossLingualTokenEmbedder.__elmo_embedders__.update({lang: embedder})

        return ElmoCrossLingualTokenEmbedder.__elmo_embedders__.get(lang)

    @staticmethod
    def __get_elmo_alignment_matrix__(layer: int, lang: str):
        key = lang + "_" + str(layer)
        if key not in ElmoCrossLingualTokenEmbedder.__aligning_files_layer__:
            raise ValueError(
                "Cannot find ELMo cross-lingual alignment matrix for language '%s' and layer '%s'" % lang % layer)
        if key not in ElmoCrossLingualTokenEmbedder.__aligning_matrices:
            ElmoCrossLingualTokenEmbedder.__aligning_matrices. \
                update({key: torch.load(ElmoCrossLingualTokenEmbedder.__aligning_files_layer__.get(key))})

        return ElmoCrossLingualTokenEmbedder.__aligning_matrices.get(key)

    @staticmethod
    def get_elmo_embed_layer_1(passage: List[str], lang: str):
        passage.insert(0, "<S>")
        passage.append("</S>")
        embeddings = ElmoCrossLingualTokenEmbedder.__get_elmo_embedder__(lang).embed_sentence(passage)
        embeddings = [i[1:-1] for i in embeddings][1]
        # alignment_matrix_0 = torch.load(ElmoTokenEmbedder.__get_elmo_alignment_matrix__(0, lang), pickle_load_args={"encoding":"utf-8"})
        alignment_matrix_1 = ElmoCrossLingualTokenEmbedder.__get_elmo_alignment_matrix__(1, lang)
        # alignment_matrix_2 = torch.load(ElmoTokenEmbedder.__get_elmo_alignment_matrix__(2, lang))
        aligned_embeddings = np.matmul(embeddings, alignment_matrix_1.transpose())
        return aligned_embeddings
