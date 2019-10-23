from collections import OrderedDict
from enum import Enum
from typing import Union
from typing_extensions import Literal

import ucca
from ucca.core import Passage
from ucca.layer0 import Terminal

from tupa.alignment_utils import get_primary_parent, get_primary_incoming_edge, get_depth, EdgeTags, edge_tags
from tupa.features.feature_params import FeatureParameters, NumericFeatureParameters
from tupa.states.node import Node
from tupa.states.state import State


# --- for a single terminal ---
#  --Label - Embedded --
#     primary parent (edge) label
#     primary grandparent label

# So the vector will look like:
# s0-p-l s0-gp-l b0-p-l b0-gp-l

#  -- Numerical
#     depth
#     number of parents

# --- for a stack-terminal, buffer-node couple

# --- for a single

# for a couple of nodes [stack-node, buffer-node] or [stack, stack]
#   if terminals - are siblings?
#   if terminal and node - do the node have other terminals that are sibling wit the terminal?
#   if node and node - do they have terminal that indicate they are siblings?

class AlignmentFeatureExtractor:

    def __init__(self):
        self.features_params = AlignmentFeatureExtractor._get_features_params_dict()

    @staticmethod
    def _get_features_params_dict():
        features_params = OrderedDict()
        projected_labels_param = FeatureParameters(suffix="l", dim=50, size=len(edge_tags)+1, dropout=0, updated=True,
                                                   num=12, init=None, data=None, indexed=False, copy_from=None,
                                                   filename=None, min_count=1, enabled=True, node_dropout=0, vocab=None,
                                                   lang_specific=False)
        features_params["l"] = projected_labels_param

        projected_labels_param = NumericFeatureParameters(6)
        features_params["a_numeric"] = projected_labels_param

        return features_params

    def extract_features(self, state: State):
        # self._get_features_from_template("b0_p1_l b0_n_d")
        s0 = AlignmentFeatureExtractor._get_node(state, 'stack', 0)
        s0_parent_tag = AlignmentFeatureExtractor._extract_parent_tag(state, s0, 1)
        s0_grandparent_tag = AlignmentFeatureExtractor._extract_parent_tag(state, s0, 2)
        b0 = AlignmentFeatureExtractor._get_node(state, 'buffer', 0)
        b0_parent_tag = AlignmentFeatureExtractor._extract_parent_tag(state, b0, 1)
        b0_grandparent_tag = AlignmentFeatureExtractor._extract_parent_tag(state, b0, 2)

        s1 = AlignmentFeatureExtractor._get_node(state, 'stack', 1)
        s1_parent_tag = AlignmentFeatureExtractor._extract_parent_tag(state, s1, 1)
        s1_grandparent_tag = AlignmentFeatureExtractor._extract_parent_tag(state, s1, 2)
        b1 = AlignmentFeatureExtractor._get_node(state, 'buffer', 1)
        b1_parent_tag = AlignmentFeatureExtractor._extract_parent_tag(state, b1, 1)
        b1_grandparent_tag = AlignmentFeatureExtractor._extract_parent_tag(state, b1, 2)

        s2 = AlignmentFeatureExtractor._get_node(state, 'stack', 2)
        s2_parent_tag = AlignmentFeatureExtractor._extract_parent_tag(state, s2, 1)
        s2_grandparent_tag = AlignmentFeatureExtractor._extract_parent_tag(state, s2, 2)
        b2 = AlignmentFeatureExtractor._get_node(state, 'buffer', 2)
        b2_parent_tag = AlignmentFeatureExtractor._extract_parent_tag(state, b2, 1)
        b2_grandparent_tag = AlignmentFeatureExtractor._extract_parent_tag(state, b2, 2)

        s0_depth = AlignmentFeatureExtractor._extract_depth(state, s0.orig_node) if s0 else 0
        b0_depth = AlignmentFeatureExtractor._extract_depth(state, b0.orig_node) if b0 else 0
        s1_depth = AlignmentFeatureExtractor._extract_depth(state, s1.orig_node) if s1 else 0
        b1_depth = AlignmentFeatureExtractor._extract_depth(state, b1.orig_node) if b1 else 0
        s2_depth = AlignmentFeatureExtractor._extract_depth(state, s2.orig_node) if s2 else 0
        b2_depth = AlignmentFeatureExtractor._extract_depth(state, b2.orig_node) if b2 else 0

        features = OrderedDict()
        features['l'] = [s0_parent_tag, s0_grandparent_tag, b0_parent_tag, b0_grandparent_tag,
                         s1_parent_tag, s1_grandparent_tag, b1_parent_tag, b1_grandparent_tag,
                         s2_parent_tag, s2_grandparent_tag, b2_parent_tag, b2_grandparent_tag]
        assert self.features_params["l"].num == len(features['l'])
        features['a_numeric'] = [s0_depth, b0_depth, s1_depth, b1_depth, s2_depth, b2_depth]
        assert self.features_params["a_numeric"].num == len(features['a_numeric'])

        return features

    @staticmethod
    def _get_features_from_template(template: str, state: State):
        temp = [entry.split("_") for entry in template.split()]
        for source, relation, feature in temp:
            source_node = None
            if source[0] == 'b':
                source_node = AlignmentFeatureExtractor._get_node(state, 'buffer', source[1:])
            elif source[0] == 's':
                source_node = AlignmentFeatureExtractor._get_node(state, 'buffer', source[1:])
            else:
                raise Exception(f'Unknown source in template: {source}')

            # if relation[0] == 'p'

    @staticmethod
    def _extract_depth(state: State, node: Node) -> int:
        parallel_passage: Passage = state.passage.parallel_passage
        alignment_list = state.passage.alignment_fr_to_en
        depth = 0
        if isinstance(node, Node) and node.text is not None:  # is Terminal Node
            node_id = node.node_id
            aligned_id = AlignmentFeatureExtractor._get_aligned_ids(alignment_list, node_id)
            if len(aligned_id) == 1:  # 1-1 alignment
                aligned_id = aligned_id[0]
                aligned_terminal: Terminal = parallel_passage.by_id(aligned_id)
                assert isinstance(aligned_terminal, Terminal)
                depth = get_depth(aligned_terminal)
        return depth

    @staticmethod
    # TODO: hanlde the case of no parent of specific order
    # TODO: Chnge to standard passage function and have a function that get the aligned node
    def _extract_parent_tag(state: State, node: Node, parent_order: Literal[1, 2, 3]) -> EdgeTags:
        parent_tag = -1
        parallel_passage: Passage = state.passage.parallel_passage
        alignment_list = state.passage.alignment_fr_to_en
        if isinstance(node, Node) and node.text is not None:  # is Terminal Node
            node_id = node.node_id
            aligned_id = AlignmentFeatureExtractor._get_aligned_ids(alignment_list, node_id)
            if len(aligned_id) == 1:  # 1-1 alignment
                aligned_id = aligned_id[0]
                aligned_terminal: Terminal = parallel_passage.by_id(aligned_id)
                assert isinstance(aligned_terminal, Terminal)
                parent = aligned_terminal
                for i in range(parent_order):
                    parent = get_primary_parent(parent)

                parent_tag = get_primary_incoming_edge(parent)

        return parent_tag

    @staticmethod
    def _get_aligned_ids(alignment_list, source_id: str):
        for key in alignment_list.keys():
            if source_id in key:
                return alignment_list[key]
        return []

    SourceType = Literal['stack', 'buffer']

    @staticmethod
    def _get_node(state: State, source: SourceType, index: int) -> Union[Node, None]:
        assert source in ['stack', 'buffer']
        try:
            if source == "stack":
                return state.stack[-1 - index]
            else:  # source== "buffer":
                return state.buffer[index]
        except IndexError:
            return None
