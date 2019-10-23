from typing import List
from typing_extensions import Literal

import pandas
import ucca.core
import ucca.layer0

EdgeTags = Literal['E', 'C', 'A', 'U', 'R', 'P', 'H', 'D', 'F', 'L', 'S', 'N', 'G']
edge_tags = ['E', 'C', 'A', 'U', 'R', 'P', 'H', 'D', 'F', 'L', 'S', 'N', 'G']


def create_alignment_dict_from_alignment_string(alignment_str: str):
    arr = alignment_str.split()
    arr = [entry.split("-") for entry in arr]
    arr = [list(map(lambda x: "{}{}{}".format(ucca.layer0.LAYER_ID, ucca.core.Node.ID_SEPARATOR, int(x) + 1), sublist))
           for sublist in arr]
    alignment_dict = {}
    for sublist in arr:
        key = sublist[0]
        value = sublist[1]
        found_key = False
        for dict_key in alignment_dict.keys():
            if key in dict_key:
                dict_value = alignment_dict[dict_key]
                alignment_dict[dict_key] = dict_value + (value,)
                assert found_key is False
                found_key = True
        if not found_key:
            alignment_dict[(key,)] = (value,)

    full_iteration_without_changes = False
    while not full_iteration_without_changes:
        for i in range(len(alignment_dict)):
            key_1 = list(alignment_dict.keys())[i]
            to_break = False
            val_1 = alignment_dict[key_1]
            for j in range(i + 1, len(alignment_dict)):
                key_2 = list(alignment_dict.keys())[j]
                if key_1 is key_2:
                    assert False
                val_2 = alignment_dict[key_2]
                if not set(val_1).isdisjoint(val_2):
                    key = tuple(set(key_1 + key_2))
                    val = tuple(set(val_1 + val_2))
                    del alignment_dict[key_1]
                    del alignment_dict[key_2]
                    alignment_dict[key] = val
                    to_break = True
                    break
            if to_break:
                break
        if i == len(alignment_dict) - 1:
            full_iteration_without_changes = True

    return alignment_dict


def get_alignment_list_keys(alignment_file_path: str) -> pandas.DataFrame:
    dataframe = pandas.read_csv(alignment_file_path, skipinitialspace=True)
    dataframe = dataframe[dataframe.English.apply(lambda x: x.isnumeric())]
    dataframe = dataframe[dataframe.French.apply(lambda x: x.isnumeric())]
    dataframe = dataframe.reset_index()

    return dataframe


def get_alignment_list(alignment_file_path: str):
    alignment_list = []
    with open(alignment_file_path, 'r') as reader:
        line = reader.readline()
        while line:
            line = line.rstrip('\n')
            alignment_list.append(create_alignment_dict_from_alignment_string(line))
            line = reader.readline()

    return alignment_list


# region Passage Utilities
def get_primary_parent(node: ucca.core.Node) -> ucca.core.Node:
    assert isinstance(node, ucca.core.Node)

    primary_parents = []
    incoming = node.incoming
    for edge in incoming:
        if not edge.attrib.get("remote") and not edge.parent.attrib.get("implicit") \
                and edge.parent.tag != "LKG":
            primary_parents.append(edge.parent)

    assert len(primary_parents) == 1, "More than one primary parent for a node! UCCA trees structure enforce " \
                                      "one primary parent per node. The function is probably erroneous."
    return primary_parents[0]


def get_primary_incoming_edge(node: ucca.core.Node) -> EdgeTags:
    assert isinstance(node, ucca.core.Node)

    primary_edges: List[ucca.core.Edge] = []
    incoming = node.incoming
    for edge in incoming:
        if not edge.attrib.get("remote") and not edge.parent.attrib.get("implicit") \
                and edge.parent.tag != "LKG":
            primary_edges.append(edge)
    if len(primary_edges) == 0:
        return -1
    assert len(primary_edges) == 1, "More than one primary parent for a node! UCCA trees structure enforce " \
                                    "one primary parent per node. The function is probably erroneous."
    tag = primary_edges[0].tag
    assert tag in edge_tags, f'{tag} not in pre-defined EdgeTags'
    return edge_tags.index(tag)


def get_depth(node: ucca.core.Node) -> int:
    # TODO: Not good enough check because might have remote parent ect.
    if len(node.parents) == 0:
        return 0
    else:
        return get_depth(get_primary_parent(node))+1


# endregion
