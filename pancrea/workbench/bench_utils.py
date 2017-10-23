# -*- coding: utf-8 -*-
"""
@Author: rlane
@Date:   20-10-2017 10:39:06
@Last Modified by:   rlane
@Last Modified time: 20-10-2017 17:52:08
"""

from itertools import product


def dict_product(arg_dict):
    """
    """
    # Wrap all non-list dict values (floats/ints) in a list
    # so that they can be handled by product
    for k, v in arg_dict.items():
        try:
            arg_dict[k] = [float(v)]
        except TypeError:
            pass

    # Create cartesian product from dict values
    dict_list = []
    for val in product(*arg_dict.values()):
        dict_list.append(dict(zip(arg_dict, val)))

    return dict_list
