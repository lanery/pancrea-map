# -*- coding: utf-8 -*-
"""
@Author: rlane
@Date:   20-10-2017 10:39:06
@Last Modified by:   rlane
@Last Modified time: 23-10-2017 15:33:33
"""

from itertools import product


def dict_product(arg_dict):
    """
    Take the Cartesian product of a dictionary of lists

    Parameters
    ----------
    arg_dict : dict
        Dictionary of lists or scalars
        If a particular value within the dict is scalar, it will be converted
        to a (length 1) list to make itertools product happy

    Returns
    -------
    dict_list : list
        Cartesian product of 

    Notes
    -----
    https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists

    Examples
    --------
    >>> params = {'number': [1, 2, 3],
                  'color': ['orange', 'blue']}

    >>> dict_product(params)

    [ {"number": 1, "color": "orange"},
      {"number": 1, "color": "blue"},
      {"number": 2, "color": "orange"},
      {"number": 2, "color": "blue"},
      {"number": 3, "color": "orange"},
      {"number": 3, "color": "blue"}
    ]
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
