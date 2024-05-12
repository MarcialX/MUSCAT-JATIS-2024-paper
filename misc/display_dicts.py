# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# MUSCAT data reduction
# display_dicts.py
# Display in a nice way the info of a dictionary
#
# Marcial Becerril, @ 23 June 2023
# Latest Revision: 23 Jun 2023, 15:16 GMT
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# mbecerrilt@inaoep.mx
#
# --------------------------------------------------------------------------------- #

import numpy as _np

from .msg_custom import *


def print_tree(data, show_basic_info=True):
    """
        Display all the fields of the file as a tree structure
    """

    # Color and style columns
    column_colors = [BOLD, OKBLUE, WARNING]
    column_style = ['*', '**', '->']

    # Extract all the fields
    fields = explore_fields(data, fields=[])

    # Display them
    n = 0
    for field in fields:
        sub_fields = field.split("/")
        tabs = ""
        if n == 0:
            prev_sub = [[]]*len(sub_fields)
        for i, sub in enumerate(sub_fields):
            if prev_sub[i] != sub:
                format_idx = i
                if i >= 3:
                    format_idx = 2

                msg = tabs+column_colors[format_idx]+column_style[format_idx]+sub+ENDC
                if show_basic_info and i == len(sub_fields)-1:
                    info = _get_data_from_tab_line(data, field)
                    if isinstance(info, bytes) or isinstance(info, int) or isinstance(info, str) or isinstance(info, float):
                        msg += ": "+str(info)
                    elif isinstance(info, _np.ma.core.MaskedArray):
                        if info.data.shape == ():
                            msg += ": "+str(info)
                        else:
                            msg += ": array "+str(info.data.shape)
                    else:
                        msg += ": other format"
                print(msg)
                prev_sub[i+1:] = [[]]*(len(prev_sub)-i+1)
            tabs += "\t"

        prev_sub = sub_fields
        n += 1


def explore_fields(data, prev="", fields=[]):

    for a in data.keys():
        if isinstance(data[a], dict):
            prev += a+"/"
            explore_fields(data[a], prev=prev, fields=fields)
            prev = prev[prev.index("/")+1:]
        else:
            fields.append(prev+a)

    return fields


def _get_data_from_tab_line(data, addr, sep="/"):

    for i in addr.split(sep):
        data = data[i]
    return data