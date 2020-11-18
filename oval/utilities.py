"""utilities: Utility functions

.. contents:: TOC
   :local:


Description
===========

Usage
=====

Interactive Usage
~~~~~~~~~~~~~~~~~

Commandline Usage
~~~~~~~~~~~~~~~~~

Utilities Listing
=================
"""

import os
import toml
from pathlib import Path as _Path

import logging

import pandas as pd

CONF = toml.load(_Path(__file__).parent.joinpath("conf.toml"))

LOGFORMAT = CONF["LOGGING"]["format"]
logging.basicConfig(format=LOGFORMAT)
logger = logging.getLogger(__name__)
logger.setLevel("INFO")


def to_excel(dfs, sht_names, filepath, replace=False):
    """export dataframes as excel workbook.

    Parameters
    ==========

    dfs : list of dfs or single dataframe.

    sht_names : list of strings or single string

    filepath : str

    replace: bool

    Returns
    =======
    Export dataframes as excel workbook.
    """

    exist = os.path.isfile(filepath)

    if exist and not replace:
        raise FileExistsError("File already exist")
        return

    writer = pd.ExcelWriter(filepath, engine="xlsxwriter")

    if isinstance(dfs, list):
        for sht_names, df in zip(sht_names, dfs):
            df.to_excel(writer, sht_names)
    else:
        dfs.to_excel(writer, sht_names)

    writer.save()

    return
