"""portfolio is a module that provides tools to manage portfolio

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

portfolio Listing
=================
"""
from pathlib import Path as _Path
import pandas as pd
import plotly.express as px
from copy import deepcopy as _deepcopy
from pyarrow import feather

try:
    from asset import Stock, Cash
except (ImportError, ModuleNotFoundError):
    from oval.asset import Stock, Cash


class Position:
    """Building blocks of the portfolio class."""

    def __init__(self, Asset, shares):
        self._asset = Asset
        self._shares = shares
        self._value = self._calculate_value()

    def __repr__(self):
        return (
            f"<{self.shares} shares in {self.asset.ticker} ({self.asset.asset_class})>"
        )

    @property
    def asset(self):
        return self._asset

    @property
    def shares(self):
        return self._shares

    def update_shares(self, change_in_shares):
        self._shares += change_in_shares
        self._value = self._calculate_value()

    @property
    def value(self):
        return self._value

    def _calculate_value(self):
        return self.shares * self.asset.price

    def copy(self):
        return _deepcopy(self)


class Portfolio:
    """Portfolio consists of multiple Position objects.

    Parameters:
    -----------
    positions : list of Position objects or file path string

    """

    def __init__(self, positions):
        self._positions = self._load_positions(positions)
        self._time_idx = self.get_longest_time_index()
        self._val_date = self.time_idx[-1]
        self._summary = self._summarize()
        self._value = self.summary["value"].sum()

    @property
    def positions(self):
        return self._positions

    @property
    def summary(self):
        return self._summary

    @property
    def value(self):
        return self._value

    @property
    def time_idx(self):
        return self._time_idx

    @property
    def val_date(self):
        return self._val_date

    def _load_positions(self, positions):
        """
        Parameters
        ----------
        positions : list of Position objects or file path string

        Returns
        -------
        pos_dict : dictionary of Position objects
        """

        if isinstance(positions, list):
            pos_dict = {pos.asset.ticker: pos for pos in positions}

        elif isinstance(positions, _Path) or isinstance(positions, str):
            summary = feather.read_feather(positions)

            ticker = summary["ticker"]
            asset_class = summary["asset_class"]
            shares = summary["shares"]

            pos_dict = {}

            for i in zip(ticker, asset_class, shares):

                if i[1] == "Stock":
                    pos = Position(Stock(i[0], financials=True), i[2])
                elif i[1] == "Cash":
                    pos = Position(Cash(), i[2])

                pos_dict[i[0]] = pos

        return pos_dict

    def update(self, change_in_positions):
        """update positions in the portfolio. In addition, reflect
        summary and value accordingly.

        Parameters
        ==========
        new_positions : list of Position objects

        Returns
        =======

        """

        # <TODO> need fix, the change in position is not
        # reflected on portfolio level.
        _new_positions = {pos.asset.ticker: pos for pos in new_positions}
        self._positions.update(_new_positions)
        self._time_idx = self.get_longest_time_index()
        self._val_date = self.time_idx[-1]
        self._summary = self._summarize()
        self._value = self.summary["value"].sum()

    def _summarize(self):
        """Internal function that produce a summary of the portfolio."""

        _summary = {}
        _summary["ticker"] = []
        _summary["asset_class"] = []
        _summary["shares"] = []
        _summary["price"] = []
        _summary["value"] = []

        for pos in self.positions.values():
            _summary["ticker"].append(pos.asset.ticker)
            _summary["asset_class"].append(pos.asset.asset_class)
            _summary["price"].append(pos.asset.price)
            _summary["shares"].append(pos.shares)
            _summary["value"].append(pos.value)

        df = pd.DataFrame(_summary)
        df["val_date"] = self.val_date
        df.set_index(["val_date", "ticker"], inplace=True)

        total_value = df["value"].sum()

        # Add weight variable
        df["weight"] = df["value"] / total_value

        # round to two decimal place
        df = df.round(2)

        # finally sort by asset class and weight
        df.sort_values(["asset_class", "weight"], inplace=True, ascending=False)

        return df

    def plot(self, topic="Weights"):

        df = self.summary.reset_index().set_index("ticker")

        if topic == "Weights":
            fig = px.pie(df, values="weight", names=df.index, title="Weights")
            fig.show()

    def get_longest_time_index(self):
        """return the longest time index given the positions in the portfolio.

        Parameters
        ==========

        Returns
        =======
        longest time index given the positions in the portfolio
        """
        keys = []
        start_dates = []

        for _key, _pos in self.positions.items():
            try:
                keys.append(_key)
                start_dates.append(_pos.asset.start_date)
            except AttributeError:
                # for assets with no start date, pass
                pass

        # figure out the earliest start dates
        earliest_start_date = start_dates[0]

        for date in start_dates:
            if date < earliest_start_date:
                earliest_start_date = date

        # figure out the key by earliest start date
        earliest_start_date_key = keys[start_dates.index(earliest_start_date)]

        return self.positions[earliest_start_date_key].asset.history.index
