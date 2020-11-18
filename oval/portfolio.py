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

import pandas as pd


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

    @shares.setter
    def update_shares(self, new_shares):
        self._shares = new_shares
        self._value = self._calculate_value()

    @property
    def value(self):
        return self._value

    def _calculate_value(self):
        return self.shares * self.asset.price


class Portfolio:
    def __init__(self, positions):
        self._positions = {pos.asset.ticker: pos for pos in positions}
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

    def update(self, new_positions):
        """update positions in the portfolio. In addition, reflect
        summary and value accordingly.

        Parameters
        ==========
        new_positions : list of Position objects

        Returns
        =======

        """
        _new_positions = {pos.asset.ticker: pos for pos in new_positions}
        self._positions.update(_new_positions)
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
        df.set_index("ticker", inplace=True)
        df.sort_values("asset_class", inplace=True)

        return df
