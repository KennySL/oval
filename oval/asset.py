"""asset is a module that defines various asset classes.

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

asset Listing
=================
"""

import pandas as pd
import yfinance as yf
from abc import ABC, abstractmethod
import abc


class Asset(ABC):
    """Abstract class that defines the blue print for
    every type of asset.

    """

    @abstractmethod
    def __init__(self, units):
        ...

    @abc.abstractproperty
    def price(self):
        ...

    @abc.abstractproperty
    def ticker(self):
        ...

    @abc.abstractproperty
    def asset_class(self):
        ...


class Stock(Asset):
    def __init__(self, ticker):
        self._asset_class = "Stock"
        self._ticker = ticker
        self._yfticker = yf.Ticker(ticker)
        self._price_history = self.yfticker.history(period="max")
        self._price = self.price_history["Close"][-1]

    @property
    def price(self):
        return self._price

    @property
    def price_history(self):
        return self._price_history

    @property
    def ticker(self):
        return self._ticker

    @property
    def yfticker(self):
        return self._yfticker

    @property
    def asset_class(self):
        return self._asset_class


class Cash(Asset):
    def __init__(self):
        self._price = 1.0
        self._ticker = "Cash"
        self._asset_class = "Cash"

    @property
    def price(self):
        return self._price

    @property
    def ticker(self):
        return self._ticker

    @property
    def asset_class(self):
        return self._asset_class
