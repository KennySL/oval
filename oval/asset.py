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
import numpy as np


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
    def __init__(self, ticker, history_period="max", financials=False):
        self._asset_class = "Stock"
        self._ticker = ticker
        self._yfticker = yf.Ticker(ticker)
        self._history = self.get_history(history_period, financials)
        self._start_date = self.history.index[0]
        self._val_date = self.history.index[-1]
        self._price = self.history["Close"][-1]

    @property
    def price(self):
        return self._price

    @property
    def history(self):
        return self._history

    @property
    def ticker(self):
        return self._ticker

    @property
    def yfticker(self):
        return self._yfticker

    @property
    def asset_class(self):
        return self._asset_class

    @property
    def start_date(self):
        return self._start_date

    @property
    def val_date(self):
        return self._val_date

    def get_history(self, period="max", financials=False):
        """
        get price history and add relevant data to the result.
        """
        df = self.yfticker.history(period=period)

        # add new columns of data
        # <TODO> when period is specified to be 1 day, an extra
        # row is produced. Need to fix that so that no
        # new rows are produced.
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["return"] = df["Close"] / df["Close"].shift(1) - 1

        # add marker
        df = self._add_markers(df)

        # join financials data if True
        if financials:
            try:
                df = self.get_financials(df)

            # for stock type that has no financials, skip
            except AttributeError:
                pass

        return df

    def get_moving_average(
        self, cols: "list of str or str", average_type="simple", window=30, **kwargs
    ):
        """
        Add simple or exponential moving average.

        Parameters
        ==========
        cols : list of str or str
            add moving average to these columns.

        average_type : str
            either 'simple' or 'exp'.

        window : int
            number of smoothing periods.

        **kwargs : dict
            keyword arguments for rolling or ewm function in pandas.

        Returns
        =======
        df : dataframe
            The original dataframe with the added moving average columns.

        """

        df = self.history

        if average_type == "simple":
            return df.join(
                df[cols].rolling(window, **kwargs).mean(), rsuffix=f"_sma_{window}"
            )
        elif average_type == "exp":
            return df.join(
                df[cols].ewm(span=window, **kwargs).mean(), rsuffix=f"_ema_{window}"
            )

    def get_option_data(self):
        """Get option data in a single dataframe."""
        maturities = self.yfticker.options
        val_date = self.val_date

        df = pd.DataFrame()

        for mat in maturities:
            opt = self.yfticker.option_chain(mat)
            call = opt[0]
            put = opt[1]

            straddle_view = call.merge(put, on="strike", suffixes=("_c", "_p"))

            mat_date = pd.to_datetime(mat, format="%Y-%m-%d")
            straddle_view["val_date"] = val_date
            straddle_view["mat_date"] = mat_date
            straddle_view["days_to_mat"] = (mat_date - val_date).days
            straddle_view["busdays_to_mat"] = np.busday_count(straddle_view["val_date"].dt.date, straddle_view["mat_date"].dt.date)

            straddle_view.set_index(["val_date", "mat_date", "strike"], inplace=True)

            df = df.append(straddle_view)

        return df

    def _add_markers(self, history_df):
        """add marker columns to history df, the markers can then be used to
        look up key period rows for quarterly and annual data.

        The markers are empty string except for the key period rows.

        For quarter_markers, the key period rows are the end date of every quarter.
        for example, '2020-03-31' would have a marker '1Q2020'.

        For year_markers, the key period rows are the end date of every year.
        for example, '2019-12-31' would have a market '2019'.
        """

        df = history_df.copy()
        idx = df.index

        df["year"] = idx.year
        df["year_changed"] = df["year"].shift(-1) != df["year"]
        df["quarter"] = idx.quarter
        df["quarter_changed"] = df["quarter"].shift(-1) != df["quarter"]

        def _set_year_marker(year, year_changed):
            if year_changed:
                year_marker = str(year)
            else:
                year_marker = "N/A"
            return year_marker

        df["year_marker"] = np.vectorize(_set_year_marker)(
            df["year"], df["year_changed"]
        )

        def _set_quarter_marker(quarter, year, quarter_changed):

            if quarter_changed:
                quarter_marker = f"{quarter}Q{year}"
            else:
                quarter_marker = "N/A"

            return quarter_marker

        df["quarter_marker"] = np.vectorize(_set_quarter_marker)(
            df["quarter"], df["year"], df["quarter_changed"]
        )

        # change markers of the last row
        # df.loc[df.index[-1], "year_marker"] = "latest_year"
        # df.loc[df.index[-1], "quarter_marker"] = "latest_quarter"

        return df

    def get_financials(self, history_df):
        """get annual and quarter balance sheet, financials and cashflows and merge with history df."""

        # get annual data first
        # balance sheet
        # financials
        # cashflows
        df = history_df.copy()

        idx = df.index
        bs = self.yfticker.balance_sheet.T
        fin = self.yfticker.financials.T
        cf = self.yfticker.cashflow.T

        def _join(main_df, merge_df, rsuffix, freq):

            merge_df = merge_df.sort_index()

            merge_df = self._add_markers(merge_df)

            if freq == "y":

                merge_df.columns = [
                    f"{col}_{rsuffix}_{freq}" for col in merge_df.columns
                ]
                main_df = main_df.merge(
                    merge_df,
                    how="left",
                    left_on="year_marker",
                    right_on=f"year_marker_{rsuffix}_{freq}",
                )
            elif freq == "q":

                merge_df.columns = [
                    f"{col}_{rsuffix}_{freq}" for col in merge_df.columns
                ]
                main_df = main_df.merge(
                    merge_df,
                    how="left",
                    left_on="quarter_marker",
                    right_on=f"quarter_marker_{rsuffix}_{freq}",
                )

            return main_df

        df = _join(df, bs, "bs", "y")
        df = _join(df, fin, "fin", "y")
        df = _join(df, cf, "cf", "y")

        # get quarterly data
        bs_q = self.yfticker.quarterly_balance_sheet.T
        fin_q = self.yfticker.quarterly_financials.T
        cf_q = self.yfticker.quarterly_cashflow.T

        df = _join(df, bs_q, "bs", "q")
        df = _join(df, fin_q, "fin", "q")
        df = _join(df, cf_q, "cf", "q")

        df.index = idx

        return df


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


class Fund(Asset):
    def __init__(self):
        """securities that cannot be splitted to its components are grouped
        into this class.
        """
        self._price = 1.0
        self._ticker = "Fund"
        self._asset_class = "Fund"

    @property
    def price(self):
        return self._price

    @property
    def ticker(self):
        return self._ticker

    @property
    def asset_class(self):
        return self._asset_class


class Option(Asset):
    def __init__(self, ticker, mat_date, opt_type, strike):
        """
        ticker : str
            underlying asset

        mat_date : str
            follows yyyy-mm-dd

        opt_type : str
            either "call" or "put"

        strike : float
        """

        self._ticker = f"Option_{ticker}"
        self._asset_class = "Option"

        self._underlying = Stock(ticker, history_period="5d", financials=False)
        self._u_price = self.underlying.price

        self._whole_data = self.underlying.get_option_data()
        self._data = self._whole_data.loc[slice(None), mat_date, strike]

        if opt_type == "call":
            self._price = self._data["bid_c"][0]
        else:
            self._price = self._data["bid_p"][0]

    @property
    def u_price(self):
        return self._u_price

    @property
    def price(self):
        return self._price

    @property
    def whole_data(self):
        return self._whole_data

    @property
    def data(self):
        return self._data

    @property
    def ticker(self):
        return self._ticker

    @property
    def underlying(self):
        return self._underlying

    @property
    def asset_class(self):
        return self._asset_class
