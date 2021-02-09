"""Analysis is a module that provides tools to analysis stocks and options.

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

import numpy as np
from pyarrow import feather
from fredapi import Fred
import plotly.graph_objects as go
from scipy.stats import norm


try:
    from asset import Stock, Cash, Option
    from portfolio import Portfolio
except (ImportError, ModuleNotFoundError):
    from oval.asset import Stock, Cash, Option
    from oval.portfolio import Portfolio


class OptionAnalyzer:
    def __init__(self, ticker, history_file):

        self._underlying = Stock(ticker, history_period="5d", financials=False)
        self._u_price = self.underlying.price
        self._history = self.combine_data(history_file, self.underlying.get_option_data())

    @property
    def underlying(self):
        return self._underlying
    
    @property
    def u_price(self):
        return self._u_price

    @property
    def history(self):
        return self._history

    @staticmethod
    def combine_data(history_file, new_data):
        """
        Parameters
        ==========
        history : feather file

        new_data : dataframe
        """
        history = feather.read_feather(history_file)

        updated = history.append(new_data.reset_index())
        updated.reset_index(drop=True, inplace=True)

        # calculate Bdays for history file, some history files
        # dont have this field.

        updated["busdays_to_mat"] = np.busday_count(updated["val_date"].dt.date, updated["mat_date"].dt.date)

        # set index
        updated.set_index(["val_date", "mat_date", "strike"], inplace=True)

        return updated


class YieldCurve:
    def __init__(self, start="2021-01-01"):
        self._interest_rates, self._cc_interest_rates = self.get_daily_interest_rates(start)
        self._cur_rates = [self.interest_rates[i].iloc[-1] for i in self.interest_rates]
        self._cc_cur_rates = [self.cc_interest_rates[i].iloc[-1] for i in self.cc_interest_rates]
        self._time_periods = [1, 3, 6, 12, 24]
    
    @property
    def cur_rates(self):
        return self._cur_rates

    @property
    def cc_cur_rates(self):
        return self._cc_cur_rates
    
    @property
    def cc_interest_rates(self):
        return self._cc_interest_rates

    @property
    def interest_rates(self):
        return self._interest_rates

    @property
    def time_periods(self):
        return self._time_periods

    def interp(self, months, cc=True):
        """Return interpolated rates based on the current rates and time periods.
        """

        if cc:
            rates = np.interp(months, xp=self.time_periods, fp=self.cc_cur_rates)    
        else:
            rates = np.interp(months, xp=self.time_periods, fp=self.cur_rates)
        return rates

    def plot(self, interp=False, cc=True):

        if cc:
            rates = self.cc_cur_rates
        else:
            rates = self.cur_rates

        if interp:
            _months = np.arange(0, 36, 1)
            _rates = np.interp(_months, xp=self.time_periods, fp=rates)
            fig = go.Figure(
                data=go.Scatter(
                    x=_months,
                    y=_rates,
                    mode="lines+markers",
                    name="yield curve (interpolated)",
                )
            )
        else:
            fig = go.Figure(
                data=go.Scatter(
                    x=self.time_periods,
                    y=rates,
                    mode="lines+markers",
                    name="yield curve",
                )
            )

        fig.show()

    @staticmethod
    def get_daily_interest_rates(start="2021-01-01"):
        """get nominal interest rates
        """
        fred = Fred()
        interest_rates = {}
        interest_rates["1m"] = fred.get_series("DGS1MO", observation_start=start)
        interest_rates["3m"] = fred.get_series("DGS3MO", observation_start=start)
        interest_rates["6m"] = fred.get_series("DGS6MO", observation_start=start)
        interest_rates["1y"] = fred.get_series("DGS1", observation_start=start)
        interest_rates["2y"] = fred.get_series("DGS2", observation_start=start)

        # get daily interest rates in annualized, continuously compounding form
        cc_interest_rates = {}
        cc_interest_rates["1m"] = np.log((interest_rates["1m"]/12+1)**12)
        cc_interest_rates["3m"] = np.log((interest_rates["3m"]/4+1)**4)
        cc_interest_rates["6m"] = np.log((interest_rates["6m"]/2+1)**2)
        cc_interest_rates["1y"] = np.log((interest_rates["1y"]+1))
        cc_interest_rates["2y"] = np.log((interest_rates["2y"]/0.5+1)**0.5)

        return interest_rates, cc_interest_rates




