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
import pandas as pd


try:
    from asset import Stock, Cash, Option
    from portfolio import Portfolio
except (ImportError, ModuleNotFoundError):
    from oval.asset import Stock, Cash, Option
    from oval.portfolio import Portfolio


class YieldCurve:
    def __init__(self, start="2021-01-01"):
        self._interest_rates, self._cc_interest_rates = self.get_daily_interest_rates(
            start
        )
        self._cur_rates = [self.interest_rates[i].iloc[-1] for i in self.interest_rates]
        self._cc_cur_rates = [
            self.cc_interest_rates[i].iloc[-1] for i in self.cc_interest_rates
        ]
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
        """Return interpolated rates based on the current rates and time periods."""

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
        """get nominal interest rates"""
        fred = Fred()
        interest_rates = {}
        interest_rates["1m"] = fred.get_series("DGS1MO", observation_start=start)
        interest_rates["3m"] = fred.get_series("DGS3MO", observation_start=start)
        interest_rates["6m"] = fred.get_series("DGS6MO", observation_start=start)
        interest_rates["1y"] = fred.get_series("DGS1", observation_start=start)
        interest_rates["2y"] = fred.get_series("DGS2", observation_start=start)

        # get daily interest rates in annualized, continuously compounding form
        cc_interest_rates = {}
        cc_interest_rates["1m"] = np.log((interest_rates["1m"] / 12 + 1) ** 12)
        cc_interest_rates["3m"] = np.log((interest_rates["3m"] / 4 + 1) ** 4)
        cc_interest_rates["6m"] = np.log((interest_rates["6m"] / 2 + 1) ** 2)
        cc_interest_rates["1y"] = np.log((interest_rates["1y"] + 1))
        cc_interest_rates["2y"] = np.log((interest_rates["2y"] / 0.5 + 1) ** 0.5)

        return interest_rates, cc_interest_rates


class OptionAnalyzer:
    def __init__(
        self,
        ticker,
        history_file,
        get_updated_option_data=True,
        yield_curve=YieldCurve(),
        model="BS",
        sample_iv_size=5,
        sample_points=1000,
    ):

        self._underlying = Stock(ticker, history_period="max", financials=False)
        self._u_price = self.underlying.price

        if get_updated_option_data:
            new_data = self.underlying.get_option_data()
            self._history = self.combine_data(history_file, new_data)
        else:
            self._history = self.combine_data(history_file)

        self._yield_curve = yield_curve

        self._analysis = self.calculate_iv_and_greeks(
            model, sample_iv_size, sample_points
        )

    @property
    def underlying(self):
        return self._underlying

    @property
    def analysis(self):
        return self._analysis

    @property
    def u_price(self):
        return self._u_price

    @property
    def history(self):
        return self._history

    @property
    def yield_curve(self):
        return self._yield_curve

    def write(self, fpath, replace=False):
        if fpath.is_file():
            if replace:
                self.history.reset_index().to_feather(fpath)
            else:
                raise FileExistsError("File exists.")

        else:
            self.history.reset_index().to_feather(fpath)

    @staticmethod
    def combine_data(history_file, new_data=None):
        """
        Parameters
        ==========
        history : feather file

        new_data : dataframe
            if None, the function simply read the history file,
            set busdays_to_mat, and set indexes.
        """
        history = feather.read_feather(history_file)

        if new_data is None:
            updated = history
        else:
            updated = history.append(new_data.reset_index())
            updated.reset_index(drop=True, inplace=True)
        # calculate Bdays for history file, some history files
        # dont have this field.

        updated["busdays_to_mat"] = np.busday_count(
            updated["val_date"].dt.date, updated["mat_date"].dt.date
        )

        # set index
        updated.set_index(["val_date", "mat_date", "strike"], inplace=True)

        return updated

    @staticmethod
    def _convert_to_array(series, n, m):
        return np.array(series).reshape(n, m)

    def calculate_iv_and_greeks(self, model="BS", sample_iv_size=5, sample_points=1000):
        """calculate implied volatility and greeks.

        Parameters
        ==========
        model : str

        sample_iv_size : int

        sample_points : int

        <TODO> dividend yield is set to 0. but it should
        take the expected annualized dividend yield.

        Returns
        =======
        results : Dataframe

        Note:
        - theta is sensitivity of option prices on daily basis.
        - rho is sensitivity of option prices on 1% increase in interest rate.
        - vega is sensitivity of option prices on 1% increase in volatility.

        """

        # 1. calculate hv and convert to annual volatility
        hv = np.std(self.underlying.history["log_return"]) * np.sqrt(252)

        # 2. turn series to array for linear algebra computations.
        # S, K, r, q, T_t,
        T_t = self._convert_to_array(self.history["busdays_to_mat"] / 252, -1, 1)
        r = self.yield_curve.interp(T_t)
        q = np.zeros(T_t.shape)
        K = self._convert_to_array(self.history.reset_index()["strike"], -1, 1)
        S = self.history.reset_index([1, 2]).merge(
            self.underlying.history["Close"], left_index=True, right_index=True
        )
        S = self._convert_to_array(S["Close"], -1, 1)

        # calculate call price and its greeks
        loop = True
        loop_sample_iv_size = sample_iv_size
        while loop:
            try:
                call_results = self._calculate_iv_and_greeks_c_bs(
                    loop_sample_iv_size, sample_points, hv, S, q, T_t, K, r
                )
            except ValueError:
                loop_sample_iv_size *= 2
                print(f"try call sample iv size {loop_sample_iv_size}")
            else:
                loop = False

        # calculate put price and its greeks
        loop = True
        loop_sample_iv_size = sample_iv_size
        while loop:
            try:
                put_results = self._calculate_iv_and_greeks_p_bs(
                    loop_sample_iv_size, sample_points, hv, S, q, T_t, K, r
                )
            except ValueError:
                loop_sample_iv_size *= 2
                print(f"try put sample iv size {loop_sample_iv_size}")
            else:
                loop = False

        # output results in dataframe
        results = {}
        results["T_t"] = T_t.squeeze()
        results["r"] = r.squeeze()
        results["q"] = q.squeeze()
        results["K"] = K.squeeze()
        results["S"] = S.squeeze()

        results["d1_c"] = call_results[0].squeeze()
        results["d2_c"] = call_results[1].squeeze()
        results["iv_c"] = call_results[2].squeeze()
        results["c"] = call_results[3].squeeze()
        results["delta_c"] = call_results[4].squeeze()
        results["gamma_c"] = call_results[5].squeeze()
        results["vega_c"] = call_results[6].squeeze()
        results["theta_c"] = call_results[7].squeeze()
        results["rho_c"] = call_results[8].squeeze()
        results["idx_c"] = call_results[9].squeeze()
        results["zero_vol_c"] = call_results[10].squeeze()
        results["max_vol_c"] = call_results[11].squeeze()

        results["d1_p"] = put_results[0].squeeze()
        results["d2_p"] = put_results[1].squeeze()
        results["iv_p"] = put_results[2].squeeze()
        results["p"] = put_results[3].squeeze()
        results["delta_p"] = put_results[4].squeeze()
        results["gamma_p"] = put_results[5].squeeze()
        results["vega_p"] = put_results[6].squeeze()
        results["theta_p"] = put_results[7].squeeze()
        results["rho_p"] = put_results[8].squeeze()
        results["idx_p"] = put_results[9].squeeze()
        results["zero_vol_p"] = put_results[10].squeeze()
        results["max_vol_p"] = put_results[11].squeeze()

        self._analysis = pd.concat(
            [self.history.reset_index(), pd.DataFrame(results)], axis=1
        )
        self._analysis.set_index(["val_date", "mat_date", "strike"], inplace=True)

        self.calculate_leverage()

        return self._analysis

    def _calculate_iv_and_greeks_c_bs(
        self, sample_iv_size, sample_points, hv, S, q, T_t, K, r
    ):
        """calculate iv and greeks for call based on BS model."""

        sample_iv = np.linspace(0.000001, sample_iv_size * hv, sample_points)

        # calculate d1, d2 based on sample iv values
        numerator = (0.5 * sample_iv ** 2).reshape(1, sample_points)
        numerator = r - q + numerator
        numerator = numerator * T_t
        numerator = np.log(S / K) + numerator

        denominator = sample_iv * np.sqrt(T_t)

        d1 = numerator / denominator
        d2 = d1 - denominator

        # sample call prices
        c = S * np.exp(-q * T_t) * norm.cdf(d1) - K * np.exp(-r * T_t) * norm.cdf(d2)

        # pick the iv where the difference between observed price and theorectical price is the smallest
        # for illiquid options, such as deep ITM/OTM, use last trade price.
        lastPrice_c = self._convert_to_array(self.history["lastPrice_c"], -1, 1)
        bid_c = self._convert_to_array(self.history["bid_c"], -1, 1)

        # observed call price is set as bid price if bid price is not 0,
        # if bid price is 0, use last traded price.
        observed_c = np.where(bid_c == 0, lastPrice_c, bid_c)
        diff_c = np.abs(observed_c - c)
        idx_c = diff_c.argmin(axis=1).reshape(-1, 1)

        # now the indexes are determined,
        # if indexes exceeds the max volatility value, return some distinct values
        zero_vol_c = c[:, 0].reshape(-1, 1)
        max_vol_c = c[:, -1].reshape(-1, 1)
        lower_than_zero_vol = np.where((observed_c - zero_vol_c) <= 0, 0, 1)
        higher_than_max_vol = np.where((observed_c - max_vol_c) > 0, 1, 0)

        if higher_than_max_vol.sum() > 0:
            raise ValueError(
                "observed price exceeds max sample iv implied price. will try bigger sample iv size."
            )

        idx_c = np.where(lower_than_zero_vol == 0, 0, idx_c)
        iv_c = sample_iv[idx_c]
        # get the theorectical call price
        c = np.take_along_axis(c, idx_c, 1)

        # calculate greeks
        # calculate d1, d2 using implied volatilities calculated in previous step.
        numerator = 0.5 * iv_c ** 2
        numerator = r - q + numerator
        numerator = numerator * T_t
        numerator = np.log(S / K) + numerator

        denominator = iv_c * np.sqrt(T_t)

        d1 = numerator / denominator
        d2 = d1 - denominator

        delta_c = np.exp(-q * T_t) * norm.cdf(d1)
        gamma_c = np.exp(-q * T_t) * self._inverseNorm(d1) / (S * denominator)
        vega_c = S * np.exp(-q * T_t) * self._inverseNorm(d1) * np.sqrt(T_t) / 100
        theta_c = (
            (-S * iv_c * np.exp(-q * T_t) * self._inverseNorm(d1) / (2 * np.sqrt(T_t)))
            - r * K * np.exp(-r * T_t) * norm.cdf(d2)
            + q * S * np.exp(-q * T_t) * norm.cdf(d1)
        ) / 252

        rho_c = K * T_t * np.exp(-r * T_t) * norm.cdf(d2) / 100

        return (
            d1,
            d2,
            iv_c,
            c,
            delta_c,
            gamma_c,
            vega_c,
            theta_c,
            rho_c,
            idx_c,
            zero_vol_c,
            max_vol_c,
        )

    def _calculate_iv_and_greeks_p_bs(
        self, sample_iv_size, sample_points, hv, S, q, T_t, K, r
    ):
        """calculate iv and greeks for call based on BS model."""

        sample_iv = np.linspace(0.000001, sample_iv_size * hv, sample_points)

        # calculate d1, d2 based on sample iv values
        numerator = (0.5 * sample_iv ** 2).reshape(1, sample_points)
        numerator = r - q + numerator
        numerator = numerator * T_t
        numerator = np.log(S / K) + numerator

        denominator = sample_iv * np.sqrt(T_t)

        d1 = numerator / denominator
        d2 = d1 - denominator

        # sample put prices
        p = K * np.exp(-r * T_t) * norm.cdf(-d2) - S * np.exp(-q * T_t) * norm.cdf(-d1)

        # pick the iv where the difference between observed price and theorectical price is the smallest
        # for illiquid options, such as deep ITM/OTM, use last trade price.
        lastPrice_p = self._convert_to_array(self.history["lastPrice_p"], -1, 1)
        bid_p = self._convert_to_array(self.history["bid_p"], -1, 1)

        # observed put price is set as bid price if bid price is not 0,
        # if bid price is 0, use last traded price.
        observed_p = np.where(bid_p == 0, lastPrice_p, bid_p)
        diff_p = np.abs(observed_p - p)
        idx_p = diff_p.argmin(axis=1).reshape(-1, 1)

        # now the indexes are determined,
        # if indexes exceeds the max volatility value, return some distinct values
        zero_vol_p = p[:, 0].reshape(-1, 1)
        max_vol_p = p[:, -1].reshape(-1, 1)
        lower_than_zero_vol = np.where((observed_p - zero_vol_p) <= 0, 0, 1)
        higher_than_max_vol = np.where((observed_p - max_vol_p) > 0, 1, 0)

        if higher_than_max_vol.sum() > 0:
            raise ValueError(
                "observed price exceeds max sample iv implied price. will try bigger sample iv size."
            )

        idx_p = np.where(lower_than_zero_vol == 0, 0, idx_p)
        iv_p = sample_iv[idx_p]
        # get the theorectical call price
        p = np.take_along_axis(p, idx_p, 1)

        # calculate greeks
        # calculate d1, d2 using implied volatilities calculated in previous step.
        numerator = 0.5 * iv_p ** 2
        numerator = r - q + numerator
        numerator = numerator * T_t
        numerator = np.log(S / K) + numerator

        denominator = iv_p * np.sqrt(T_t)

        d1 = numerator / denominator
        d2 = d1 - denominator

        delta_p = np.exp(-q * T_t) * (norm.cdf(d1) - 1)
        gamma_p = np.exp(-q * T_t) * self._inverseNorm(d1) / (S * denominator)
        vega_p = S * np.exp(-q * T_t) * self._inverseNorm(d1) * np.sqrt(T_t) / 100
        theta_p = (
            (-S * iv_p * np.exp(-q * T_t) * self._inverseNorm(d1) / (2 * np.sqrt(T_t)))
            + r * K * np.exp(-r * T_t) * norm.cdf(d2)
            - q * S * np.exp(-q * T_t) * norm.cdf(d1)
        ) / 252

        rho_p = -K * T_t * np.exp(-r * T_t) * norm.cdf(-d2) / 100

        return (
            d1,
            d2,
            iv_p,
            p,
            delta_p,
            gamma_p,
            vega_p,
            theta_p,
            rho_p,
            idx_p,
            zero_vol_p,
            max_vol_p,
        )

    @staticmethod
    def _inverseNorm(y):
        """ normal pdf """
        return 1 / np.sqrt(2 * np.pi) * np.exp(-(y ** 2) / 2)

    def calculate_leverage(self):
        """calculate leverage and other relevant calculations using
        the greeks and iv from analysis attribute.
        """
        self._analysis["leverage_c"] = (
            self._analysis["S"]
            * (self._analysis["delta_c"] + 0.5 * self._analysis["gamma_c"])
            / self._analysis["c"]
        )
        self._analysis["breakeven_c"] = self._analysis["S"] + self._analysis["c"]
        self._analysis["instrinc_c"] = self._analysis["S"] - self._analysis["K"]

        self._analysis["leverage_p"] = (
            self._analysis["S"]
            * (self._analysis["delta_p"] + 0.5 * self._analysis["gamma_p"])
            / self._analysis["p"]
        )
        self._analysis["breakeven_p"] = self._analysis["S"] - self._analysis["p"]
        self._analysis["instrinc_p"] = self._analysis["K"] - self._analysis["S"]
