import numpy as np 
import pandas as pd
import os 
from joblib import Parallel, delayed

from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm


class TradingObject(object):
    """
    A trading object
    """
    def __init__(self) -> None:
        self.df = None  # dataframe that contains 3 columns: asset, time, value
        self.name = "JustATradingObject"  # name of the trading object
        self.name_value = None  # name of the value that it contains
        self.context = None  # trading context
    
    def get_value(self):
        return self.df[self.name_value].values
    
    def get_df(self):
        return self.df.copy()
    
    def get_name_value(self):
        return self.name_value


class Indicator(TradingObject):
    def assign_indicator_df(self, df):
        self.df = df
        self.name_value = df.columns[2]

    def generate_indicator_gr_by_fixed_period(self, win):
        """
        tính growth theo window
        - win: time window
        """
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        if time_column == 'DATE_TRADING':
            unit = "D"
        elif time_column == 'AM_QUARTER_STR':
            unit = "Q"
        elif time_column == 'MONTH_STR':
            unit = "M"
        if "." in name_factor:
            name_indicator = f"{name_factor}.-GR_{win}{unit}"
        else:
            name_indicator = f"{name_factor}-GR_{win}{unit}"
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column])
        indicator_df[name_indicator] = indicator_df.groupby('TICKER')[name_factor].transform(lambda x: x / x.shift(win))
        self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator

    def generate_indicator_gr_period_to_date(self, date_count_ref, period_col="MONTH_STR"):
        """
        tính growth so với ngày đầu tháng/quý
        - period_col nhận giá trị "MONTH_STR" nếu so với ngày đầu tháng
        - period_col nhận giá trị "AM_QUARTER_STR"  nếu so với ngày đầu quý
        """
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        merge_df = pd.merge(indicator_df, date_count_ref, how="left", on=['DATE_TRADING'])
        merge_df["SESS_NO"] = merge_df.groupby(['TICKER', period_col])['DATE_TRADING'].rank(method="dense")
        merge_df1 = merge_df[indicator_df.columns.to_list() + [period_col]]
        merge_df2 = merge_df[['TICKER', period_col, name_factor]][merge_df["SESS_NO"] == 1]
        indicator_df = pd.merge(merge_df1, merge_df2, how="left", on=['TICKER', period_col])
        if period_col == "MONTH_STR":
            win = "MTD"
        elif period_col == "AM_QUARTER_STR":
            win = "AM_QTD"
        if "." in name_factor:
            name_indicator = f"{name_factor}.-GR_{win}"
        else:
            name_indicator = f"{name_factor}-GR_{win}"
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column])
        indicator_df[name_indicator] = indicator_df[name_factor + "_x"] / indicator_df[name_factor + "_y"]
        self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator

    def generate_indicator_minmax_by_fixed_period(self, func, win, unit):
        """ 
        tính min hoặc max theo window
        - func: nhận giá trị "MIN" hoặc "MAX"
        - win: time window
        - unit: time unit ('D', hoặc 'M', hoặc 'Q' tùy theo time_column frequency)
        """
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        if "." in name_factor:
            name_indicator = f"{name_factor}.-{func}_{win}{unit}"
        else:
            name_indicator = f"{name_factor}-{func}_{win}{unit}"
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column])
        if func == "MIN":
            indicator_df[name_indicator] = (
                indicator_df.groupby('TICKER')[name_factor]
                .rolling(win)
                .apply(lambda x: np.min(x))
                .reset_index(drop=True)
            )
        if func == "MAX":
            indicator_df[name_indicator] = (
                indicator_df.groupby('TICKER')[name_factor]
                .rolling(win)
                .apply(lambda x: np.max(x))
                .reset_index(drop=True)
            )
        self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator

    def generate_indicator_div(self, name, numerator, denominator):
        """
        div: phép chia
        - numerator nhận giá trị là 1 column hoặc 1 số (float, int)
        - denominator là 1 column
        """
        df = self.df.copy()
        time_column = df.columns[1]
        if name == None:
            name_indicator = f"DIV.{numerator}.{denominator}"
        else:
            name_indicator = name
        if isinstance(numerator, (float, int)):
            df[name_indicator] = np.where(df[denominator] == 0, np.nan, numerator / df[denominator])
        else:
            df[name_indicator] = np.where(df[denominator] == 0, np.nan, df[numerator] / df[denominator])
        self.df = df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator

    def generate_indicator_reci(self):  # div by 1
        """
        reci: phép nghịch đảo (div by 1)
        """
        df = self.df.copy()
        name_factor = self.name_value
        name_indicator = f"{name_factor}-RECI"
        df[name_indicator] = np.where(df[name_factor] == 0, np.nan, 1 / df[name_factor])
        self.df = df[['TICKER', 'DATE_TRADING', name_indicator]]
        self.name_value = name_indicator

    def generate_indicator_mul(self, name, mul1, mul2):
        """
        mul: phép nhân
        - input mul1 nhận giá trị là 1 thừa số, hoặc 1 column
        - input mul1 nhận giá trị là 1 column
        """
        df = self.df.copy()
        if name == None:
            name_indicator = f"MUL.{mul1}.{mul2}"
        else:
            name_indicator = name
        if isinstance(mul1, (float, int)):
            df[name_indicator] = mul1 * df[mul2]
        else:
            df[name_indicator] = df[mul1] * df[mul2]
        self.df = df[['TICKER', 'DATE_TRADING', name_indicator]]
        self.name_value = name_indicator

    def generate_factor_fs_a1(self, win):
        """ 
        a1 = quý hiện tại (name_factor) / trung bình 4 quý liền trước (C22)
        """
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        # indicator_df[name_factor] = np.where()
        if "." in name_factor:
            name_indicator = f"{name_factor}.-A1"
        else:
            name_indicator = f"{name_factor}-A1"

        # average 4 quý gần nhất
        indicator_df["C21"] = (
            indicator_df.groupby('TICKER')[name_factor]
            .rolling(win)
            .apply(lambda x: np.average(x))
            .reset_index(drop=True)
        )

        # average 4 quý liền trước (không tính quý hiện tại)
        indicator_df["C22"] = indicator_df.groupby('TICKER')["C21"].transform(lambda x: x.shift(1))
        indicator_df[name_indicator] = np.where(
            indicator_df["C22"] == 0, np.nan, indicator_df[name_factor] / indicator_df["C22"]
        )

        # indicator = name_factor/C22
        indicator_df[name_indicator] = indicator_df[name_indicator].astype("float64")
        self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator

    def generate_factor_fs_a11(self, win):
        """ 
        a11 = quý hiện tại - 1 (C1) / trung bình 4 quý liền trước của (quý hiện tại - 1) (C22)
        """
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        if "." in name_factor:
            name_indicator = f"{name_factor}.-A11"
        else:
            name_indicator = f"{name_factor}-A11"

        # average 4 quý gần nhất
        indicator_df["C21"] = (
            indicator_df.groupby('TICKER')[name_factor]
            .rolling(win)
            .apply(lambda x: np.average(x))
            .reset_index(drop=True)
        )

        # average 4 quý liền trước của 4 (quý hiện tại - 1)
        indicator_df["C22"] = indicator_df.groupby('TICKER')["C21"].transform(lambda x: x.shift(2))

        # quý hiện tại - 1
        indicator_df["C1"] = indicator_df.groupby('TICKER')[name_factor].transform(lambda x: x.shift(1))
        indicator_df[name_indicator] = np.where(
            indicator_df["C22"] == 0, np.nan, indicator_df["C1"] / indicator_df["C22"]
        )
        self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator

    def generate_indicator_sum_rolling_fixed_period(self, win, unit="D", keep_factor_column=False):
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        if "." in name_factor:
            name_indicator = f"{name_factor}.-SUM_{win}{unit}"
        else:
            name_indicator = f"{name_factor}-SUM_{win}{unit}"
        indicator_df[name_indicator] = (
            indicator_df.groupby('TICKER')[name_factor].rolling(win).apply(lambda x: np.sum(x)).reset_index(drop=True)
        )
        if keep_factor_column:
            self.df = indicator_df
        else:
            self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator

    def generate_indicator_roc_rolling_fixed_period(self, win, func1=None, func2="MAX", unit="D"):  # rate of change
        """
        roc: rate of change = current or min / max
        toc: time of change = T_current or T_min - T_max
        voc: velocity of change = ROC ^ (1/TOC)
        """
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        # name indicator
        if func1 != None:
            if "." in name_factor:
                name_indicator = f"{name_factor}.-ROC_{func1}_{func2}_{win}{unit}"
            else:
                name_indicator = f"{name_factor}-ROC_{func1}_{func2}_{win}{unit}"
        else:
            if "." in name_factor:
                name_indicator = f"{name_factor}.-ROC_{func2}_{win}{unit}"
            else:
                name_indicator = f"{name_factor}-ROC_{func2}_{win}{unit}"
        # func1
        if func1 == "MIN":
            indicator_df["A"] = (
                indicator_df.groupby('TICKER')[name_factor]
                .rolling(win)
                .apply(lambda x: np.min(x))
                .reset_index(drop=True)
            )
        elif func1 == "MAX":
            indicator_df["A"] = (
                indicator_df.groupby('TICKER')[name_factor]
                .rolling(win)
                .apply(lambda x: np.max(x))
                .reset_index(drop=True)
            )
        elif func1 == None:
            indicator_df["A"] = indicator_df[name_factor]
        # func2
        if func2 == "MIN":
            indicator_df["B"] = (
                indicator_df.groupby('TICKER')[name_factor]
                .rolling(win)
                .apply(lambda x: np.min(x))
                .reset_index(drop=True)
            )
        elif func2 == "MAX":
            indicator_df["B"] = (
                indicator_df.groupby('TICKER')[name_factor]
                .rolling(win)
                .apply(lambda x: np.max(x))
                .reset_index(drop=True)
            )
        indicator_df[name_indicator] = indicator_df["A"] / indicator_df["B"]
        self.df = indicator_df
        self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator

    def generate_indicator_dif_by_fixed_period(self, win):
        """
        dif: difference = phép trừ 2 theo window = x - x.shift(win)
        """
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        if time_column == 'DATE_TRADING':
            unit = "D"
        elif time_column == 'AM_QUARTER_STR':
            unit = "Q"
        elif time_column == 'MONTH_STR':
            unit = "M"
        if "." in name_factor:
            name_indicator = f"{name_factor}.-DIF_{win}{unit}"
        else:
            name_indicator = f"{name_factor}-DIF_{win}{unit}"
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column])
        indicator_df[name_indicator] = indicator_df.groupby('TICKER')[name_factor].transform(lambda x: x - x.shift(win))
        self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator

    def generate_indicator_dif_function_rolling_fixed_period(self, win, func1="MAX", func2="MIN", unit="D"):
        """
        dif: difference = phép trừ 2 functions
        mỗi function có thể nhận 1 trong các giá trị: MIN, MAX, None
        """
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        # name_indicator
        if func1 != None:
            if "." in name_factor:
                name_indicator = f"{name_factor}.-DIF_{func1}_{func2}_{win}{unit}"
            else:
                name_indicator = f"{name_factor}-DIF_{func1}_{func2}_{win}{unit}"
        else:
            if "." in name_factor:
                name_indicator = f"{name_factor}.-DIF_{func2}_{win}{unit}"
            else:
                name_indicator = f"{name_factor}-DIF_{func2}_{win}{unit}"
        # func1
        if func1 == "MIN":
            indicator_df["A"] = (
                indicator_df.groupby('TICKER')[name_factor]
                .rolling(win)
                .apply(lambda x: np.min(x))
                .reset_index(drop=True)
            )
        elif func1 == "MAX":
            indicator_df["A"] = (
                indicator_df.groupby('TICKER')[name_factor]
                .rolling(win)
                .apply(lambda x: np.max(x))
                .reset_index(drop=True)
            )
        elif func1 == None:
            indicator_df["A"] = indicator_df[name_factor]
        # func2
        if func2 == "MIN":
            indicator_df["B"] = (
                indicator_df.groupby('TICKER')[name_factor]
                .rolling(win)
                .apply(lambda x: np.min(x))
                .reset_index(drop=True)
            )
        elif func2 == "MAX":
            indicator_df["B"] = (
                indicator_df.groupby('TICKER')[name_factor]
                .rolling(win)
                .apply(lambda x: np.max(x))
                .reset_index(drop=True)
            )
        indicator_df[name_indicator] = indicator_df["A"] - indicator_df["B"]
        self.df = indicator_df
        self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator

    def generate_indicator_toc_rolling_fixed_period(self, win, func1=None, func2="MAX", unit="D"):  # time of change
        """
        roc: rate of change = current or min / max
        toc: time of change = T_current or T_min - T_max
        voc: velocity of change = ROC ^ (1/TOC)
        """
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        # name_indicator
        if func1 != None:
            if "." in name_factor:
                name_indicator = f"{name_factor}.-TOC_{func1}_{func2}_{win}{unit}"
            else:
                name_indicator = f"{name_factor}-TOC_{func1}_{func2}_{win}{unit}"
        else:
            if "." in name_factor:
                name_indicator = f"{name_factor}.-TOC_{func2}_{win}{unit}"
            else:
                name_indicator = f"{name_factor}-TOC_{func2}_{win}{unit}"
        # func1
        if func1 == "MIN":
            indicator_df["A_ID"] = (
                indicator_df.groupby('TICKER')[name_factor]
                .rolling(win)
                .apply(lambda x: pd.Series(x).idxmin())
                .reset_index(drop=True)
            )
        elif func1 == "MAX":
            indicator_df["A_ID"] = (
                indicator_df.groupby('TICKER')[name_factor]
                .rolling(win)
                .apply(lambda x: pd.Series(x).idxmax())
                .reset_index(drop=True)
            )
        elif func1 == None:
            indicator_df["A_ID"] = indicator_df.index
        # func2
        if func2 == "MIN":
            indicator_df["B_ID"] = (
                indicator_df.groupby('TICKER')[name_factor]
                .rolling(win)
                .apply(lambda x: pd.Series(x).idxmin())
                .reset_index(drop=True)
            )
        elif func2 == "MAX":
            indicator_df["B_ID"] = (
                indicator_df.groupby('TICKER')[name_factor]
                .rolling(win)
                .apply(lambda x: pd.Series(x).idxmax())
                .reset_index(drop=True)
            )
        indicator_df[name_indicator] = indicator_df["A_ID"] - indicator_df["B_ID"] + 1
        self.df = indicator_df
        self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator

    def generate_indicator_voc_rolling_fixed_period(self, win, func1=None, func2="MAX", unit="D"):  # velocity of change
        """
        roc: rate of change = current or min / max
        toc: time of change = T_current or T_min - T_max
        voc: velocity of change = ROC ^ (1/TOC)
        """
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        # name_indicator
        if func1 != None:
            if "." in name_factor:
                name_indicator = f"{name_factor}.-VOC_{func1}_{func2}_{win}{unit}"
            else:
                name_indicator = f"{name_factor}-VOC_{func1}_{func2}_{win}{unit}"
        else:
            if "." in name_factor:
                name_indicator = f"{name_factor}.-VOC_{func2}_{win}{unit}"
            else:
                name_indicator = f"{name_factor}-VOC_{func2}_{win}{unit}"
        # func1
        if func1 == "MIN":
            indicator_df["A"] = (
                indicator_df.groupby('TICKER')[name_factor]
                .rolling(win)
                .apply(lambda x: np.min(x))
                .reset_index(drop=True)
            )
            indicator_df["A_ID"] = (
                indicator_df.groupby('TICKER')[name_factor]
                .rolling(win)
                .apply(lambda x: pd.Series(x).idxmin())
                .reset_index(drop=True)
            )
        elif func1 == "MAX":
            indicator_df["A"] = (
                indicator_df.groupby('TICKER')[name_factor]
                .rolling(win)
                .apply(lambda x: np.max(x))
                .reset_index(drop=True)
            )
            indicator_df["A_ID"] = (
                indicator_df.groupby('TICKER')[name_factor]
                .rolling(win)
                .apply(lambda x: pd.Series(x).idxmax())
                .reset_index(drop=True)
            )
        elif func1 == None:
            indicator_df["A"] = indicator_df[name_factor]
            indicator_df["A_ID"] = indicator_df.index
        indicator_df["SESSION_CNT"] = indicator_df["A_ID"] - indicator_df["B_ID"] + 1
        indicator_df["X"] = 1 / indicator_df["SESSION_CNT"]
        indicator_df[name_indicator] = pow(indicator_df["A"] / indicator_df["B"], 1 / indicator_df["SESSION_CNT"])
        self.df = indicator_df
        self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator

    def generate_indicator_pow(self, name_indicator, power):
        """
        pow: power
        - if power = 2 -> SQUARE
        - if power = 0.5 -> SQRT
        - else: POW.FACTOR.power
        """
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        if name_indicator == None:
            if power == 2:
                name_indicator = f"{name_factor}-SQUARE"
            elif power == 0.5:
                name_indicator = f"{name_factor}-SQRT"
            else:
                name_indicator = f"POW.{name_factor}.{power}"
        else:
            name_indicator = name_indicator
        if isinstance(power, (float, int)):
            indicator_df[name_indicator] = pow(indicator_df[name_factor], power)
        else:
            indicator_df[name_indicator] = pow(indicator_df[name_factor], indicator_df[power])
        self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator

    
    def generate_indicator_rank_horizontal_within_list(self, daily_ticker_df):

        ranking_indicator_df = self.get_df()
        ranking_indicator = self.get_name_value()

        # Merge with ticker following a list
        ranking_indicator_position_df = daily_ticker_df.merge(
            ranking_indicator_df, how="left", on=['TICKER', 'DATE_TRADING']
        )
        ranking_indicator_position_df = ranking_indicator_position_df[['TICKER', 'DATE_TRADING', ranking_indicator]]

        # Create indicator based on rank horizontal, ranks are represented by number, not by quantile (this might not be necessary anw)
        self.set_df(ranking_indicator_position_df)
        self.generate_indicator_rank_horizontal(method="dense", ascending=False, pct=False)


    def generate_indicator_top_rank(self, top):
        """
        tính top x%, output của indicator là 1 nếu trong top, 0 nếu ngoài top
        - top: đơn vị % (ví dụ 10%)
        """
        df = self.df.copy()
        factor_name = df.columns[2]
        time_column = df.columns[1]
        df[factor_name] = df.groupby([time_column])[factor_name].rank(ascending=False)
        df["TICKER_ON_EXCHANGE"] = df.groupby(['DATE_TRADING'])['TICKER'].transform("count")
        df["TICKER_TOP"] = np.round(df["TICKER_ON_EXCHANGE"] * top / 100, -1)
        if "." in factor_name:
            name_indicator = f"{factor_name}.-TOP_{top}PCT"
        else:
            name_indicator = f"{factor_name}-TOP_{top}PCT"
        df[name_indicator] = np.where(df[factor_name] <= df["TICKER_TOP"], 1, 0)
        self.df = df[['TICKER', time_column, name_indicator]].sort_values(['TICKER', time_column])
        self.name_value = name_indicator

    def generate_indicator_ma_upto(self, keep_factor_column=False):
        """moving average upto"""
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        ticker_list = np.sort(np.unique(indicator_df['TICKER'].unique()))
        if "." in name_factor:
            name_indicator = f"{name_factor}.-MA_UPTO"
        else:
            name_indicator = f"{name_factor}-MA_UPTO"
        parallel_output = Parallel(n_jobs=-1)(
            delayed(get_ma_for_ticker)(ticker, indicator_df, name_factor) for ticker in ticker_list
        )
        indicator_df = pd.concat(parallel_output)
        if keep_factor_column:
            self.df = indicator_df
        else:
            self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator

    def generate_indicator_ma_rolling_fixed_period(self, win, unit="D", keep_factor_column=False):
        """moving average theo window"""
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        if "." in name_factor:
            name_indicator = f"{name_factor}.-MA_{win}{unit}"
        else:
            name_indicator = f"{name_factor}-MA_{win}{unit}"
        indicator_df[name_indicator] = (
            indicator_df.groupby('TICKER')[name_factor]
            .rolling(win)
            .apply(lambda x: np.average(x))
            .reset_index(drop=True)
        )
        if keep_factor_column:
            self.df = indicator_df
        else:
            self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator

    def generate_indicator_ema_rolling_fixed_period(self, win, unit="D", keep_factor_column=False):
        """exponential moving average"""
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        if "." in name_factor:
            name_indicator = f"{name_factor}.-EMA_{win}{unit}"
        else:
            name_indicator = f"{name_factor}-EMA_{win}{unit}"
        indicator_df[name_indicator] = (
            indicator_df.groupby('TICKER')[name_factor]
            .apply(lambda x: x.ewm(span=win, adjust=False, min_periods=win).mean())
            .reset_index(drop=True)
        )
        
        if keep_factor_column:
            self.df = indicator_df
        else:
            self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator

    def generate_indicator_rsi_rolling_fixed_period(self, rolling_win, rsi_win, unit="D", keep_factor_column=False):

        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        if "." in name_factor:
            name_indicator = f"{name_factor}.-ROLLING_{rolling_win}{unit}-RSI_{rsi_win}{unit}"
        else:
            name_indicator = f"{name_factor}-ROLLING_{rolling_win}{unit}-RSI_{rsi_win}{unit}"
        
        
        indicator_df[name_factor + "B_1" + unit] = indicator_df.groupby('TICKER')[name_factor].transform(
            lambda x: x.shift(1)
        )

        delta = indicator_df.groupby('TICKER')[name_factor].diff() / indicator_df[name_factor + "B_1" + unit]
        indicator_df["GAIN"] = delta.clip(lower=0)
        indicator_df["LOSS"] = -1 * delta.clip(upper=0)
        
        # MA Gain rolling 
        gain_array = list(indicator_df["GAIN"])
        ticker_array = list(indicator_df['TICKER'])
        ma_gain_array = gain_array.copy()

        loss_array = list(indicator_df["LOSS"])
        ticker_array = list(indicator_df['TICKER'])
        ma_loss_array = loss_array.copy()

        n = 0
        while n < len(gain_array): 
            if n < rolling_win:
                ma_gain_array[n] = np.NaN
                ma_loss_array[n] = np.NaN
            elif (ticker_array[n] != ticker_array[n-rolling_win]) & (ticker_array[n] != ticker_array[n-rolling_win-1]):
                ma_gain_array[n] = np.NaN
                ma_loss_array[n] = np.NaN
            else: 
                ma_gain_array[n] = pd.Series(gain_array[(n-rolling_win)+1:n+1]).ewm(com=rsi_win - 1, adjust=False).mean().to_list()[-1]
                ma_loss_array[n] = pd.Series(loss_array[(n-rolling_win)+1:n+1]).ewm(com=rsi_win - 1, adjust=False).mean().to_list()[-1]
            n = n+1

        indicator_df["MA_GAIN" + str(rsi_win)] = ma_gain_array
        indicator_df["MA_LOSS" + str(rsi_win)] = ma_loss_array
        indicator_df["RS" + str(rsi_win)] = indicator_df["MA_GAIN" + str(rsi_win)] / indicator_df["MA_LOSS" + str(rsi_win)]

        name_indicator =  f'RSI-ROLLING_{rolling_win}D_{rsi_win}D'
        indicator_df[name_indicator] = 100 - (100 / (1 + indicator_df["RS" + str(rsi_win)]))
        self.df = indicator_df[['TICKER', time_column, name_indicator]]    
        
        if keep_factor_column:
            self.df = indicator_df
        else:
            self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator
    

    def shift_indicator_forward(self, win, unit="D"):
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        if "." in name_factor:
            name_indicator = f"{name_factor}.-B_{win}{unit}"
        else:
            name_indicator = f"{name_factor}-B_{win}{unit}"
        indicator_df = self.df.copy()
        indicator_df[name_indicator] = indicator_df.groupby('TICKER')[name_factor].transform(lambda x: x.shift(win))
        self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator

    def shift_indicator_backward(self, win, unit="D"):
        """WARNING: This causes look forward bias"""
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        if "." in name_factor:
            name_indicator = f"{name_factor}.-A_{win}{unit}"
        else:
            name_indicator = f"{name_factor}-A_{win}{unit}"
        indicator_df[name_indicator] = indicator_df.groupby('TICKER')[name_factor].transform(lambda x: x.shift(-win))
        self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator

    def generate_indicator_drawdown(self):
        """drawdown tính từ price_change_df"""
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        ticker_list = np.sort(np.unique(indicator_df['TICKER'].unique()))
        # replace with old function because the new function return invalid values
        parallel_output = Parallel(n_jobs=-1)(
            delayed(calculate_drawdown_ticker)(indicator_df, ticker) for ticker in ticker_list
        )
        dd = np.concatenate((parallel_output), axis=0)
        name_indicator = f"DRAWDOWN"
        indicator_df[name_indicator] = dd
        # output_df = output_df[['TICKER', 'DATE_TRADING', name_indicator]]
        self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator

    def generate_indicator_std_rolling_fixed_period(self, win, unit):
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        if "." in name_factor:
            name_indicator = f"{name_factor}.-SD_{win}{unit}"
        else:
            name_indicator = f"{name_factor}-SD_{win}{unit}"

        ticker_list = np.sort(np.unique(indicator_df['TICKER'].unique()))
        data = indicator_df.pivot(index='DATE_TRADING', columns='TICKER', values=name_factor)
        rolling_sd = data.rolling(window=win).std()
        rolling_sd = rolling_sd.reset_index()      
        rolling_sd = pd.melt(rolling_sd, id_vars='DATE_TRADING', value_vars=ticker_list, var_name='TICKER', value_name=name_indicator)
        
        self.df = rolling_sd[['TICKER', time_column, name_indicator]].sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        self.name = name_indicator
        self.name_value = name_indicator

    def generate_indicator_std_upto(self):
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        
        if "." in name_factor:
            name_indicator = f"{name_factor}.-SD_UPTO"
        else:
            name_indicator = f"{name_factor}-SD_UPTO"
        indicator_df[name_indicator] = (
            indicator_df.groupby('TICKER')[name_factor].apply(lambda x: np.std(x)).reset_index(drop=True)
        )
        self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name = name_indicator
        self.name_value = name_indicator

    def generate_indicator_icv_rolling_fixed_period(self, win, unit):
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        if "." in name_factor:
            name_indicator = f"{name_factor}.-ICV_{win}{unit}"
        else:
            name_indicator = f"{name_factor}-ICV_{win}{unit}"
        indicator_df["SD"] = (
            indicator_df.groupby('TICKER')[name_factor].rolling(win).apply(lambda x: np.std(x)).reset_index(drop=True)
        )
        indicator_df["AVG"] = (
            indicator_df.groupby('TICKER')[name_factor]
            .rolling(win)
            .apply(lambda x: np.average(x))
            .reset_index(drop=True)
        )
        indicator_df[name_indicator] = np.abs(indicator_df["AVG"] / indicator_df["SD"])
        self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name = name_indicator
        self.name_value = name_indicator


    def generate_indicator_rsq_rolling_fixed_period(self, win):
        """ R-Squared"""
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        name_indicator = f"{name_factor}_RSQ"
        indicator_df[name_indicator] = (
            indicator_df.groupby('TICKER')[name_factor].rolling(win).apply(get_coeff_with_linear).reset_index(drop=True)
        )
        self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name = name_indicator
        self.name_value = name_indicator

    def generate_indicator_zscore_vertical_rolling_fixed_period(self, win, unit):
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        if "." in name_factor:
            name_indicator = f"{name_factor}.-ZV_{win}{unit}"
        else:
            name_indicator = f"{name_factor}-ZV_{win}{unit}"
        indicator_df[name_indicator] = (
            indicator_df.groupby('TICKER')[name_factor].apply(lambda x: zscore_rolling(x, win)).reset_index(drop=True)
        )
        self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator

    def generate_indicator_zscore_horizontal(self):
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        if "." in name_factor:
            name_indicator = f"{name_factor}.-ZH"
        else:
            name_indicator = f"{name_factor}-ZH"
        indicator_df[name_indicator] = (
            indicator_df.groupby(time_column)[name_factor].apply(lambda x: zscore(x)).reset_index(drop=True)
        )
        self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator

    def generate_indicator_zscore_vertical_upto(self):
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        ticker_list = np.sort(np.unique(indicator_df['TICKER'].unique()))
        parallel_output = Parallel(n_jobs=-1)(
            delayed(get_zv_for_ticker)(ticker, indicator_df, name_factor) for ticker in ticker_list
        )
        indicator_df = pd.concat(parallel_output)
        if "." in name_factor:
            name_indicator = f"{name_factor}.-ZV_UPTO"
        else:
            name_indicator = f"{name_factor}-ZV_UPTO"
        # output_df = output_df[['TICKER', 'DATE_TRADING', name_indicator]]
        self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator

    def generate_indicator_rsi(self, win, unit):
        indicator_df = self.df.copy()      
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        if "." in name_factor:
            name_indicator = f"{name_factor}.-RSI_{win}{unit}"
        else:
            name_indicator = f"{name_factor}-RSI_{win}{unit}"

        indicator_df[name_factor + "B_1" + unit] = indicator_df.groupby('TICKER')[name_factor].transform(
            lambda x: x.shift(1)
        )
        delta = indicator_df.groupby('TICKER')[name_factor].diff() / indicator_df[name_factor + "B_1" + unit]
        indicator_df["GAIN"] = delta.clip(lower=0)
        indicator_df["LOSS"] = -1 * delta.clip(upper=0)

        indicator_df["MA_GAIN" + str(win)] = indicator_df.groupby('TICKER')['GAIN'].apply(lambda x: x.ewm(com=win - 1, adjust=False).mean())
        indicator_df["MA_LOSS" + str(win)] = indicator_df.groupby('TICKER')['LOSS'].apply(lambda x: x.ewm(com=win - 1, adjust=False).mean())

        indicator_df["RS" + str(win)] = indicator_df["MA_GAIN" + str(win)] / indicator_df["MA_LOSS" + str(win)]
        indicator_df[name_indicator] = 100 - (100 / (1 + indicator_df["RS" + str(win)]))
        self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator

    def generate_indicator_stochastic(self, win, unit):
        """ stochastic = (current - lowest) / (highest - lowest)"""
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        if "." in name_factor:
            name_indicator = f"{name_factor}.-STOCH_{win}{unit}"
        else:
            name_indicator = f"{name_factor}-STOCH_{win}{unit}"

        indicator_df["HIGHEST"] = (
            indicator_df.groupby('TICKER')[name_factor].rolling(win).apply(lambda x: np.max(x)).reset_index(drop=True)
        )
        indicator_df["LOWEST"] = (
            indicator_df.groupby('TICKER')[name_factor].rolling(win).apply(lambda x: np.min(x)).reset_index(drop=True)
        )
        indicator_df[name_indicator] = (indicator_df[name_factor] - indicator_df["LOWEST"]) / (
            indicator_df["HIGHEST"] - indicator_df["LOWEST"]
        )
        self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator

    def generate_indicator_bin(self, nbins):
        """
        chia indicator thành nbins và xem mỗi dòng thuộc bin nào, dùng qcut
        """
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        if "." in name_factor:
            name_indicator = f"{name_factor}.-BIN_{nbins}"
        else:
            name_indicator = f"{name_factor}-BIN_{nbins}"
        indicator_df[name_indicator] = indicator_df.groupby(time_column)[name_factor].transform(
            lambda x: pd.qcut(x, nbins, labels=False, duplicates="drop")
        )
        self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator

    def generate_indicator_rank_horizontal(self, method="dense", ascending=True, pct=True):
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        if "." in name_factor:
            name_indicator = f"{name_factor}.-RH"
        else:
            name_indicator = f"{name_factor}-RH"
        indicator_df[name_indicator] = indicator_df.groupby(time_column)[name_factor].rank(
            method=method, ascending=ascending, pct=pct
        )
        self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator

    def generate_indicator_rank_vertical_rolling_win(self, win, unit, method="dense", ascending=True, pct=True):
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        if "." in name_factor:
            name_indicator = f"{name_factor}.-RV_{win}{unit}"
        else:
            name_indicator = f"{name_factor}-RV_{win}{unit}"
        indicator_df[name_indicator] = (
            indicator_df.groupby('TICKER')[name_factor]
            .rolling(win)
            .apply(lambda x: pd.Series(x).rank(method=method, ascending=ascending, pct=pct).iloc[-1])
            .reset_index(drop=True)
        )
        self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator

    def generate_indicator_rank_vertical_upto(self):
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        ticker_list = np.sort(np.unique(indicator_df['TICKER'].unique()))
        parallel_output = Parallel(n_jobs=-1)(
            delayed(get_rank_vertical_for_ticker)(ticker, indicator_df, name_factor) for ticker in ticker_list
        )
        indicator_df = pd.concat(parallel_output)
        if "." in name_factor:
            name_indicator = f"{name_factor}.-RV_UPTO"
        else:
            name_indicator = f"{name_factor}-RV_UPTO"
        # output_df = output_df[['TICKER', 'DATE_TRADING', name_indicator]]
        self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator

    def get_rank_vertical_for_factor_rolling_win(self, win, unit="D"):
        factor_df = self.p_factor.get_df()
        name_factor = self.name_value
        time_column = factor_df.columns[1]
        factor_df = factor_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        ticker_list = np.sort(np.unique(factor_df['TICKER'].unique()))
        parallel_output = Parallel(n_jobs=-1)(
            delayed(get_rank_vertical_for_ticker_rolling_win)(ticker, indicator_df, name_factor, win)
            for ticker in ticker_list
        )
        indicator_df = pd.concat(parallel_output)
        if "." in name_factor:
            name_indicator = f"{name_factor}.-RV_{win}{unit}"
        else:
            name_indicator = f"{name_factor}-RV_{win}{unit}"
        self.df = indicator_df[['TICKER', time_column, name_indicator]]
        self.name_value = name_indicator
        # return output_df

    def generate_indicator_regression_upto(self, regression_indicator):
        """regression_indicator là 1 trong các indicator sau:
        - regression_indicator_list = [INTERCEPT, SLOPE, MSE, PREDICTED, RESIDUAL, RSE, RESIDUAL_STANDARDIZED]
        - regression_indicator = indicator_name + '-' + i for i in regression_indicator_list
        vd. regression_indicator = PCA_LOG-SLOPE
        """
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        ticker_list = np.sort(np.unique(indicator_df['TICKER'].unique()))
        parallel_output = Parallel(n_jobs=-1)(
            delayed(get_regression_upto_for_ticker)(indicator_df, ticker, name_factor) for ticker in ticker_list
        )
        indicator_df = pd.concat(parallel_output)
        name_indicator = f"{regression_indicator}_UPTO"
        # output_df = output_df[['TICKER', 'DATE_TRADING', name_indicator]]
        self.df = indicator_df[['TICKER', time_column, regression_indicator]].rename(
            {regression_indicator: name_indicator}, axis=1
        )
        self.name_value = name_indicator

    def generate_indicator_regression_rolling_win(self, regression_indicator, win, unit="D"):
        """regression_indicator là 1 trong các indicator sau:
        - regression_indicator_list = [INTERCEPT, SLOPE, MSE, PREDICTED, RESIDUAL, RSE, RESIDUAL_STANDARDIZED]
        - regression_indicator = indicator_name + '-' + i for i in regression_indicator_list
        vd. regression_indicator = PCA_LOG-SLOPE
        """
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        check_len = indicator_df.groupby('TICKER', as_index=False).apply(lambda x: len(x))
        check_len.columns = ['TICKER', "LEN"]
        check_len = check_len[check_len["LEN"] <= win]
        ticker_excluded = check_len['TICKER'].unique()
        ticker_list = [i for i in indicator_df['TICKER'].unique() if i not in ticker_excluded]
        # np.sort(np.unique(indicator_df['TICKER'].unique()))
        parallel_output = Parallel(n_jobs=-1)(
            delayed(get_regression_rolling_win_for_ticker)(indicator_df, ticker, name_factor, win)
            for ticker in ticker_list
        )
        indicator_df = pd.concat(parallel_output)
        name_indicator = f"{regression_indicator}_{win}{unit}"
        # output_df = output_df[['TICKER', 'DATE_TRADING', name_indicator]]
        self.df = indicator_df[['TICKER', time_column, regression_indicator]].rename(
            {regression_indicator: name_indicator}, axis=1
        )
        self.name_value = name_indicator

    def generate_indicator_streak(self):
        """
        streak = số ngày liên tiếp mà giá trị T > giá trị T-1
        """
        indicator_df = self.df.copy()
        name_factor = self.name_value
        time_column = indicator_df.columns[1]
        indicator_df = indicator_df.sort_values(by=['TICKER', time_column]).reset_index(drop=True)
        a = []
        for ticker in indicator_df['TICKER'].unique():
            df_ticker = indicator_df[indicator_df['TICKER'] == ticker]
            series = pd.DataFrame(df_ticker[name_factor])
            geq = series >= series.shift(1)  # True if rising
            eq = series == series.shift(1)  # True if equal
            logic_table = pd.concat([geq, eq], axis=1)

            streaks = [0]  # holds the streak duration, starts with 0

            for row in logic_table.iloc[1:].itertuples():  # iterate through logic table
                if row[2]:  # same value as before
                    streaks.append(0)
                    continue
                last_value = streaks[-1]
                if row[1]:  # higher value than before
                    streaks.append(last_value + 1 if last_value >= 0 else 1)  # increase or reset to +1
                else:  # lower value than before
                    streaks.append(last_value - 1 if last_value < 0 else -1)  # decrease or reset to -1
            a.append(streaks)
            b = [item for sublist in a for item in sublist] 
        name_indicator = f'{name_factor}-STREAK'
        indicator_df[name_indicator] = b    
        self.df = indicator_df[['TICKER', 'DATE_TRADING', name_indicator]]
        self.name_value = name_indicator

# temp 

def get_coeff_with_linear(input_series):
    win = len(input_series)
    acc_array = np.cumprod(input_series.values)
    output = np.corrcoef(acc_array, np.arange(1, win + 1))[0][1]
    return output

def get_regression_rolling_win_for_ticker(df, ticker, factor_name, win):
    df1 = df[df['TICKER'] == ticker].copy()
    df1 = df1.sort_values(by=['DATE_TRADING'])
    df1["SESS_NO"] = df1.groupby('TICKER')['DATE_TRADING'].rank().astype(int) - 1
    rolling_params = RollingOLS(
        endog=df1[factor_name], exog=sm.add_constant(df1["SESS_NO"]), expanding=False, window=win
    ).fit()
    params_output = rolling_params.params
    mse = np.sqrt(rolling_params.mse_resid)
    mse.rename("MSE", inplace=True)
    params_output = pd.merge(params_output, mse, how="left", left_index=True, right_index=True)
    params_output.rename(
        columns={"const": factor_name + "-INTERCEPT", "SESS_NO": factor_name + "-SLOPE", "MSE": factor_name + "-MSE"},
        inplace=True,
    )
    output = pd.merge(df1, params_output, how="left", left_index=True, right_index=True)
    # predicted value
    output["INTERCEPT-A_1D"] = output.groupby('TICKER')[factor_name + "-INTERCEPT"].shift(1)
    output["SLOPE-A_1D"] = output.groupby('TICKER')[factor_name + "-SLOPE"].shift(1)
    output[factor_name + "-PREDICTED"] = output["INTERCEPT-A_1D"] + output["SESS_NO"] * output["SLOPE-A_1D"]
    a = output.groupby('TICKER')[factor_name].shift(1)
    b = output.groupby('TICKER')[factor_name + "-PREDICTED"].shift(1)
    # residual
    output[factor_name + "-RESIDUAL"] = a - b
    output[factor_name + "-RESIDUAL2"] = output.groupby('TICKER')[factor_name + "-RESIDUAL"].apply(lambda x: x ** 2)
    output[factor_name + "-RSE"] = output.groupby('TICKER')[factor_name + "-RESIDUAL2"].transform(
        lambda x: x.rolling(win).sum() / (win - 2)
    )
    output[factor_name + "-RSE"] = np.sqrt(output[factor_name + "-RSE"])
    output[factor_name + "-RESIDUAL_STANDARDIZED"] = output[factor_name + "-RESIDUAL"] / output[factor_name + "-RSE"]
    output = output[
        [
            'TICKER',
            'DATE_TRADING',
            factor_name + "-INTERCEPT",
            factor_name + "-SLOPE",
            factor_name + "-MSE",
            factor_name + "-PREDICTED",
            factor_name + "-RESIDUAL",
            factor_name + "-RSE",
            factor_name + "-RESIDUAL_STANDARDIZED",
        ]
    ]
    # output['DIV.RESID.RSE'] = output['RESID'] / output['RSE']
    return output

def get_regression_upto_for_ticker(df, ticker, factor_name):
    df1 = df[df['TICKER'] == ticker].copy()
    df1 = df1.sort_values(by=['DATE_TRADING'])
    df1["SESS_NO"] = df1.groupby('TICKER')['DATE_TRADING'].rank().astype(int) - 1
    rolling_params = RollingOLS(endog=df1[factor_name], exog=sm.add_constant(df1["SESS_NO"]), expanding=True).fit()
    params_output = rolling_params.params
    mse = np.sqrt(rolling_params.mse_resid)
    mse.rename("MSE", inplace=True)
    params_output = pd.merge(params_output, mse, how="left", left_index=True, right_index=True)
    params_output.rename(
        columns={"const": factor_name + "-INTERCEPT", "SESS_NO": factor_name + "-SLOPE", "MSE": factor_name + "-MSE"},
        inplace=True,
    )
    output = pd.merge(df1, params_output, how="left", left_index=True, right_index=True)
    # predicted value
    output["INTERCEPT-A_1D"] = output.groupby('TICKER')[factor_name + "-INTERCEPT"].shift(1)
    output["SLOPE-A_1D"] = output.groupby('TICKER')[factor_name + "-SLOPE"].shift(1)
    output[factor_name + "-PREDICTED"] = output["INTERCEPT-A_1D"] + output["SESS_NO"] * output["SLOPE-A_1D"]
    a = output.groupby('TICKER')[factor_name].shift(1)
    b = output.groupby('TICKER')[factor_name + "-PREDICTED"].shift(1)
    # residual
    output[factor_name + "-RESIDUAL"] = a - b
    output[factor_name + "-RESIDUAL2"] = output.groupby('TICKER')[factor_name + "-RESIDUAL"].apply(lambda x: x ** 2)
    output[factor_name + "-RSE"] = output.groupby('TICKER')[factor_name + "-RESIDUAL2"].transform(
        lambda x: x.expanding().sum() / (x.expanding().count() - 2)
    )
    output[factor_name + "-RSE"] = np.sqrt(output[factor_name + "-RSE"])
    output[factor_name + "-RESIDUAL_STANDARDIZED"] = output[factor_name + "-RESIDUAL"] / output[factor_name + "-RSE"]
    output = output[
        [
            'TICKER',
            'DATE_TRADING',
            factor_name + "-INTERCEPT",
            factor_name + "-SLOPE",
            factor_name + "-MSE",
            factor_name + "-PREDICTED",
            factor_name + "-RESIDUAL",
            factor_name + "-RSE",
            factor_name + "-RESIDUAL_STANDARDIZED",
        ]
    ]
    # output['DIV.RESID.RSE'] = output['RESID'] / output['RSE']
    return output

def get_ma_for_ticker(ticker, factor_df, name_factor):
    ticker_df = factor_df[factor_df['TICKER'] == ticker]
    time_column = ticker_df.columns[1]
    if "." in name_factor:
        name_indicator = f"{name_factor}.-MA_UPTO"
    else:
        name_indicator = f"{name_factor}-MA_UPTO"
    ma_series = pd.Series(index=ticker_df[time_column])
    if np.isnan(factor_df[name_factor].iloc[0]) == False:
        ma_series[0] = ticker_df[name_factor].iloc[0]
    for i in range(1, len(ticker_df.index)):
        ma_series[i] = ticker_df[name_factor].iloc[0 : i + 1].mean()
    ticker_df[name_indicator] = ma_series.values
    ticker_df = ticker_df[['TICKER', time_column, name_indicator]]
    return ticker_df

def get_rank_vertical_for_ticker(ticker, factor_df, name_factor):
    ticker_df = factor_df[factor_df['TICKER'] == ticker]
    time_column = ticker_df.columns[1]
    if "." in name_factor:
        name_indicator = f"{name_factor}.-RV_UPTO"
    else:
        name_indicator = f"{name_factor}-RV_UPTO"
    rank_series = pd.Series(index=ticker_df[time_column])
    if np.isnan(factor_df[name_factor].iloc[0]) == False:
        rank_series[0] = 1
    for i in range(1, len(ticker_df.index)):
        rank_series[i] = (
            ticker_df[name_factor].iloc[0 : i + 1].rank(method="dense", ascending=True, pct=True).values[-1]
        )
    ticker_df[name_indicator] = rank_series.values
    ticker_df = ticker_df[['TICKER', time_column, name_indicator]]
    return ticker_df


def get_rank_vertical_for_ticker_rolling_win(ticker, factor_df, name_factor, win):
    ticker_df = factor_df[factor_df['TICKER'] == ticker].copy()
    time_column = ticker_df.columns[1]
    if "." in name_factor:
        name_indicator = f"{name_factor}.-RV_{win}"
    else:
        name_indicator = f"{name_factor}-RV_{win}"
    ticker_df[name_indicator] = (
        ticker_df[name_factor].rolling(win).apply(lambda x: pd.Series(x).rank(method="dense", pct=True).iloc[-1])
    )
    ticker_df = ticker_df[['TICKER', time_column, name_indicator]]
    return ticker_df


def get_zv_for_ticker(ticker, factor_df, name_factor):
    ticker_df = factor_df[factor_df['TICKER'] == ticker]
    time_column = ticker_df.columns[1]
    if "." in name_factor:
        name_indicator = f"{name_factor}.-ZV_UPTO"
    else:
        name_indicator = f"{name_factor}-ZV_UPTO"
    zv_series = pd.Series(index=ticker_df[time_column])
    if np.isnan(factor_df[name_factor].iloc[0]) == False:
        zv_series[0] = 0
    for i in range(1, len(ticker_df.index)):
        zv_series[i] = zscore(ticker_df[name_factor].iloc[0 : i + 1]).values[-1]
    ticker_df[name_indicator] = zv_series.values
    ticker_df = ticker_df[['TICKER', time_column, name_indicator]]
    return ticker_df

def zscore(x):
    m = x.mean()
    s = x.std()
    z = (x - m) / s
    return z


def zscore_rolling(x, win):
    r = x.rolling(window=win)
    m = r.mean()
    s = r.std()
    z = (x - m) / s
    return z

def calculate_drawdown_ticker(price_change_df, ticker): 
    return_df_ticker = price_change_df[price_change_df['TICKER'] == ticker]
    portfolio_return_array = np.array(return_df_ticker[return_df_ticker.columns[2]])
    portfolio_cumm_change_percent = np.cumprod(portfolio_return_array)
    mmd_max_diff_acc = (portfolio_cumm_change_percent / np.maximum.accumulate(portfolio_cumm_change_percent) - 1) 
    return -mmd_max_diff_acc