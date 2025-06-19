import json
import traceback

from sklearn.utils import resample

from taosanalytics.service import AbstractForecastService
from prophet import Prophet
import pandas as pd
import re
# 算法实现类名称 需要以下划线 "_" 开始，并以 Service 结束
class _MyForecastService(AbstractForecastService):
    """ 定义类，从 AbstractForecastService 继承并实现其定义的抽象方法 execute  """
    name = 'myprophet'

    # 该算法的描述信息 (建议添加)
    desc = """return the prophet time series data"""

    params = None

    # Prophet 支持的参数，避免传入不支持的参数
    allowed_params = [
        "growth",
        "changepoint_prior_scale",
        "changepoint_range",
        "yearly_seasonality",
        "weekly_seasonality",
        "daily_seasonality",
        "holidays",
        "seasonality_mode",
        "seasonality_prior_scale",
        "holidays_prior_scale",
        "interval_width",
        "mcmc_samples",
        "uncertainty_samples",
        "stan_backend"
    ]

    # 节假日支持的参数, 避免传入不支持的参数
    # {
    #     'holiday': ['spring_festival', 'spring_festival'],
    #     'ds': pd.to_datetime(['2025-01-29', '2026-02-17']),
    #     'lower_window': [-1, -1],  # 提前1天开始生效
    #     'upper_window': [2, 2],  # 节假日延续2天
    # }
    holidays_allowed_params = [
        "holidays",
    ]

    # 采样参数, 避免传入不支持的参数
    # 数据采样
    # '1S' 每 1 秒
    # '1T' 每 1 分钟
    # '1H' 每 1 小时
    # '1D' 每 1 天
    # '1W' 每 1 周
    # '1M' 每 1 月
    # '1Q' 每 1 季度
    # '1A' 每 1 年
    # 数据采样方法
    # .mean() 平均值（默认）
    # .sum() 求和
    # .min() 最小值
    # .max() 最大值
    # .count() 数量（样本个数）
    resample_allowed_params = [
        "resample",
        "resample_mode"
    ]

    # 饱和参数, 避免传入不支持的参数
    # 饱和最大值
    # 0 未指定, 如果saturating_cap不为None那么使用手动指定的值
    # 1: 自动选择最大值, saturating_cap = saturating_cap * saturating_cap_scale
    # 2: 自动选择最大值 * 1.2
    # 饱和最小值
    # 0 未指定, 如果saturating_floor不为None那么使用手动指定的值
    # 1: 自动选择最大值, saturating_floor = saturating_floor * saturating_floor_scale
    # 2: 自动选择最大值 * 1.2
    saturating_allowed_params = [
        "saturating_cap_max",
        "saturating_cap",
        "saturating_cap_scale",
        "saturating_floor_min",
        "saturating_floor"
        "saturating_floor_scale",
    ]

    def __init__(self):

        """类初始化方法"""
        super().__init__()

    def execute(self):
        try:
            data = self.__dict__

            # 数据预处理、数据读取
            df = pd.DataFrame({
                'ds': pd.to_datetime(data['ts_list'], unit='ms'),
                'y': data['list']
            })

            print(data)

            # 数据预处理、数据排序
            df = df.sort_values(by='ds').reset_index(drop=True)

            # 数据预处理、数据采样
            resample_params = self.parseResampleParams()
            df_resampled = self.doResample(df, resample_params)

            # 开始训练、进行预测
            prophet_params = self.parseProphetParams()
            m = Prophet(**prophet_params)

            # 预测点开始时间
            start_time = df_resampled['ds'].max()
            if self.start_ts is not None:
                # 如果 start_ts 不为 None，则使用 start_ts
                start_time = pd.to_datetime(self.start_ts, unit='ms')

            # 预测点数量，预测点间隔, 单位为秒
            freq = None
            if self.time_step is not None:
                # 如果 time_step 不为 None，则使用 time_step
                freq = f'{self.time_step}S'

            validated_freq = freq
            if freq is None and resample_params is not None:
                validated_freq = self.validate_freq(resample_params)

            # 预测点数量，不需要包含历史数据
            future_pd = pd.date_range(start=start_time, periods=self.rows, freq=validated_freq)
            future = pd.DataFrame({'ds': future_pd})
            print(f"预测时间范围：{future_pd.min()} ~ {future_pd.max()}")
            # future = m.make_future_dataframe(start_time=start_time,
            #                                 periods=self.rows,
            #                                 freq=validated_freq,
            #                                 include_history=False)

            # 饱和参数设置
            self.configSaturatingParams(m, df_resampled, future)

            # 参数设置 - 开启节假日参数设置

            """ 算法逻辑的核心实现"""
            # 训练并进行预测
            m.fit(df)
            forecast = m.predict(future)

            timestamp_ms = forecast['ds'].astype('int64') // 10 ** 6

            res = []
            # 设置预测结果时间戳列
            res.append(timestamp_ms.tolist())
            # 设置预测结果
            res.append(forecast['yhat'].tolist())

            """检查用户输入，是否要求返回预测置信区间上下界"""
            if self.return_conf:
                res.append(forecast['yhat_lower'].tolist())
                res.append(forecast['yhat_upper'].tolist())

            """返回结果"""
            return {"res": res, "mse": 0}

        except Exception as e:
            traceback.print_exc()  # 打印详细堆栈

    def doResample(self, df, resample_params):
        if resample_params is None or "resample" not in resample_params:
            return df

        default_resample_method = "mean"
        if "resample_method" in resample_params:
            default_resample_method = resample_params["resample_method"]
        return df.set_index('ds').resample(resample_params["resample"]).agg(default_resample_method).reset_index()

    def parseProphetParams(self):
        prophet_params = {k: self.params[k] for k in self.params if k in self.allowed_params}

        if "holidays" in self.params:
            json_str = self.params["holidays"]
            holiday_data = json.loads(json_str)
            holidays = pd.DataFrame(holiday_data)
            holidays['ds'] = pd.to_datetime(holidays['ds'])

            prophet_params["holidays"] = holidays

        return prophet_params

    def parseResampleParams(self):
        resample_params = {k: self.params[k] for k in self.params if k in self.resample_allowed_params}
        return resample_params

    def configSaturatingParams(self, m, df_resampled, future):
        growth = m.__getattribute__("growth")
        if growth != "logistic":
            return

        saturating_params = {k: self.params[k] for k in self.params if k in self.saturating_allowed_params}
        cap = self.get_cap(df_resampled, saturating_params)
        floor = self.get_floor(saturating_params)
        if cap is not None:
            df_resampled['cap'] = cap
            future['cap'] = cap

        if floor is not None:
            df_resampled['floor'] = floor
            future['floor'] = floor

        return

    # 设置算法参数
    def set_params(self, params):
        """该算法无需任何输入参数，直接调用父类函数，不处理算法参数设置逻辑"""
        # 获取默认参数设置
        super().set_params(params)

        # 获取扩展参数
        self.params = params

        # 获取自定义参数
        filtered = {k: v for k, v in params.items() if k != "ts_list" and k != "list"}
        print(filtered)

        return

    # 获取饱和上限
    def get_cap(self, df_resampled, saturating_params):
        """获取饱和下限"""
        cap = None
        if saturating_params.saturating_cap_max:
            cap = df_resampled['y'].max()
        if saturating_params.saturating_cap is not None:
            cap = saturating_params.saturating_cap
        if cap is not None and saturating_params.saturating_cap_scale is not None:
            cap = cap * saturating_params.saturating_cap_scale

        return cap
        
    # 获取饱和下限
    def get_floor(self, df_resampled, saturating_params):
        """获取饱和下限"""
        floor = None
        if saturating_params.saturating_floor_min:
            floor = df_resampled['y'].min()
        if saturating_params.saturating_floor is not None:
            floor = saturating_params.saturating_floor
        if floor is not None and saturating_params.saturating_floor_scale is not None:
            floor = floor * saturating_params.saturating_floor_scale

        return floor
        
    def validate_freq(self, resample_params):
        if "resample" not in resample_params:
            return

        resample = resample_params["resample"]
        pattern = r'^\d*[STHDWMQA]$'  # 支持可选数字+单位，单位为秒、分钟、小时等
        if re.match(pattern, resample):
            return resample
        else:
            raise ValueError(f"Unsupported resample frequency: {resample}")