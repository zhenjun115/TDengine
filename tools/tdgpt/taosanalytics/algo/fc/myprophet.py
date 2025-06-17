from taosanalytics.service import AbstractForecastService
from prophet import Prophet
import pandas as pd
# 算法实现类名称 需要以下划线 "_" 开始，并以 Service 结束
class _MyForecastService(AbstractForecastService):
    """ 定义类，从 AbstractForecastService 继承并实现其定义的抽象方法 execute  """
    # 定义算法调用关键词，全小写 ASCII 码
    name = 'myprophet'

    # 该算法的描述信息 (建议添加)
    desc = """return the prophet time series data"""

    # 饱和最大值
    # 0 未指定,如果saturating_cap不为None那么使用手动指定的值
    # 1: 自动选择最大值, saturating_cap = saturating_cap * saturating_cap_scale
    # 2: 自动选择最大值 * 1.2
    saturating_cap_mode = 0
    saturating_cap_scale = 1.2
    saturating_cap = None

    # 饱和最小值
    # 0 未指定,如果saturating_cap不为None那么使用手动指定的值
    # 1: 自动选择最大值, saturating_cap = saturating_cap * saturating_cap_scale
    # 2: 自动选择最大值 * 1.2
    saturating_floor_mode = 0
    saturating_floor = None
    saturating_floor_scale = 1.2

    # growth 模式 liner 或者 logistic
    growth_mode = "linear"

    def __init__(self):

        """类初始化方法"""
        super().__init__()

        """ 参数设置 """


    def execute(self):
        # df = pd.DataFrame({
        #     'ds': pd.date_range(start='2020-01-01', periods=80, freq='D'),
        #     'y': [min(100, i + np.random.randn() * 10) for i in range(80)]
        # })
        data = self.__dict__
        # 打印data参数
        # print(data)
        df = pd.DataFrame({
            'ds': pd.to_datetime(data['ts_list'], unit='ms'),
            'y': data['list']
        })
        df = df.sort_values(by='ds').reset_index(drop=True)
        # 设置饱和上限
        df['cap'] = 342
        # 初始化并拟合模型，指定增长为逻辑增长，这是必要的步骤以使用饱和趋势
        m = Prophet(growth='logistic')
        m.fit(df)
        # 创建未来日期的数据帧，需要包括饱和上限
        # future = m.make_future_dataframe(start='2025-05-28 00:00:00', periods=1024, freq='84S')
        start_time = pd.to_datetime('2025-05-26 00:00:00')
        periods = 1024
        interval = '84S'
        future = pd.DataFrame({'ds': pd.date_range(start=start_time, periods=periods, freq=interval)})
        future['cap'] = 342

        # 预测未来的数据
        forecast = m.predict(future)
        print("zhenjun debug-----------------------------------")
        print(forecast)
        print("zhenjun debug-----------------------------------")

        # df 是训练数据
        last_date = df['ds'].max()  # 训练数据最后一天

        # 预测结果，只取未来日期 > last_date 的行
        forecast_1024 = forecast.tail(1024)
        # future_forecast = forecast[forecast['ds'] > last_date]

        """ 算法逻辑的核心实现"""
        res = []

        """这个预测算法固定返回 1 作为预测值，预测值的数量是用户通过 self.fc_rows 指定"""
        # ts_list = [self.start_ts + i * self.time_step for i in range(self.rows)]
        timestamp_ms = forecast_1024['ds'].astype('int64') // 10 ** 6
        res.append(timestamp_ms.tolist())  # 设置预测结果时间戳列

        """生成全部为 1 的预测结果 """
        res_list = [1] * self.rows
        # res.append(res_list)
        res.append(forecast_1024['yhat'].tolist())

        """检查用户输入，是否要求返回预测置信区间上下界"""
        if self.return_conf:
            """对于没有计算预测置信区间上下界的算法，直接返回预测值作为上下界即可"""
            bound_list = [1] * self.rows
            # res.append(bound_list)  # 预测结果置信区间下界
            # res.append(bound_list)  # 预测结果执行区间上界
            res.append(forecast_1024['yhat_lower'].tolist())
            res.append(forecast_1024['yhat_upper'].tolist())

        """返回结果"""
        return {"res": res, "mse": 0}

    def set_params(self, params):
        """该算法无需任何输入参数，直接调用父类函数，不处理算法参数设置逻辑"""
        # 获取默认参数设置
        super().set_params(params)
        # 获取自定义参数
        filtered = {k: v for k, v in params.items() if k != "ts_list" and k != "list"}
        print(filtered)
        return
