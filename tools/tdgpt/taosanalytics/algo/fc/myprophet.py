import numpy as np
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

    def __init__(self):
        """类初始化方法"""
        super().__init__()

    def execute(self):
        # df = pd.DataFrame({
        #     'ds': pd.date_range(start='2020-01-01', periods=80, freq='D'),
        #     'y': [min(100, i + np.random.randn() * 10) for i in range(80)]
        # })

        data = self.__dict__
        df = pd.DataFrame({
            'ds': pd.to_datetime(data['ts_list'], unit='ms'),
            'y': data['list']
        })

        df = df.sort_values(by='ds').reset_index(drop=True)

        # 设置饱和上限
        df['cap'] = 100

        # 初始化并拟合模型，指定增长为逻辑增长，这是必要的步骤以使用饱和趋势
        m = Prophet(growth='logistic')
        m.fit(df)

        # 创建未来日期的数据帧，需要包括饱和上限
        future = m.make_future_dataframe(periods=self.rows)
        future['cap'] = 100

        # 预测未来的数据
        forecast = m.predict(future)
        print("zhenjun debug-----------------------------------")
        print(forecast.tail(self.rows))
        print("zhenjun debug-----------------------------------")

        # df 是训练数据
        last_date = df['ds'].max()  # 训练数据最后一天

        # 预测结果，只取未来日期 > last_date 的行
        future_forecast = forecast[forecast['ds'] > last_date]

        """ 算法逻辑的核心实现"""
        res = []

        """这个预测算法固定返回 1 作为预测值，预测值的数量是用户通过 self.fc_rows 指定"""
        ts_list = [self.start_ts + i * self.time_step for i in range(self.rows)]
        res.append(ts_list)  # 设置预测结果时间戳列

        """生成全部为 1 的预测结果 """
        res_list = [1] * self.rows
        # res.append(res_list)
        res.append(future_forecast['yhat'].tolist())

        """检查用户输入，是否要求返回预测置信区间上下界"""
        if self.return_conf:
            """对于没有计算预测置信区间上下界的算法，直接返回预测值作为上下界即可"""
            bound_list = [1] * self.rows
            # res.append(bound_list)  # 预测结果置信区间下界
            # res.append(bound_list)  # 预测结果执行区间上界
            res.append(future_forecast['yhat_lower'].tolist())
            res.append(future_forecast['yhat_upper'].tolist())

        """返回结果"""
        return {"res": res, "mse": 0}

    def set_params(self, params):
        """该算法无需任何输入参数，直接调用父类函数，不处理算法参数设置逻辑"""
        return super().set_params(params)
