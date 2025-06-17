from taosanalytics.algo.tsfm import TsfmBaseService
# encoding:utf-8
# pylint: disable=c0103
""" Deepseek MOE algorithms to detect anomaly for time series data"""
class _DeepseekMOEService(TsfmBaseService):
    """
    Deepseek MOE service.
    """

    name = 'deepseek-moe-fc'

    desc = ("Deepseek MOE: Billion-Scale Time Series Foundation Models with Mixture of Experts; ")

    def __init__(self):
        super().__init__()
        self._model = None
        if self.service_host is None:
            self.service_host = 'http://127.0.0.1:8001/ds_predict'

    def execute(self):
        # 检查是否支持历史协变量分析，如果不支持，触发异常。time-moe 不支持历史协变量分析，因此触发异常
        if len(self.past_dynamic_real):
            raise ValueError("covariate forecast is not supported yet")

        # 调用父类的 execute 方法
        super().execute()