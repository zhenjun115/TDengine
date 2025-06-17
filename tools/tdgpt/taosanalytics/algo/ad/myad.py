from taosanalytics.service import AbstractAnomalyDetectionService

# 算法实现类名称 需要以下划线 "_" 开始，并以 Service 结束
class _MyAdService(AbstractAnomalyDetectionService):
    """ 定义类，从 AbstractAnomalyDetectionService 继承，并实现 AbstractAnomalyDetectionService 类的抽象方法  """

    # 定义算法调用关键词，全小写 ASCII 码
    name = 'myad'

    # 该算法的描述信息 (建议添加)
    desc = """return the last value as the anomaly data"""

    def __init__(self):
        """类初始化方法"""
        super().__init__()

    def execute(self):
        """ 算法逻辑的核心实现"""

        # 打印方法参数
        print(f"execute method called with params: {self.params}")

        """检查数据是否为空"""

        """创建一个长度为 len(self.list)，全部值为 1 的结果数组，然后将最后一个值设置为 -1，表示最后一个值是异常值"""
        res = [1] * len(self.list)
        res[-1] = -1

        """返回结果数组"""
        return res

    def set_params(self, params):
        """该算法无需任何输入参数，直接重载父类该函数，不处理算法参数设置逻辑"""
        return super().set_params(params)