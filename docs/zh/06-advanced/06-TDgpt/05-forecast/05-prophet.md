---
title: "prophet"
sidebar_label: "prophet"
---

本节说明 prophet 时序数据预测分析模型使用方法。

## 功能概述

prophet：**Prophet** 是由 Facebook（Meta）开源的时间序列预测工具，专为处理具有明显趋势变化、强季节性以及节假日效应的业务数据而设计。在 TDengine 的 **TD-GPT** 系统中，Prophet 已作为可选预测算法被集成，用户可通过 SQL 查询直接调用。

## 可用参数列表

### 以下为 TDengine 中 Prophet 支持的常规参数:

| 参数名                        | 类型       | 默认值      | 说明                                               |
|------------------------------|------------|------------|----------------------------------------------------|
| `growth`  | string      | linear        | 趋势增长模型的类型，linear: 线性增长; logistic: 饱和式增长          |
| `changepoint_prior_scale`  | float      | 0.05        | 趋势突变敏感度，值越大越敏感                         |
| `changepoint_range`       | float      | 0.05        | 控制模型趋势变点（changepoints）搜索范围的参数，用于指定变点可以出现在数据的哪个时间段 |
| `daily_seasonality`        | bool       | auto        | 是否启用日季节性建模                               |
| `weekly_seasonality`       | bool       | auto        | 是否启用周季节性建模                               |
| `yearly_seasonality`       | bool       | auto        | 是否启用年季节性建模                               |
| `holidays`       | string       | 无        | 建模节假日效应的参数。节假日可能会对时间序列产生重要的干扰或影响 |
| `seasonality_mode`         | string     | additive    | 季节性模型：`additive` 或 `multiplicative`         |
| `seasonality_prior_scale`         | float     | 10.0    | 控制季节性项的复杂度或灵活性         |
| `holidays_prior_scale`         | float     | 10.0    | 控制节假日效应（holidays）的强度和拟合程度         |
| `mcmc_samples`                    | float      | 0          | 控制是否使用 贝叶斯采样（MCMC，即 Markov Chain Monte Carlo） 来对模型参数进行后验分布估计 |
| `uncertainty_samples`                      | float      | 1000          | 控制在 不使用 MCMC 采样（即 mcmc_samples=0）时，Prophet 如何估算预测不确定性（即置信区间）的 模拟次数                                     |
| `stan_backend`                 | string| CMDSTANPY          | Prophet 在底层使用 Stan（概率编程系统）进行模型拟合，该参数控制 Prophet 使用的 Stan 后端接口类型                     |

以下为 TDengine 中 Prophet 支持的重采样参数:

| 参数名                        | 类型       | 默认值      | 说明                                               |
|------------------------------|------------|------------|----------------------------------------------------|
| `resample`                 | string| 无          | 控制重采样                     |
| `resample_mode`                 | string| 无          | Prophet 在底层使用 Stan（概率编程系统）进行模型拟合，该参数控制 Prophet 使用的 Stan 后端接口类型                     |

### 以下为 TDengine 中 Prophet 支持的重饱和参数:

| 参数名                        | 类型       | 默认值      | 说明                                               |
|------------------------------|------------|------------|----------------------------------------------------|
| `saturating_cap_max`                 | bool | false          | 自动取输入参数最大值做为饱和上限                     |
| `saturating_cap`                 | float | 无          | 手动指定一个值做为饱和上限                     |
| `saturating_cap_scale`                 | float| 无          | 饱和上限的倍数                     |
| `saturating_floor_min`                 | bool | 无          |      自动取输入参数最大值做为饱和下限                |
| `saturating_floor`                 | float | 无          | 手动指定一个值做为饱和下限                     |
| `saturating_floor_scale`                 | float| 无          | 饱和下限的倍数                     |

### 以下为 TDengine 中 Prophet 支持的自定义节假日参数:

可自定义节假日（如补班、公司内部节日）

```json
[
  {"holiday": "spring_festival", "ds": "2023-01-22", "lower_window": -1, "upper_window": 2},
  {"holiday": "spring_festival", "ds": "2024-02-10", "lower_window": -1, "upper_window": 2},
  {"holiday": "spring_festival", "ds": "2025-01-29", "lower_window": -1, "upper_window": 2},
  {"holiday": "spring_festival", "ds": "2026-02-17", "lower_window": -1, "upper_window": 2}
]
```

### 参数示例

```sql
FORECAST(temperature, "algo=myprophet,rows=288,freq=5m,changepoint_prior_scale=0.1,interval_width=0.9,seasonality_mode=multiplicative,growth=logistic,holidays='[{\"holiday\":\"spring_festival\",\"ds\":\"2023-01-22\",\"lower_window\":-1,\"upper_window\":2},{\"holiday\":\"spring_festival\",\"ds\":\"2024-02-10\",\"lower_window\":-1,\"upper_window\":2},{\"holiday\":\"spring_festival\",\"ds\":\"2025-01-29\",\"lower_window\":-1,\"upper_window\":2},{\"holiday\":\"spring_festival\",\"ds\":\"2026-02-17\",\"lower_window\":-1,\"upper_window\":2}]'")
```

### 简单查询示例及结果

针对 i32 列进行数据预测，输入列 i32 每 10 个点是一个周期；start_p 起始是 1，最大拟合是 5；start_q 是 1，最大值是 5，预测结果中返回 95% 置信区间范围边界。

```
FORECAST(i32, "algo=myprophet,alpha=95,period=10,start_p=1,max_p=5,start_q=1,max_q=5")
```

完整的调用 SQL 语句如下：

```SQL
select _frowts, _fhigh, _frowts, FORECAST(temperature, "algo=myprophet,fc_rows=10,start_ts=1748361600000") from ( select * from test.boiler_temp where ts >= '2025-05-21 00:00:00' and ts < '2025-05-25 00:00:00' order by ts desc limit 30000) foo;
```

### 复杂查询示例及结果

针对 i32 列进行数据预测，输入列 i32 每 10 个点是一个周期；start_p 起始是 1，最大拟合是 5；start_q 是 1，最大值是 5，预测结果中返回 95% 置信区间范围边界。

```
FORECAST(i32, "algo=myprophet,alpha=95,period=10,start_p=1,max_p=5,start_q=1,max_q=5")
```

完整的调用 SQL 语句如下：

```SQL
SELECT _frowts, FORECAST(i32, "algo=myprophet,alpha=95,period=10,start_p=1,max_p=5,start_q=1,max_q=5") from foo
```

### 参考文献

- https://facebook.github.io/prophet/docs/quick_start.html#python-api
