# Indicator Calculation Reference

本文件说明 `technical_signal_system` 当前已经实现的结构化数据字段、技术指标、派生状态和评分口径。系统只输出结构化市场事实和技术信号，不写宏观解读、盘前影响判断或基本面研究结论。

机器执行的公式和评分参数以 `config/indicator_formulas.json` 为准；本文件是给人看的说明。修改指标周期、阈值或权重时，应先改 JSON 清单，再同步更新本说明。

## 数据口径

- 计算技术指标使用 `daily_bars.adj_open/adj_high/adj_low/adj_close`，也就是前复权价格。
- 日常行情更新会刷新最近抓取窗口的前复权字段；遇到最新 `adj_factor` 变化的股票，会刷新该股票全历史前复权字段。
- 展示价格和涨跌幅使用 Tushare 原始日线字段，如 `close`、`pct_chg`。
- 成交量使用 `daily_bars.vol`，成交额使用 `daily_bars.amount`。Tushare 日线成交额单位为千元，系统在全 A 信号层换算为 `amount_yi = amount / 100000`。
- 换手率使用 `daily_basic.turnover_rate`。
- 个股资金流使用 `moneyflow_daily.net_mf_amount`。Tushare 资金流金额单位为万元，系统派生 `net_mf_amount_yi = net_mf_amount / 10000`。
- `stock_signal_daily` 默认每个交易日向前取近 180 个交易日历史生成全 A 信号。180 日不是一个指标周期，只是给 MA60、RSI14、MACD 和 20 日量价结构留足计算窗口。

## 滚动技术指标

这些指标由 `tech_signal.indicators.add_indicators()` 按股票分组、按交易日排序后计算。

| 字段 | 计算方式 | 输入 | 说明 |
| --- | --- | --- | --- |
| `ma5` | `adj_close` 的 5 日简单移动平均 | 前复权收盘价 | 至少 5 条历史才有值 |
| `ma10` | `adj_close` 的 10 日简单移动平均 | 前复权收盘价 | 至少 10 条历史才有值 |
| `ma20` | `adj_close` 的 20 日简单移动平均 | 前复权收盘价 | 趋势中枢和回踩判断使用 |
| `ma60` | `adj_close` 的 60 日简单移动平均 | 前复权收盘价 | 中期趋势背景使用 |
| `vol_ma5` | `vol` 的 5 日均量 | 成交量 | 含当日 |
| `vol_ma20` | `vol` 的 20 日均量 | 成交量 | 含当日 |
| `prev_vol_ma5` | `vol` 的 5 日均量再向后平移 1 日 | 成交量 | 不含当日，用于当日量比 |
| `prev_vol_ma20` | `vol` 的 20 日均量再向后平移 1 日 | 成交量 | 不含当日，用于突破/回踩确认 |
| `volume_ratio_5` | `vol / prev_vol_ma5` | 成交量 | 当日相对前 5 日均量 |
| `volume_ratio_20` | `vol / prev_vol_ma20` | 成交量 | 当日相对前 20 日均量 |
| `high20` | `adj_high` 的 20 日滚动最高值 | 前复权最高价 | 突破/接近新高判断使用 |
| `low20` | `adj_low` 的 20 日滚动最低值 | 前复权最低价 | 当前入库但第一版评分较少使用 |
| `bias5` | `(adj_close / ma5 - 1) * 100` | 前复权收盘价、MA5 | 5 日乖离率 |
| `rsi14` | 14 日平均上涨幅度 / 平均下跌幅度换算 RSI | 前复权收盘价 | 简单滚动均值版 RSI，非 Wilder 平滑版 |
| `macd` | `EMA12(adj_close) - EMA26(adj_close)` | 前复权收盘价 | EMA 使用 pandas `ewm(adjust=False)` |
| `macd_signal` | `macd` 的 9 日 EMA | MACD | MACD 信号线 |
| `macd_hist` | `macd - macd_signal` | MACD、signal | MACD 柱 |

## 量能状态

`volume_state` 使用 `volume_ratio_5` 和当天涨跌幅 `pct_chg`：

| 状态 | 条件 |
| --- | --- |
| `放量上涨` | `volume_ratio_5 >= 1.5` 且 `pct_chg > 0` |
| `放量下跌` | `volume_ratio_5 >= 1.5` 且 `pct_chg <= 0` |
| `缩量上涨` | `volume_ratio_5 <= 0.7` 且 `pct_chg > 0` |
| `缩量回调` | `volume_ratio_5 <= 0.7` 且 `pct_chg <= 0` |
| `量能正常` | 以上条件都不满足 |

阈值来自 `config/settings.json` 的 `signals.heavy_volume_ratio` 和 `signals.shrink_volume_ratio`。

## 全 A 个股信号

`stock_signal_daily` 是全 A 个股级结构化信号层。每只股票每个交易日写一行。

### 技术分 `technical_score`

初始分 50，范围限制到 0-100：

| 条件 | 分数影响 | 标签/风险 |
| --- | ---: | --- |
| `ma5 > ma10 > ma20` | `+18` | `多头排列` |
| `ma5 < ma10 < ma20` | `-22` | `空头排列` |
| `adj_close > ma20` | `+8` | `站上MA20` |
| `adj_close <= ma20` | `-10` | `跌破MA20` |
| `adj_close > ma60` | `+6` | `站上MA60` |
| `adj_close <= ma60` | `-5` | 无新增风险标签 |
| `macd_hist > 0` | `+7` | `MACD偏强` |
| `macd_hist <= 0` | `-5` | 无新增风险标签 |
| `bias5 >= 8` | `-14` | `短线乖离过大` |
| `rsi14 >= 80` | `-10` | `RSI过热` |

`trend_phase` 当前按技术结构派生：

| 阶段 | 条件 |
| --- | --- |
| `breakout` | 有 `多头排列`，且 `adj_close >= high20 * 0.995` |
| `uptrend` | 有 `多头排列` |
| `weakening` | 有 `空头排列` 或 `跌破MA20` |
| `sideways` | 其他情况 |

### 价量分 `price_volume_score`

初始分 50，范围限制到 0-100：

```text
price_volume_score =
  50
  + clamp(pct_chg, -20, 20) * 1.5
  + clamp(amount_yi, 0, 120) * 0.16
  + clamp(turnover_rate, 0, 30) * 0.55
  + clamp(volume_ratio_5, 0, 8) * 2.0
  + volume_state_adjustment
```

`volume_state_adjustment`：

| 状态 | 分数影响 | 标签/风险 |
| --- | ---: | --- |
| `放量上涨` | `+8` | `放量上涨` |
| `放量下跌` | `-10` | `放量下跌` |
| `缩量回调` | `0` | `缩量回调` |
| `缩量上涨` | `-4` | `缩量上涨` |

### 资金分 `moneyflow_score`

如果当天没有资金流数据，默认 50。否则：

```text
moneyflow_score =
  50
  + clamp(net_mf_amount_yi, -8, 8) * 3.0
  + clamp(net_mf_rate, -12, 12) * 1.2
```

其中：

```text
net_mf_amount_yi = net_mf_amount / 10000
net_mf_rate = net_mf_amount * 1000 / daily_bars.amount
```

标签/风险：

- `net_mf_amount_yi > 0` 或 `net_mf_rate > 0`：`资金净流入`
- `net_mf_amount_yi < 0` 或 `net_mf_rate < 0`：`资金净流出`

### 涨跌停分 `limit_score`

来自 `limit_events`：

| 情况 | 分数 | 标签/风险 |
| --- | ---: | --- |
| 无涨跌停/炸板事件 | `50` | 无 |
| 涨停 `limit_type='U'` | `82 + min(limit_times, 5) * 3 - min(open_times, 5) * 2` | `涨停`，二连板以上加 `N连板` |
| 跌停 `limit_type='D'` | `15` | `跌停` |
| 炸板 `limit_type='Z'` | `42` | `炸板` |

### 龙虎榜分 `lhb_score`

没有龙虎榜数据时默认 50。否则：

```text
lhb_score =
  50
  + clamp(lhb_net_buy_yi, -8, 8) * 2.2
  + clamp(institution_net_buy_yi, -5, 5) * 3.0
  + clamp(northbound_net_buy_yi, -5, 5) * 2.0
  + clamp(amount_rate, 0, 30) * 0.18
```

标签/风险：

- 龙虎榜净买为正：`龙虎榜净买`
- 龙虎榜净买为负：`龙虎榜净卖`
- 机构净买为正：`机构净买`
- 陆股通净买为正：`陆股通净买`

### 总分 `total_signal_score`

```text
total_signal_score =
  technical_score * 0.45
  + price_volume_score * 0.25
  + moneyflow_score * 0.15
  + limit_score * 0.10
  + lhb_score * 0.05
```

`signal_level`：

| 等级 | 条件 |
| --- | --- |
| `strong` | `total_signal_score >= 78` |
| `watch` | `62 <= total_signal_score < 78` |
| `risk` | `total_signal_score <= 35` |
| `neutral` | 其他 |

## 主题信号

`theme_signal_daily` 分行业和概念两类。

行业热度：

```text
heat_score =
  50
  + clamp(net_amount_yi, -50, 50) * 0.8
  + clamp(pct_chg, -10, 10) * 2.0
  + limit_up_count * 3.0
  + strong_stock_count * 2.0
```

行业动量：

```text
momentum_score =
  50
  + clamp(pct_chg, -10, 10) * 3.0
  + limit_up_count * 2.0
  - broken_count * 1.5
```

概念热度：

```text
heat_score =
  50
  + clamp(net_amount_yi, -50, 50) * 0.8
  + clamp(pct_chg, -10, 10) * 2.0
```

概念动量：

```text
momentum_score =
  50
  + clamp(pct_chg, -10, 10) * 3.0
```

`signal_level` 与个股信号相同：`strong >= 78`，`watch >= 62`，`risk <= 35`，其他为 `neutral`。

## 龙头候选

`dragon_leader_daily` 从 `stock_signal_daily`、`theme_signal_daily`、涨跌停和龙虎榜信息生成。它只输出技术层候选，不做基本面判断。

候选过滤：

- 默认要求 `amount_yi >= 20`。
- 每日最多保留 `top_n = 300`。

`leader_score` 初始 35，范围限制到 0-120：

```text
leader_score =
  35
  + clamp(pct_chg, -20, 20) * 1.6
  + clamp(amount_yi, 0, 300) * 0.08
  + clamp(turnover_rate, 0, 30) * 0.55
  + clamp(volume_ratio, 0, 8) * 1.8
  + clamp(total_signal_score - 50, -30, 40) * 0.35
  + limit_adjustment
  + clamp_positive(lhb_net_buy_yi, 10) * 1.4
  + clamp_positive(institution_net_buy_yi, 5) * 2.0
  + clamp_positive(northbound_net_buy_yi, 5) * 1.6
  + theme_adjustment
```

其中：

- 涨停：`limit_adjustment = 16 + min(limit_times, 5) * 3.5`
- 炸板：`limit_adjustment = 3`
- 无涨停/炸板：`limit_adjustment = 0`
- 主题加分：如果命中行业/概念主题，取最高 `heat_score`，`theme_adjustment = max(heat_score - 60, 0) * 0.25`

排序规则：

- 先按 `leader_score` 降序。
- 分数相同按 `amount_yi` 降序。

`leader_level`：

| 等级 | 条件 |
| --- | --- |
| `strong` | `leader_score >= 88` |
| `watch` | `68 <= leader_score < 88` |
| `candidate` | 其他入选候选 |

风险标签继承 `stock_signal_daily.risk_flags`，并额外补充：

- 炸板：`炸板`
- 换手率 `>= 25`：`高换手`
- `volume_state='放量下跌'`：`放量下跌`

## 轻量日报信号

`technical_signals` 是面向自选股和观察池日报的轻量信号层。它使用同一套滚动指标，但评分较简单：

- `MA5 > MA10 > MA20`：加分并标记 `多头排列`
- `MA5 < MA10 < MA20`：扣分并标记 `空头排列`
- 站上/跌破 `MA20`、站上/跌破 `MA60` 调整趋势分
- 接近 `high20` 且 `volume_ratio_20 >= 1.3`：标记 `放量突破`
- 多头排列下接近 `MA10` 或 `MA20`，且 `volume_ratio_20 <= 0.9`：标记 `缩量回踩`
- `macd_hist > 0`：标记 `MACD偏强`
- `bias5 >= 8`：标记 `短线乖离过大`
- `rsi14 >= 80`：标记 `RSI过热`
- 资金流为正/负会小幅加减分

这张表暂时主要服务技术日报，不作为全市场龙头排序主表。
