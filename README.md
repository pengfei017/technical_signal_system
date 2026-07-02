# Technical Signal System

独立 A 股交易信号子系统，代码放在 `D:\technical_signal_system`，数据和输出放在 `E:\technical_signals`。

定位：

- 只做 A 股量价/技术结构、资金流、涨跌停生态、龙虎榜确认、主题热度和结构化风险标签。
- 不写投研结论，不读取知识星球、雪球、格隆汇、研报或公告正文。
- 与主投研系统独立运行。
- 共用现有 PostgreSQL 实例，但只写 `tech_signal` schema。
- 第一阶段先影子运行，不接入主系统复盘、观察池和网页。

消费边界：

- `tech_signal.stock_signal_daily` 是未来观察池粗筛主表。
- `tech_signal.theme_signal_daily` 用于复盘、主题雷达和市场背景。
- `tech_signal.limit_market_stats` 只描述短线生态，如涨停家数、炸板率和连板高度。
- `tech_signal.lhb_stocks` / `tech_signal.lhb_seats` 只做龙虎榜确认或风险提示。
- 行业热度、概念资金流、大盘资金流、涨停家数、炸板率和连板高度不能单独推股票；必须先有个股级交易信号。

默认范围：

- 自选股。
- 观察池 `priority` / `watching` / `A` / `B` 候选。
- 全 A 股保留数据库已有历史；首轮或手动 `--days` 回补时使用 90 个交易日作为默认历史窗口。
- 日常定时任务只抓最近 5 个交易日的日线、复权因子和基础估值数据，降低 Tushare 调用压力。
- 交易日历独立同步，默认请求 `2010-01-01` 到当前年份后 2 年年底；Tushare 当前实际可返回到的未来日期以接口结果为准。
- 前复权价格字段每天按数据库已有的全部 `daily_bars` 历史重算，不局限于 90 天。
- 资金流第一阶段只拉重点股票池最近 5 个交易日，避免过早触发 Tushare 频率压力。
- 成交量和成交额随日线入库；量能状态使用不含当日的前 5 日均量判断，放量突破/缩量回踩使用不含当日的前 20 日均量确认。
- `stock_signal_daily` 每次从数据库已有全 A 历史重算，不局限于观察池/自选股范围。
- 涨停/炸板/跌停使用 Tushare `limit_list_d`，龙虎榜使用 `top_list` / `top_inst`，市场/行业/概念资金流使用 Tushare 对应资金流接口；这些辅助接口失败时记录 metrics，不阻断日线和技术信号。

常用命令：

```powershell
cd D:\technical_signal_system
python run_technical_signal.py init-db
python run_technical_signal.py run
python run_technical_signal.py run --days 90
python run_technical_signal.py report
python install_windows_task.py
```

说明：日常运行使用 `python run_technical_signal.py run`；`--days` 用于首轮初始化或需要补历史数据时手动扩大抓取窗口。

输出：

```text
E:\technical_signals\reports\YYYYMMDD_technical_signal_report.md
E:\technical_signals\logs\technical_signal.log
```

数据库：

```text
tech_signal.trade_calendar
tech_signal.daily_bars
tech_signal.daily_basic
tech_signal.moneyflow_daily
tech_signal.moneyflow_stock
tech_signal.moneyflow_market
tech_signal.moneyflow_industry
tech_signal.moneyflow_concept
tech_signal.limit_events
tech_signal.limit_market_stats
tech_signal.lhb_stocks
tech_signal.lhb_seats
tech_signal.signal_universe
tech_signal.technical_signals
tech_signal.latest_signals
tech_signal.stock_signal_daily
tech_signal.theme_signal_daily
tech_signal.signal_runs
```
