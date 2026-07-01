# Technical Signal System

独立技术信号层，代码放在 `D:\technical_signal_system`，数据和输出放在 `E:\technical_signals`。

定位：

- 只做 A 股技术信号、趋势结构、量价状态和风险标签。
- 与主投研系统独立运行。
- 共用现有 PostgreSQL 实例，但只写 `tech_signal` schema。
- 第一阶段先影子运行，不接入主系统复盘、观察池和网页。

默认范围：

- 自选股。
- 观察池 `priority` / `watching` / `A` / `B` 候选。
- 全 A 股保留数据库已有历史；首轮或手动 `--days` 回补时使用 90 个交易日作为默认历史窗口。
- 日常定时任务只抓最近 5 个交易日的日线、复权因子和基础估值数据，降低 Tushare 调用压力。
- 前复权价格字段每天按数据库已有的全部 `daily_bars` 历史重算，不局限于 90 天。
- 资金流第一阶段只拉重点股票池最近 5 个交易日，避免过早触发 Tushare 频率压力。
- 成交量和成交额随日线入库；量能状态使用不含当日的前 5 日均量判断，放量突破/缩量回踩使用不含当日的前 20 日均量确认。

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
tech_signal.signal_universe
tech_signal.technical_signals
tech_signal.latest_signals
tech_signal.signal_runs
```
