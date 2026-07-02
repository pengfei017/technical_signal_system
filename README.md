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
- `tech_signal.index_daily` / `tech_signal.global_index_daily` 只提供结构化指数事实，不写宏观判断或盘前影响判断。
- `tech_signal.dragon_leader_daily` 只提供技术层龙头候选排序和评分解释，不做基本面判断。

默认范围：

- 自选股。
- 观察池 `priority` / `watching` / `A` / `B` 候选。
- 全 A 股保留数据库已有历史；首轮或手动 `--days` 回补时使用 90 个交易日作为默认历史窗口。
- 日常定时任务只抓最近 5 个交易日的日线、复权因子和基础估值数据，降低 Tushare 调用压力。
- 交易日历独立同步，默认请求 `2010-01-01` 到当前年份后 2 年年底；Tushare 当前实际可返回到的未来日期以接口结果为准。
- 前复权价格字段每天按数据库已有的全部 `daily_bars` 历史重算，不局限于 90 天。
- 个股资金流默认按交易日拉全市场 `pro.moneyflow(trade_date=...)`，日常只拉最近 5 个交易日。
- 成交量和成交额随日线入库；量能状态使用不含当日的前 5 日均量判断，放量突破/缩量回踩使用不含当日的前 20 日均量确认。
- `stock_signal_daily` 不局限于观察池/自选股范围；默认使用近 180 个交易日生成全 A 个股交易信号，避免历史库变大后每天重复扫描多年全量数据。
- 日报里的个股交易信号和龙虎榜确认暂时只展示 `signal_universe`，也就是自选股 + 观察池；全市场结果先保留在数据库里。
- 涨停/炸板/跌停使用 Tushare `limit_list_d`，龙虎榜使用 `top_list` / `top_inst`，市场/行业/概念资金流使用 Tushare 对应资金流接口；抓取阶段会记录失败细节，晚间处理和报告前会统一验数，验不过不出新报告。

常用命令：

```powershell
cd D:\technical_signal_system
python run_technical_signal.py init-db
python run_technical_signal.py update-calendar
python run_technical_signal.py update-market-data
python run_technical_signal.py update-trading-data
python run_technical_signal.py update-indexes
python run_technical_signal.py update-global-indexes
python run_technical_signal.py backfill-daily --year 2010
python run_technical_signal.py backfill-daily --start-date 2010-01-01 --end-date 2010-12-31
python run_technical_signal.py backfill-market-layers --year 2026
python run_technical_signal.py backfill-market-layers --year 2026 --force-signals
python run_technical_signal.py validate-data
python run_technical_signal.py refresh-dragon-leaders
python run_technical_signal.py process
python run_technical_signal.py evening-pipeline
python run_technical_signal.py run
python run_technical_signal.py run --days 90
python run_technical_signal.py report
python install_windows_task.py
```

说明：日常自动化使用拆分任务；`run` 保留为手动全流程；`--days` 用于首轮初始化或需要补历史数据时手动扩大抓取窗口。`backfill-daily` 用于慢速回补历史日线、复权因子和 daily_basic，默认每次 Tushare 请求后等待 1.2 秒，失败后可重新运行续补。`backfill-market-layers` 用于按日期区间补 `index_daily`、`global_index_daily` 和 `dragon_leader_daily`；龙头候选只做技术层评分，若当天底层全 A 信号不存在会先生成 `stock_signal_daily` / `theme_signal_daily`，加 `--force-signals` 会强制重算这些底层信号。Tushare 单接口失败会短重试，抓取/晚间流水线整条失败会等待 20 分钟重试，最多再试 2 次。

定时任务：

```text
TechnicalSignalCalendarMonthly      每月 1 日 05:00，更新交易日历。
TechnicalSignalGlobalIndexMorning   每天 06:40，更新海外主要指数结构化行情。
TechnicalSignalMarketDataDaily      每天 17:20，更新全 A 日线、复权因子和 daily_basic。
TechnicalSignalEveningPipelineDaily 每天 20:20，先更新全市场个股资金流、涨跌停/炸板、龙虎榜、行业/概念资金流，再串流执行分析和报告。
```

晚间流水线会先检查全 A 日线是否已经到最新应开市交易日；如果 17:20 的日线任务失败或没有抓到当天数据，会先补跑一次 `update-market-data`。分析和报告前会校验当天日线、前复权字段、daily_basic、全市场个股资金流、涨跌停/炸板、龙虎榜、市场/行业/概念资金流是否入库；校验失败时任务记为 failed，等待任务级重试或下一个定时周期再跑。

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
tech_signal.index_daily
tech_signal.global_index_daily
tech_signal.limit_events
tech_signal.limit_market_stats
tech_signal.lhb_stocks
tech_signal.lhb_seats
tech_signal.signal_universe
tech_signal.technical_signals
tech_signal.latest_signals
tech_signal.stock_signal_daily
tech_signal.theme_signal_daily
tech_signal.dragon_leader_daily
tech_signal.signal_runs
```

新增结构化层：

- `index_daily`：A 股主要指数日行情，覆盖上证指数、深成指、创业板指、科创50、沪深300、中证500、中证1000、北证50。
- `global_index_daily`：海外主要指数结构化行情，覆盖道指、标普500、纳指、德国DAX、法国CAC40、英国富时100、日经225、韩国KOSPI；`data_status` 标记最新收盘、同日/盘中或旧数据状态。
- `dragon_leader_daily`：技术层龙头候选排序，来自全 A 个股交易信号、涨跌停、龙虎榜和主题热度；只写技术评分、排序、原因和风险标签。
