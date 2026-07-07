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
- 交易日历独立同步，默认请求 `2000-01-01` 到当前年份后 2 年年底；Tushare 当前实际可返回到的未来日期以接口结果为准。
- 前复权价格字段日常刷新最近抓取窗口；若某只股票最新 `adj_factor` 相对上一交易日变化，则刷新该股票全历史前复权价。
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
python run_technical_signal.py backfill-trading-data --year 2026 --sleep-seconds 0.35
python run_technical_signal.py backfill-market-layers --year 2026
python run_technical_signal.py backfill-market-layers --year 2026 --force-signals
python run_technical_signal.py backfill-signal-layers --year 2026
python run_technical_signal.py backfill-stock-signals --start-date 2000-01-01 --end-date 2009-12-31
python run_technical_signal.py validate-data
python run_technical_signal.py refresh-dragon-leaders
python run_technical_signal.py process
python run_technical_signal.py evening-pipeline
python run_technical_signal.py run
python run_technical_signal.py run --days 90
python run_technical_signal.py report
python run_technical_signal.py factor-lab init-schema
python run_technical_signal.py factor-lab build-factors --start-date 20250101 --end-date 20260703
python run_technical_signal.py factor-lab evaluate --start-date 20250101 --end-date 20260703
python run_technical_signal.py factor-lab correlate --start-date 20250101 --end-date 20260703
python run_technical_signal.py factor-lab event-study --start-date 20250101 --end-date 20260703
python run_technical_signal.py factor-lab weights --start-date 20250101 --end-date 20260703
python run_technical_signal.py factor-lab backtest --start-date 20250101 --end-date 20260703 --model-name event_adjusted_v1 --top-n 20 --hold-days 5
python run_technical_signal.py factor-lab backtest-grid --start-date 20250101 --end-date 20260703 --model-name event_adjusted_v1
python run_technical_signal.py factor-lab walk-forward --start-date 20250101 --end-date 20260703 --top-n 20 --hold-days 5
python run_technical_signal.py factor-lab short-strength-backtest --start-date 20250101 --end-date 20260703
python run_technical_signal.py factor-lab trend-pure-evaluate --start-date 20250101 --end-date 20260703
python run_technical_signal.py factor-lab trend-pure-backtest --start-date 20250101 --end-date 20260703
python run_technical_signal.py factor-lab trend-pure-full-run --start-date 20250101 --end-date 20260703
python run_technical_signal.py factor-lab trend-pure-period-study --start-date 20000104 --end-date 20260703 --period-years 3 --top-n 20 --hold-days 5
python run_technical_signal.py factor-lab shadow-run --date latest
python run_technical_signal.py factor-lab report --date 20260703
python install_windows_task.py
```

说明：日常自动化使用拆分任务；`run` 保留为手动全流程；`--days` 用于首轮初始化或需要补历史数据时手动扩大抓取窗口。`backfill-daily` 用于慢速回补历史日线、复权因子和 daily_basic，默认每次 Tushare 请求后等待 1.2 秒，失败后可重新运行续补。`backfill-trading-data` 用于按日期区间补全市场个股资金流、涨跌停/炸板、龙虎榜、市场/行业/概念资金流；默认跳过看起来已经完整的交易日，加 `--force` 可强制重抓。`backfill-market-layers` 用于按日期区间补 `index_daily`、`global_index_daily` 和 `dragon_leader_daily`；龙头候选只做技术层评分，若当天底层全 A 信号不存在会先生成 `stock_signal_daily` / `theme_signal_daily`，加 `--force-signals` 会强制重算这些底层信号。`backfill-signal-layers` 用于按日期区间补齐观察池技术信号、全 A 个股交易信号和主题热度；默认跳过完整日期，并只补缺的全 A 个股/主题层，加 `--force` 才会强制重算三层；该命令按日期升序执行，最后一个交易日会落到 `latest_signals`。`backfill-stock-signals` 只回算全 A `stock_signal_daily`，适合早期没有资金流、涨跌停和龙虎榜数据的历史区间；它会落库 MA、RSI、MACD、量比、20日高低点和量价评分。`factor-lab` 是研究模块，输出因子评估、相关性、权重建议、组合回测和影子名单，不修改生产评分权重；晚间定时任务只执行轻量 `shadow-run`，不自动替换生产评分。Tushare 单接口失败会短重试，抓取/晚间流水线整条失败会等待 20 分钟重试，最多再试 2 次。

定时任务：

```text
TechnicalSignalCalendarMonthly      每月 1 日 05:00，更新交易日历。
TechnicalSignalGlobalIndexMorning   每天 06:40，更新海外主要指数结构化行情。
TechnicalSignalMarketDataDaily      每天 17:20，更新全 A 日线、复权因子和 daily_basic。
TechnicalSignalEveningPipelineDaily 每天 20:20，先更新全市场个股资金流、涨跌停/炸板、龙虎榜、行业/概念资金流，再串流执行分析、报告和 factor_lab 影子运行。
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
tech_signal.factor_daily
tech_signal.factor_performance
tech_signal.factor_correlation
tech_signal.factor_event_study
tech_signal.factor_data_coverage
tech_signal.model_weight_history
tech_signal.trend_pure_indicator_performance
tech_signal.trend_pure_combo_performance
tech_signal.factor_shadow_candidates
tech_signal.factor_shadow_tracking
tech_signal.strategy_backtest_result
tech_signal.strategy_backtest_trades
tech_signal.signal_runs
```

新增结构化层：

- `index_daily`：A 股主要指数日行情，覆盖上证指数、深成指、创业板指、科创50、沪深300、中证500、中证1000、北证50。
- `global_index_daily`：海外主要指数结构化行情，覆盖道指、标普500、纳指、德国DAX、法国CAC40、英国富时100、日经225、韩国KOSPI；`data_status` 标记最新收盘、同日/盘中或旧数据状态。
- `dragon_leader_daily`：技术层龙头候选排序，来自全 A 个股交易信号、涨跌停、龙虎榜和主题热度；只写技术评分、排序、原因和风险标签。
- `factor_daily` / `factor_performance` / `factor_correlation` / `factor_event_study` / `factor_data_coverage` / `model_weight_history` / `strategy_backtest_*`：因子研究和回测校准层，只做评估报告，不替换 `stock_signal_daily` 生产评分。
- `trend_pure_indicator_performance` / `trend_pure_combo_performance`：纯技术趋势模型 `trend_pure_v1` 的单指标检验和组合回测结果。
- `factor_shadow_candidates` / `factor_shadow_tracking`：因子模型影子运行层，保存每日模型名单、生产评分对比、共振/分歧类型，以及 3/5/10/20 日跟踪收益。

因子研究模块：

- 代码在 `factor_lab/`，报告输出到 `E:\technical_signals\factor_lab\reports\`。
- 从现有 `stock_signal_daily`、`daily_bars`、资金流、涨跌停和龙虎榜字段派生因子，不新增外部数据源。
- 覆盖趋势、量能、资金、短线情绪、龙虎榜、风险、反转和相对强弱因子。
- 收益口径统一为：`trade_date` 收盘后生成信号，下一交易日复权开盘买入，持有 N 个交易日，按复权收盘退出。
- 龙虎榜、资金流、涨跌停默认视为盘后可知事件，只用于次日交易研究，避免盘中未来函数。
- 风险因子保留原始风险值，但横截面 pct_rank 和组合打分统一转为正向口径：低风险为高分。
- 单因子检验输出 1/3/5/10 日未来收益窗口的 IC、RankIC、ICIR、top/bottom 20%、long-short、胜率、回撤、衰减和市场环境拆分。
- 事件研究单独评估龙虎榜、机构/北向净买、涨停、炸板、回封、强封单、连续资金流入等事件后的 1/3/5/10/20 日收益、超额收益、胜率、回撤和环境差异。
- 相关性模块用每日横截面相关性的时间窗口均值标记长期高相关因子；绝对相关高于 0.75 的因子在去相关权重里降权。
- 权重建议包括 `baseline_v1`、`ic_weighted_v1`、`decorrelated_v1`、`event_adjusted_v1` 和 `walk_forward_v1`；权重只写 `model_weight_history` 和报告，不写回生产评分配置。
- `trend_pure_v1` 是当前主线影子模型，只使用复权价、成交量、成交额、换手率和由它们派生出的技术指标；不使用资金流、龙虎榜、涨停、主题热度或生产评分权重。
- `trend_pure_v1` 会分别检验近 5/10/20/60 日相对强弱、均线多头、MA20/MA60 斜率、20/60/120 日新高、突破、均线回踩、MACD、RSI、量能放大、成交额/换手、长上影、乖离过热、跌破均线等指标。
- `trend_pure_v1` 组合回测包含 `ma_trend`、`breakout`、`pullback`、`macd_momentum`、`quality_trend`、`all_technical` 六组纯技术组合，支持 Top10/20/30 和 Hold3/5/10/20；执行约束按主板/创业板/科创板/北交所区分涨跌停幅度，剔除一字涨停买不进样本，一字跌停无法卖出时顺延退出，并加入单票买入金额不超过入场日成交额 0.5% 的默认容量约束。
- `event_adjusted_v1`、`walk_forward_v1` 和 `short_strength_v1` 保留为历史研究命令，不再进入每日默认 shadow-run。
- 训练/验证/测试按时间切分，训练集只用于估计因子有效性和权重，验证集用于参数稳定性观察，测试集只做最终验证。
- walk-forward 回测按滚动窗口训练权重，再应用到未来窗口；报告每期权重和每期表现。
- 组合回测支持 top 10/20/30、hold 1/3/5/10，使用收盘信号、次日开盘买入、固定持有期、等权、低成交额过滤、板块涨跌停过滤、一字跌停顺延退出、成交额容量约束、交易成本和滑点；当前用于比较因子方案，不作为生产交易执行系统。
- `factor-lab report --date latest` 会优先复用已经落库的因子、评价、事件、权重和回测结果，只补缺失步骤并生成汇总报告。
- 核心报告包括 `factor_data_coverage_YYYYMMDD.md`、`factor_performance_YYYYMMDD.md`、`event_study_YYYYMMDD.md`、`factor_correlation_YYYYMMDD.md`、`weight_suggestion_YYYYMMDD.md`、`strategy_backtest_YYYYMMDD.md` 和 `factor_lab_summary_YYYYMMDD.md`。
- `trend-pure-evaluate` 输出 `trend_pure_indicator_report_YYYYMMDD.md`，回答哪些纯技术指标有 T+3/T+5/T+10/T+20 相对收益。
- `trend-pure-backtest` 输出 `trend_pure_combo_report_YYYYMMDD.md`，比较不同纯技术组合的相对收益、最大回撤、胜率、盈亏比和可买入比例。
- `trend-pure-period-study` 按指定年数分段覆盖数据库已有历史，默认适合用 Top20 Hold5 / T+5 看指标和组合在不同时期的稳定性，输出 `trend_pure_period_study_YYYYMMDD.md`；全历史建议用 3 年分段控制内存峰值。
- 影子运行命令 `factor-lab shadow-run --date latest` 每天生成 `trend_pure_v1` Top20/Top30 Hold5/Hold10 名单，对比生产 `stock_signal_daily.total_signal_score` Top30，输出“生产评分强 + 纯技术趋势强”的共振票，以及模型强/生产弱、生产强/模型弱的分歧票。
- `trend_pure_v1` 影子收益跟踪 3/5/10/20 日。
- `evening-pipeline` 会在收盘后数据抓取、技术信号处理和日报生成完成后自动执行影子运行；影子输出只写 `factor_shadow_*` 表和 `factor_shadow_YYYYMMDD.md` 报告，不修改生产评分权重。

指标说明：

- 机器读取的指标公式和评分参数清单见 `config/indicator_formulas.json`。
- 人可读的技术指标、评分项和龙头候选字段说明见 `docs/indicator_calculation.md`。
