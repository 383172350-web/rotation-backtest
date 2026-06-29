# 轮动策略回测系统 — 部署指南

## 部署状态

✅ 所有代码已推送到 GitHub: `https://github.com/383172350-web/rotation-backtest`

## 手动部署到 Streamlit Cloud（推荐）

由于 WebBridge 浏览器扩展未连接，请按以下步骤手动部署：

### 1. 访问部署页面
打开浏览器，访问：
```
https://share.streamlit.io/new
```

### 2. 填写部署信息

| 字段 | 填写内容 |
|------|----------|
| **Repository** | `383172350-web/rotation-backtest` |
| **Branch** | `master` |
| **Main file path** | `streamlit_app.py` |
| **App URL** | `rotation-backtest`（自定义短名） |

### 3. 点击 Deploy
等待约 2-3 分钟，系统会自动：
- 从 GitHub 拉取代码
- 安装依赖（requirements.txt）
- 解压内置的 9.86MB pkl 数据包
- 启动应用

### 4. 访问应用
部署成功后，访问地址为：
```
https://rotation-backtest.streamlit.app
```
（如果自定义了其他短名，请替换）

---

## 本地运行方式

如果云端部署遇到问题，也可以在本地运行：

```bash
cd rotation-web
pip install -r requirements.txt
streamlit run streamlit_app.py
```

本地运行时，侧边栏可配置数据路径：
- **默认**：优先使用 `D:\qmt_data\ETF\1d`（如果存在）
- **云端**：自动解压内置的 `data_etf_1d.zip`（450只ETF日线数据）
- **增量**：本地缺失的品种自动用 yfinance 补充并缓存

---

## 核心功能

1. **1912只 ETF+LOF 标的池** — 可视化多选
2. **排序公式构建器** — 支持 MA/EMA/RSI/MACD/ATR/BOLL/KDJ/RSRS/returns 等指标
3. **买卖规则构建器** — 自定义买入/卖出条件
4. **4种预设策略** — 全品类DIFv轮动、五斗米动量、精选LOF、动量+RSRS
5. **双数据源** — 本地pkl优先 + yfinance增量下载

---

## 数据说明

| 来源 | 说明 |
|------|------|
| 内置 zip | 450只ETF日线数据，压缩后 9.86MB，已提交到GitHub |
| 本地路径 | 支持 `D:\qmt_data\ETF\1d` 等本地目录 |
| yfinance | 云端自动补充缺失品种（A股ETF后缀为 .SS/.SZ） |

---

部署遇到任何问题请告诉我！
