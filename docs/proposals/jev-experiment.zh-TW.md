# 提案：以 Jev（TypeSafe AI）做對照實驗（繁中閱讀版）

英文權威來源：[jev-experiment.md](jev-experiment.md)
類型：完整翻譯
來源版本：aef96c6，docs/proposals/jev-experiment.md
同步狀態：已核對

狀態：**草稿，待擁有者審查**。尚未接受，不在 [scope](../product/scope.zh-TW.md) 內；接受須更新 scope 或寫 ADR。

## Jev 是什麼（截至 2026-11-05）

TypeSafe AI 的「System One」模型，2026 年 9 月以 early access 推出。它不產生文字：每次呼叫針對預先定義選項的型別化問題，回傳一個選項值與機率。目標任務是分類、路由、評分與擷取。只能透過託管 API 使用（例如 Cloudflare Workers AI、DigitalOcean Serverless Inference），權重與架構未公開。

效能、延遲（端到端 70–500 ms）與價格數據多為廠商自報；未找到表格或詐欺資料上的獨立 benchmark。

來源：

- https://www.digitalocean.com/resources/articles/what-is-jev
- https://www.datacamp.com/blog/system-one-models-jev
- https://www.eesel.ai/blog/typesafe-jev
- https://www.vktr.com/ai-platforms/chatgpt-cocreator-launches-typesafe-ai-with-jev/

## 評估

不適合當 active 模型，也不適合放進服務路徑。

| 問題 | 與本專案的衝突 |
|---|---|
| 資料列送到第三方 API | 資料使用條款未確認；把資料列外送可能被禁止 |
| 沒有表格或詐欺 benchmark | IEEE-CIS 有數百個匿名數值欄位，正是梯度提升樹的強項 |
| 廠商自報延遲 70–500 ms | 吃掉同步決策延遲預算的一大塊（Phase 2） |
| 閉源、僅託管、可能無預警變動 | 與可重現（S-1）及 offline/online 一致性（S-10）衝突 |
| 只給機率，不給理由 | 決策須可解釋、可重播 |
| Scope 排除為展示而加的 LLM 功能 | Jev 雖不產生文字，性質接近 |

## 若接受，可做的實驗

僅限離線比較，不進服務路徑：

1. 前提：Kaggle 條款允許把資料列送給第三方；scope 或 ADR 已更新。
2. 從時間切分的 test 段抽樣；只送原始欄位，不送用到未來資料的工程特徵。
3. 與 LightGBM 比較 PR-AUC、固定 review 比例下的 precision/recall、校準、每筆決策成本與實測延遲，並記錄硬體與設定。

## 待擁有者回答

- Kaggle 條款是否允許把資料送到外部模型 API。
- 課程是否重視與託管基礎模型的比較。
