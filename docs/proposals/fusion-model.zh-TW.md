# 提案：Fusion model（第二個模型與 LightGBM 結合）（繁中閱讀版）

英文權威來源：[fusion-model.md](fusion-model.md)
類型：完整翻譯
來源版本：與英文同一次提交（M3c），docs/proposals/fusion-model.md
同步狀態：已核對

狀態：**部分接受。** 離線 fusion ablation 已列入 [scope](../product/scope.zh-TW.md) 的 S-18（[ADR-004](../decisions/ADR-004-course-first-priority.md)）。在服務路徑中使用 fusion 仍是待擁有者審查的草稿，須另寫 ADR。

## 想法

在 LightGBM 之外再訓練一個模型，把兩者分數合併（平均、加權混合，或用 meta-model 做 stacking）成單一詐欺分數。

## 接受前需解決的問題

1. **Stacking 洩漏。** Meta-model 只能用「時間晚於子模型訓練段」的預測來訓練（例如 validation 段的預測），不能用 in-sample 預測。此規則已加入 dataset card（leakage rule 5），離線 ablation（`make fuse`）依此執行。
2. **部分失敗語意。** 一個子模型成功、另一個逾時時，結果必須事先定義。依 [ADR-003](../decisions/ADR-003-failure-routes-to-review.md) 最保守的做法是整筆 `review`；改成退回單一模型需另寫 ADR。
3. **版本。** Model version 變成組合版本（fusion 版本加上每個子模型版本），decision log 全部記錄。
4. **證據門檻。** Fusion 必須在每個時間切片上贏過單獨的 LightGBM，且幅度事先訂好，足以抵銷增加的延遲、一致性檢查（S-10）與維運成本。

## 建議路徑

先讓第二個模型以 shadow 方式評分（S-7），依儲存的 active/shadow 比較結果決定是否 fusion，而不是只看離線結果。在選定 fusion 之前不新增 scope。

## 待擁有者回答

- 第二個模型是什麼，以及為什麼預期它犯的錯和 LightGBM 不同。
- 課程是否評分建模深度（會改變優先順序；見 scope 的未解問題）。
