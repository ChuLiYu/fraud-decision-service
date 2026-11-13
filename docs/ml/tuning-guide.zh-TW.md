# 可調參數指南（繁中閱讀版）

英文權威來源：[tuning-guide.md](tuning-guide.md)
類型：完整翻譯
來源版本：與英文同一次提交（M3b），docs/ml/tuning-guide.md
同步狀態：已核對

說明 [`ml/params.yaml`](../../ml/params.yaml) 每個參數的作用、調整後的變化與要觀察的指標。數值只存在 `params.yaml`；本指南負責解釋，並隨里程碑擴充。

任何改動都會改變 manifest 的 params hash，訓練前須重跑 `make prepare`。

## split

| 參數 | 作用 | 調高 | 調低 | 觀察 |
|---|---|---|---|---|
| `train_fraction` | 最早一段資料的比例，用來 fit 前處理與模型 | 可學的歷史更多；validation 與 test 往後移且變小 | 訓練資料變少；評估資料變多 | 各段詐欺筆數（manifest 的 `split.ranges`）；詐欺太少會讓指標不穩 |
| `valid_fraction` | 用於調參與選門檻的比例 | 調參訊號較穩；test 變小 | 調參較雜訊 | test 必須保留足夠詐欺筆數，PR-AUC 才穩定 |

切點會移到下一個時間戳變化處，所以同一個 `TransactionDT` 不會跨兩段。各段不打亂順序，這正是時間切分的目的。

## features

在 `ml/src/fraud_ml/features.py` 中，只用該筆交易本身與更早的交易產生特徵（leakage rule 2）。

| 參數 | 作用 | 改動時 | 觀察 |
|---|---|---|---|
| `enabled` | 開關所有工程特徵 | `false` 重現 M2 的特徵集合，用於 ablation | `reports/comparison.md` 的 validation PR-AUC |
| `group_keys` | 每個清單定義一組「這些欄位值相同」的交易；每組新增 4 個歷史特徵（之前筆數、之前平均金額、金額比、距上一筆秒數） | key 越多或越長：分組越細、特徵越多、每組的歷史筆數越少 | 同上；之後也看 LightGBM 特徵重要度 |

分組只是欄位值相同，不是已驗證的客戶（dataset card）。

## preprocess

| 參數 | 作用 | 調高 | 調低 | 觀察 |
|---|---|---|---|---|
| `min_category_count` | 在訓練段出現次數低於此值的類別值視為未知（`-1`） | 類別較少、較可靠；較不會記住稀有值 | 類別較多，含模型可能過度擬合的稀有值 | `preprocessor.json` 中各欄的類別數；validation 與 test 結果的落差 |

## seed

固定 LightGBM（列與特徵抽樣）與 baseline 的隨機性。換 seed 後指標變化很大，表示該結果在雜訊範圍內。`make tune` 會直接量測這個雜訊（見下方 `tuning.seeds`）；兩個設定的差距要先和不同 seed 的標準差比較，再判斷是否真的進步。

## baseline（logistic regression）

數值特徵依訓練段學到的分位數轉成常態分布，因此超出訓練範圍的值不會把模型推到極端分數。

| 參數 | 作用 | 調高 | 調低 | 觀察 |
|---|---|---|---|---|
| `C` | 正則化強度的倒數 | 正則化較弱，更貼近訓練資料 | 正則化較強，權重較小、模型較簡單 | validation PR-AUC 與訓練段的差距 |
| `max_iter` | 求解器迭代上限 | 讓求解器收斂，較慢 | 較快，可能未收斂就停止 | models manifest 的 `baseline.iterations`：等於 `max_iter` 表示未收斂 |

## lightgbm

| 參數 | 作用 | 調高 | 調低 | 觀察 |
|---|---|---|---|---|
| `learning_rate` | 每棵樹修正前面樹的幅度 | 需要的輪數較少，可能修過頭 | 輪數與時間增加，通常略好 | 報告中的 `best_iteration` |
| `num_leaves` | 每棵樹的葉子上限（複雜度） | 能捕捉更多交互作用，較易過度擬合 | 樹較簡單，可能擬合不足 | validation 與 test 的落差 |
| `min_child_samples` | 每片葉子的最少筆數 | 規則較平滑、較通用 | 規則來自較少資料，可能記住少數案例 | 同上；詐欺稀少，值太低容易過度擬合 |
| `feature_fraction` | 每棵樹抽樣的特徵比例 | 每棵樹看到更多特徵 | 隨機性較高，較不依賴少數強特徵 | 不同 seed 下結果的穩定度 |
| `bagging_fraction`、`bagging_freq` | 每輪的列抽樣（`freq` 為 0 則關閉） | 隨機性較低 | 隨機性較高、較快 | 同上 |
| `lambda_l2` | 葉值的 L2 懲罰 | 葉值較小、較保守 | 葉值更貼近訓練資料 | 校準（ECE）與 validation PR-AUC |
| `max_rounds` | 樹的數量上限 | 早停未觸發時可跑更多輪 | 限制訓練時間 | `best_iteration` 等於 `max_rounds` 時應調高 |
| `early_stopping_rounds` | validation PR-AUC 連續幾輪未改善就停止 | 容忍停滯期，較慢 | 較早停止，可能停太早 | `best_iteration` |

這些參數只能對照 **validation** 段調整。test 段每種設定只用來產生一次最終結果。

## tuning

只有 `make tune` 會讀取：對每種設定在 train 段訓練 LightGBM、對 validation 段評分，並寫出 [`ml/reports/tuning/tuning.md`](../../ml/reports/tuning/tuning.md)。它不讀 test 段，也不修改 `params.yaml`：由擁有者把選定的值抄進 `lightgbm` 段，該次 commit 就是決策紀錄。

| 參數 | 作用 | 觀察 |
|---|---|---|
| `grid` | 要嘗試的 LightGBM 參數與值；每種組合訓練一次（成本按組合數相乘） | 最佳值落在 grid 邊界時，最佳點可能在範圍之外 |
| `seeds` | 用來重訓「目前設定」與「grid 最佳設定」的 seed | 不同 seed 的標準差：差距小於約兩個標準差，不算可靠的進步 |

分數有兩個限制：早停也使用 validation 段，而且在同一段挑出多個設定中的最佳者會帶來選擇偏差，所以 validation 分數偏樂觀。它只用來排序設定；不偏的結果看下一次評估執行的 test 段。

## evaluation

這些參數只改變結果的彙整方式，不改變模型。

| 參數 | 作用 |
|---|---|
| `review_rates` | 送審交易比例；在這些比例下報告 precision 與 recall。審查人力是情境參數（dataset card），所以列出多個比例 |
| `calibration_bins` | 可靠度表的等筆數分箱數。分箱越多細節越多，但每箱雜訊也越大 |
| `time_slices` | 每段依時間順序切成的等筆數切片數。切片越多越能看出漂移，但每片的詐欺筆數越少 |

## 規劃中

決策門檻與成本（M4）會在該階段完成時加入。
