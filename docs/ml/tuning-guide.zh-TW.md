# 可調參數指南（繁中閱讀版）

英文權威來源：[tuning-guide.md](tuning-guide.md)
類型：完整翻譯
來源版本：與英文同一次提交（bfb5446 之後），docs/ml/tuning-guide.md
同步狀態：已核對

說明 [`ml/params.yaml`](../../ml/params.yaml) 每個參數的作用、調整後的變化與要觀察的指標。數值只存在 `params.yaml`；本指南負責解釋，並隨里程碑擴充。

任何改動都會改變 manifest 的 params hash，訓練前須重跑 `make prepare`。

## split

| 參數 | 作用 | 調高 | 調低 | 觀察 |
|---|---|---|---|---|
| `train_fraction` | 最早一段資料的比例，用來 fit 前處理與模型 | 可學的歷史更多；validation 與 test 往後移且變小 | 訓練資料變少；評估資料變多 | 各段詐欺筆數（manifest 的 `split.ranges`）；詐欺太少會讓指標不穩 |
| `valid_fraction` | 用於調參與選門檻的比例 | 調參訊號較穩；test 變小 | 調參較雜訊 | test 必須保留足夠詐欺筆數，PR-AUC 才穩定 |

切點會移到下一個時間戳變化處，所以同一個 `TransactionDT` 不會跨兩段。各段不打亂順序，這正是時間切分的目的。

## preprocess

| 參數 | 作用 | 調高 | 調低 | 觀察 |
|---|---|---|---|---|
| `min_category_count` | 在訓練段出現次數低於此值的類別值視為未知（`-1`） | 類別較少、較可靠；較不會記住稀有值 | 類別較多，含模型可能過度擬合的稀有值 | `preprocessor.json` 中各欄的類別數；validation 與 test 結果的落差 |

## 規劃中

模型超參數（M2）、特徵工程時間窗（M3）、決策門檻與 review 比例（M4）會在對應階段完成時加入。
