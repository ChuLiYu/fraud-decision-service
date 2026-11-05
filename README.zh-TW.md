# Fraud Decision Service（繁中閱讀版）

英文權威來源：[README](README.md)
類型：完整翻譯
來源版本：4e14b3f，README.md
同步狀態：已核對

把 ML 模型帶進正式環境的詐欺決策服務：版本化評分、依政策輸出放行／送審／拒絕、可稽核的決策紀錄，以及安全的模型上線流程。

> **狀態：規劃中。** 目前沒有可執行的程式。本文描述預計範圍，不是已交付的功能。結果實測後才會列在下方。

## 為什麼做這個專案

訓練詐欺模型只是問題的一部分。本專案聚焦在「訓練好的模型」到「客戶實際遇到的決策」之間的系統：

- 模型分數與業務政策分別版本化。
- 每一筆決策都能追溯與重播。
- 模型故障或太慢時，絕不默默放行交易。
- 新模型版本先經 shadow 與 canary 評估再升級，並可回退。

## 預計架構

```text
Transaction request
  -> Java decision service (Spring Boot)
       -> sync HTTP -> Python inference service (active / canary model)
       -> Redis online features
       -> PostgreSQL: decision log + outbox (one transaction)
            -> Debezium CDC -> Kafka `decisions`
                 -> shadow-scoring consumer -> inference service (shadow model)
Offline training and evaluation (Python, LightGBM) -> MLflow model registry
OpenTelemetry -> Prometheus / Grafana
```

決策走同步流程；Kafka 只承載不能增加請求延遲的事件。階段與驗收條件見 [scope](docs/product/scope.md)。

## 資料

使用 Kaggle 的 IEEE-CIS Fraud Detection 資料集，以模擬流量方式重播。原始資料不存放在本 repo。條款、時間語意與限制見 [dataset card](docs/data/dataset-card.md)。

## 邊界

- 只做模擬決策；沒有真實付款、銀行或客戶資料。
- 做詐欺評分，不是 AML 或 KYC 合規。
- 日後公布的效能數字，都是在註明硬體上的實驗室量測。

## 結果

尚無。每項結果都會連到記載硬體、資料量與設定的報告。

## 文件

從 [docs/INDEX.md](docs/INDEX.md) 開始。
