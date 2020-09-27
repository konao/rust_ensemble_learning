# rust_ensemble_learning

「アンサンブル学習アルゴリズム入門」をRustに移植してみたもの

＜実装したアルゴリズム＞

+ ZeroRule
+ 線形回帰 (Linear Regression)
+ 決定木 (Decision Tree)

## How to build

```
> cd rust_ensemble_learning/ex1
> cargo build
```

## How to run

```
> cargo run csv-data-file [-d max_depth]    # max_depthのデフォルト値=3

(ex)
> cargo run winequality-red-small.csv
> cargo run winequality-red-mid.csv -d 4
```