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
> cargo run [z|l|d] csv-data-file [-d max_depth]    # max_depthのデフォルト値=3
z ... ZeroRule
l ... 線形モデル
d ... 決定木モデル

(ex)
> cargo run l winequality-red-small.csv     # 線形モデル
> cargo run d winequality-red-small.csv     # 決定木（最大深度=デフォルト値(3))
> cargo run d winequality-red-mid.csv -d 4  # 決定木（最大深度=4に指定）
```