// ************************************************
//  アンサンブル学習アルゴリズム入門 Rust移植版
//
//  2020/09/19
// ************************************************
// suppress for the whole module with inner attribute...
#![allow(non_snake_case)]

use std::env;

// 他のファイルで定義されている関数などを参照する方法：
//
// (1) main.rsから参照する場合
// mod XXXと書くと、XXX.rs内のpub指定されているもの（関数、構造体、他）が参照できる．
// 
// (2) main.rs以外から参照する場合
// use super::XXXと書く（例：linear.rsを見よ）

mod U;  // 「main.rsからU.rs内のpub要素を使う」の意味
mod zeror;
mod linear;

fn zeroRuleTest(filePath: &String) {
    let mut csv = U::CSV::new();

    let result = csv.read(filePath); // 読み込み
    if result.is_err() {
        println!("load error {}", filePath);
        return;
    }

    let mut z = zeror::ZeroRule::new();

    let ncol = csv.cols.len();

    // 説明変数を取り出す
    let x = csv.clonePartial(0, ncol-1);
    // println!("**** x ****");
    // U::printMat(&x.cols);

    // 目的変数を取り出す
    let y = csv.clonePartial(ncol-1, ncol);
    // println!("**** y ****");
    // U::printMat(&y.cols);

    // モデル作成
    z.fit(&x.cols, &y.cols);

    // 予測
    let result = z.predict(&x.cols);

    // 結果表示
    println!("{:?}", result);
}

fn linearTest(filePath: &String) {
    let mut csv = U::CSV::new();

    let result = csv.read(filePath); // 読み込み
    if result.is_err() {
        println!("load error {}", filePath);
        return;
    }

    let ncol = csv.cols.len();

    // 説明変数を取り出す
    let x = csv.clonePartial(0, ncol-1);

    // 目的変数を取り出す
    let y = csv.clonePartial(ncol-1, ncol);

    let mut l = linear::Linear::new();

    // モデル作成
    l.fit(&x.cols, &y.cols);

    // 予測
    let result: Vec<f64> = l.predict(&x.cols, false);

    println!("{:?}", result);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len()<2 {
        println!("usage {} csvFile", args[0]);
        return;
    }
    let filePath = &args[1];

    // zeroRuleTest(&filePath);
    linearTest(filePath);
}

/******* Sample run *******
> cargo run winequality-red-small.csv
*/