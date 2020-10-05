// ************************************************
//  アンサンブル学習アルゴリズム入門 Rust移植版
//
//  2020/09/27
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
mod dtree;

fn zeroRuleTest(x: &U::Matrix, y: &U::Matrix) {
    let mut z = zeror::ZeroRule::new();

    // モデル作成
    z.fit(&x, &y);

    // 予測
    let result = z.predict(&x);

    // 結果表示
    println!("*** ZeroRule output ***");
    println!("{:?}", result);
}

fn linearTest(x: &U::Matrix, y: &U::Matrix) {
    let mut l = linear::Linear::new();

    // モデル作成
    l.fit(&x, &y);

    // 予測
    let result: Vec<f64> = l.predict(&x, false);

    println!("*** Linear output ***");
    println!("{:?}", result);
}

fn decisionTreeTest(x: &U::Matrix, y: &U::Matrix, max_depth: u32) {
    let mut d = dtree::DecisionTree::new(1, max_depth);

    // モデル作成
    d.fit(&x, &y, max_depth);
    d.print();

    // 予測
    let result = d.predict(&x);
    println!("*** DecisionTree output ***");
    println!("{:?}", result);

    // U::stdev()のテスト
    // let v: &Vec<f64> = &x[0];
    // let sd = U::stdev(&v);
    // println!("sd={}", sd);

    // d.test1(&x, &y);
    // d.test_make_split(&x, &y);
    // d.test_split_tree(&x, &y);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len()<2 {
        println!("usage {} csvFile [-d max_depth]", args[0]);
        return;
    }
    let filePath = &args[1];

    let mut max_depth: u32 = 3; // default depth
    if args.len()>2 {
        if &args[2] == "-d" {
            max_depth = args[3].parse::<u32>().unwrap();
        }
    }

    let mut csv = U::CSV::new();
    
    let result = csv.read(filePath); // 読み込み
    if result.is_err() {
        println!("load error {}", filePath);
        return;
    }

    let ncol = csv.cols.len();

    // 説明変数を取り出す
    let x = csv.clonePartial(0, ncol-1).cols;
    // println!("**** x ****");
    // U::printMat(&x);

    // 目的変数を取り出す
    let y = csv.clonePartial(ncol-1, ncol).cols;
    // println!("**** y ****");
    // U::printMat(&y);

    // zeroRuleTest(&x, &y);
    // linearTest(&x, &y);
    decisionTreeTest(&x, &y, max_depth);
}

/******* Sample run *******
> cargo run winequality-red-mid.csv -d 3
*/