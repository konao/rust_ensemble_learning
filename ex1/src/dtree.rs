// ******************************************************************
//  DecisionTree（決定木）
//
//  2020/9/24 konao
// ******************************************************************

#![allow(non_snake_case)]

use super::U;
use super::zeror;
use super::linear;

enum Model {
    Zeror(zeror::ZeroRule),
    Linear(linear::Linear)
}

// メトリック関数の型定義
type Metric = fn(&U::Matrix) -> f64;

// =====================================================
//  gini impurity（ジニ不純物）を計算
//
//  @param y カテゴリ変数を水準ごとの確率値で表した行列
//  (ex) irisの場合：
//  y = array([[1., 0., 0.],
// 		    [1., 0., 0.],
// 		    [1., 0., 0.],
// 		    [1., 0., 0.],
//  			...
// 			[0., 1., 0.],
// 	        [0., 0., 1.]])
//
//  @return ジニ不純物値
// =====================================================
pub fn gini(y: &U::Matrix) -> f64 {
    let size = y[0].len();  // 行数

    // yの各水準の割合 --> e
    let e = y.iter()    // 列でループ
        .map(|col| col.iter().sum())    // 各列の合計を取る
        .map(|x: f64| (x/(size as f64)).powf(2.0))  // 行数で割ると水準の確率になる．さらにそれを2乗
        .sum::<f64>();  // 各列の水準の確率の二乗和
    
    1.0 - e
}

// ==============================================
//  標準偏差を計算
//
//  y ... 数値列(1列)
// ==============================================
pub fn deviation(y: &U::Matrix) -> f64 {
    U::stdev(&y[0])
}

// =================================================
//  決定木モデル
// =================================================
pub struct DecisionTree {
    metric: Metric,
    // leaf: Model,
    left: NodeType,
    right: NodeType,
    feat_index: usize,
    feat_val: f64,
    score: f64,
    depth: u32,
    max_depth: u32
}

enum NodeType {
    Node(Box<DecisionTree>),    // 末端でない（分岐する）所
    Leaf(Box<linear::Linear>)   // ツリーの末端
}

impl DecisionTree {
    pub fn new(depth: u32, max_depth: u32) -> Self {
        let l = linear::Linear::new();
        let r = linear::Linear::new();

        DecisionTree {
            metric: deviation,
            left: NodeType::Leaf(Box::new(l)),
            right: NodeType::Leaf(Box::new(r)),
            feat_index: 0,
            feat_val: f64::NAN,
            score: f64::NAN,
            depth: depth,
            max_depth: max_depth
        }
    }

    pub fn print(&self) {
        self.printSub(0);
    }

    fn printSub(&self, indent: u32) {
        let mut s: String = String::from("");
        for _ in 0..indent {
            s += &String::from("  ");
        }

        println!("{}------------------------", s);
        println!("{}+feat_index: {}", s, self.feat_index);
        println!("{} feat_val: {}", s, self.feat_val);
        println!("{} score: {}", s, self.score);
        println!("{} depth: {}", s, self.depth);

        if let NodeType::Node(ref node) = self.left {
            node.printSub(indent+1);
        }
        
        if let NodeType::Node(ref node) = self.right {
            node.printSub(indent+1);
        }
    }

    // ============================================================
    // ベクトルを基準値より小さいものと、それ以外とに分ける
    //
    // @param feat 数値ベクトル
    // @param val 分割基準値
    //
    // @return (left, right)
    // left = valより小さい値
    // right = val以上の値
    // ============================================================
    pub fn make_split(&self, feat: &Vec<f64>, val: f64) -> (Vec<usize>, Vec<usize>) {
        let mut left = Vec::<usize>::new();
        let mut right = Vec::<usize>::new();

        for (i, v) in feat.iter().enumerate() {
            if *v < val {
                left.push(i);
            } else {
                right.push(i);
            }
        }

        (left, right)
    }

    // ============================================================
    // 損失関数
    //
    // @param self.metric メトリック関数
    // @param y1
    // @param y2
    //
    // @return 損失値
    // ============================================================
    pub fn make_loss(&self, y1: &U::Matrix, y2: &U::Matrix) -> f64 {
        let nrow_y1 = y1[0].len();
        let nrow_y2 = y2[0].len();

        if (nrow_y1 == 0) || (nrow_y2 == 0) {
            return f64::INFINITY;
        }

        let f: Metric = self.metric;

        let nrow_total = nrow_y1 + nrow_y2;
        let m1 = f(y1) * ((nrow_y1 as f64) / (nrow_total as f64));
        let m2 = f(y2) * ((nrow_y2 as f64) / (nrow_total as f64));

        // println!("  y1={:?}", y1);
        // println!("  y2={:?}", y2);
        // println!("  nrow_y1={}, nrow_y2={}", nrow_y1, nrow_y2);
        // println!("  f(y1)={}", f(y1));
        // println!("  f(y2)={}", f(y2));

        m1 + m2
    }

    // ============================================================
    //  データを分割する列を決める．
    //  同時に分割基準値と、分割する際の行番号を求める
    //
    //  @param x 説明変数
    //  @param y 目的変数
    //
    //  @return (left, right)
    //  left .. 左側の枝に入れる行（の行番号）
    //  right .. 右側の枝に入れる行（の行番号）
    //
    //  self.feat_index 分割対象変数（の列番号）
    //  self.feat_val   分割の基準値
    //  self.score      最良の分割点のスコア
    // ============================================================
    pub fn split_tree(&mut self, x: &U::Matrix, y: &U::Matrix) -> (Vec<usize>, Vec<usize>) {
        self.feat_index = 0;
        self.feat_val = f64::INFINITY;

        // xの行数、列数を得る
        let ncol = x.len();
        let nrow = x[0].len();

        // 左右のインデックス
        let mut left: Vec<usize> = (0..nrow).collect();
        let mut right: Vec<usize> = vec![];

        // 以下を求める
        // (1) self.feat_index : 分割対象変数（の列番号）
        // (2) self.feat_val : 分割の基準値
        // (3) left : 左側の枝に入れる行（の行番号）
        // (4) right : 右側の枝に入れる行（の行番号）
        let mut score = f64::INFINITY;
        for i in 0..ncol {  // xの全列についてループ
            let feat: &Vec<f64> = &x[i]; // i列目のベクトル
            for val in feat.iter() {    // featの要素1個ずつについてループ
                let (l, r) = self.make_split(feat, *val);   // featをvalで分割 --> (l, r)
                let y1: U::Matrix = U::MatSelectRow(y, &l); // yのl行目を取り出す --> y1
                let y2: U::Matrix = U::MatSelectRow(y, &r); // yのr行目を取り出す --> y2
                let loss = self.make_loss(&y1, &y2);    // y1, y2から損失値を計算
                // println!("-------------------");
                // println!("i={}, val={}", i, val);
                // println!("l={:?}", l);
                // println!("r={:?}", r);
                // println!("loss={}", loss);
                if loss < score {   // 最小の損失値を探す
                    score = loss;
                    left = l;
                    right = r;
                    self.feat_index = i;
                    self.feat_val = *val;
                }
            }
        }
        self.score = score; // 最良の分割点のスコア

        (left, right)
    }

    // ============================================================
    //  モデルを構築する
    // ============================================================
    pub fn fit(&mut self, x: &U::Matrix, y: &U::Matrix, max_depth: u32) -> &Self {
        // x, yを最も上手く分割する仕方 --> (left, right)
        // left ... 左側の枝に入れる行番号
        // right ... 右側の枝に入れる行番号
        let (left, right) = self.split_tree(&x, &y);

        if self.depth < self.max_depth {
            if left.len() > 0 {
                // まだ最大深度に達していない、かつ左側に分割する行がある --> self.leftを新しいノードに置き換える
                self.left = NodeType::Node(Box::new(DecisionTree::new(self.depth+1, max_depth)));
            }

            if right.len() > 0 {
                // まだ最大深度に達していない、かつ右側に分割する行がある --> self.rightを新しいノードに置き換える
                self.right = NodeType::Node(Box::new(DecisionTree::new(self.depth+1, max_depth)));
            }
        }

        if left.len() > 0 {
            // x, yから左側の枝に入れるデータを取り出す --> xl, yl
            let xl: U::Matrix = U::MatSelectRow(x, &left);
            let yl: U::Matrix = U::MatSelectRow(y, &left);
            match self.left {
                // self.leftがノード --> 再帰的にfit()をコール
                NodeType::Node(ref mut node) => { node.fit(&xl, &yl, max_depth); },
                // self.leftがリーフ（末端） --> 線形モデルでfitさせる
                NodeType::Leaf(ref mut leaf) => { leaf.fit(&xl, &yl); }
            }
        }

        if right.len() > 0 {
            // x, yから右側の枝に入れるデータを取り出す --> xr, yr
            let xr: U::Matrix = U::MatSelectRow(x, &right);
            let yr: U::Matrix = U::MatSelectRow(y, &right);
            match self.right {
                // self.rightがノード --> 再帰的にfit()をコール
                NodeType::Node(ref mut node) => { node.fit(&xr, &yr, max_depth); },
                // self.rightがリーフ（末端） --> 線形モデルでfitさせる
                NodeType::Leaf(ref mut leaf) => { leaf.fit(&xr, &yr); }
            }
        }

        self
    }

    // ============================================================
    //  予測値を計算する
    // ============================================================
    pub fn predict(&self, x: &U::Matrix) -> Vec<f64> {
        // self.feat_index .. このノードでの分割対象列
        // self.feat_val .... 分割値
        let feat: &Vec<f64> = &x[self.feat_index];
        let val: f64 = self.feat_val;
        let (l, r) = self.make_split(feat, val);

        let nrow = x[0].len();
        let mut z: Vec<f64> = vec![0.0; nrow];    // 中身が0.0で長さnrowのベクトル

        if (l.len() > 0) && (r.len() > 0) {
            let xl: U::Matrix = U::MatSelectRow(x, &l);
            let left = match self.left {
                // self.leftはノード --> 再帰的にpredict()をコール
                NodeType::Node(ref node) => { node.predict(&xl) },
                // self.leftはリーフ（末端） --> 線形モデルで予測
                NodeType::Leaf(ref leaf) => { leaf.predict(&xl, false) }
            };

            let xr: U::Matrix = U::MatSelectRow(x, &r);
            let right = match self.right {
                // self.rightはノード --> 再帰的にpredict()をコール
                NodeType::Node(ref node) => { node.predict(&xr) },
                // self.rightはリーフ（末端） --> 線形モデルで予測
                NodeType::Leaf(ref leaf) => { leaf.predict(&xr, false) }
            };

            for i in 0..l.len() {
                z[l[i]] = left[i];
            }
            for i in 0..r.len() {
                z[r[i]] = right[i];
            }
        } else if l.len() > 0 {
            z = match self.left {
                NodeType::Node(ref node) => { node.predict(&x) },
                NodeType::Leaf(ref leaf) => { leaf.predict(&x, false) }
            }
        } else if r.len() > 0 {
            z = match self.right {
                NodeType::Node(ref node) => { node.predict(&x) },
                NodeType::Leaf(ref leaf) => { leaf.predict(&x, false) }
            }
        }

        z
    }

    pub fn test_make_split(&self, x: &U::Matrix, y: &U::Matrix) {
        let feat: &Vec<f64> = &x[0];
        let nrow = feat.len();
        for irow in 0..nrow {
            let val = feat[irow];
            let (left, right) = self.make_split(feat, val);
            println!("-----------------------------------");
            println!("[{}] l={:?}", irow, left);
            println!("[{}] r={:?}", irow, right);
        }
    }

    pub fn test_split_tree(&mut self, x: &U::Matrix, y: &U::Matrix) {
        let (left, right) = self.split_tree(&x, &y);
        println!("-----------------------------------");
        println!("l={:?}", left);
        println!("r={:?}", right);
        println!("feat_index={}", self.feat_index);
        println!("feat_val={}", self.feat_val);
        println!("score={}", self.score);
    }

    pub fn test1(&self, x: &U::Matrix, y: &U::Matrix) {
        let z = zeror::ZeroRule::new();
        let mut m1: Model = Model::Zeror(z);
        if let Model::Zeror(ref mut m) = m1 {
            m.fit(&x, &y);

        }

        let l = linear::Linear::new();
        let mut m2: Model = Model::Linear(l);
        if let Model::Linear(ref mut l) = m2 {
            l.fit(&x, &y);

            let result: Vec<f64> = l.predict(&x, false);

            println!("*** Linear output (2) ***");
            println!("{:?}", result);
        }
    }
}