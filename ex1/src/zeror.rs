// ******************************************************************
//  ZeroRule（性能比較用の「ルールなし」モデル）
//
//  2020/9/23 konao
// ******************************************************************

#![allow(non_snake_case)]

use super::U;   // main.rsのコメントを参照

// =================================================
//  ZeroRuleモデル
// =================================================
pub struct ZeroRule {
    r: Vec<f64>
}

impl ZeroRule {
    pub fn new() -> Self {
        return ZeroRule {
            r: Vec::<f64>::new()
        }
    }

    // ===============================================================
    //  モデル作成
    //
    // @param x: 説明変数(2次元配列. 変数(=列)ごとの値)
    // @param y: 目的変数(2次元配列．分類の時は列数>1, 回帰の時は列数=1）
    // ===============================================================
    pub fn fit(&mut self, x: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>) {
        // let ncols = x.len();
        // println!("ncols = {}", ncols);

        // 内容クリア
        self.r.clear();

        // 目的変数列の平均を計算してrにストア
        for z in y.iter() {
            self.r.push(U::mean(z));
        }
    }

    // ============================================================
    // モデルを適用して予測値を計算
    //
    // @param x : 説明変数(2次元配列(mxn). 変数(=列)ごとの値)
    //
    // @return 予測値．m行x1列の行列（mはxの行数）
    // ZeroRuleは予測値の計算にxの値は使わない．
    // fitの際に計算した目的変数の平均値をxの値に無関係に返すだけ．
    // すなわち全要素がself.rのmx1の行列を返す
    // ============================================================
    pub fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut result = vec![];
        let ncols = self.r.len();
        let nrows = x[0].len(); // xの行数
        for icol in 0..ncols {
            let mut resultCol = Vec::<f64>::new();
            for _ in 0..nrows {
                resultCol.push(self.r[icol]);
            }
            result.push(resultCol);
        }
        result
    }

    pub fn print(&self) {
        println!("r={:?}", self.r);
    }
}