// ******************************************************************
//  Linear（線形モデル）
//
//  2020/9/23 konao
// ******************************************************************

#![allow(non_snake_case)]

use super::U;   // main.rsのコメントを参照

// =================================================
//  線形モデル
// =================================================
pub struct Linear {
    // r: Vec<f64>,
    epochs: u32,
    lr: f64,

    // 回帰式の係数．beta[0]=切片、beta[1..]=係数
    beta: Vec<f64>,

    // 正規化計算用
    // norm[0] = 目的変数の最大、最小
    // norm[1..] = 説明変数の最大、最小
    // 説明変数の次元数がnのとき、norm.len() = n+1になる
    norm: Vec<U::MinMax>
}

impl Linear {
    pub fn new() -> Self {
        return Linear {
            epochs: 20,
            lr: 0.01,
            beta: vec![],
            norm: vec![]
        };
    }

    pub fn print(&self) {
        println!("epochs = {}", self.epochs);
        println!("lr = {}", self.lr);
        println!("beta = {:?}", self.beta);
        for e in &self.norm {
            println!("norm(min, max)=({}, {})", e.min, e.max);
        }
    }

    // ==========================================================
    //  説明変数、目的変数の最大・最小を計算してself.normに格納
    //  後で正規化の際に使う
    //
    //  @param x: 説明変数(2次元配列. 変数(=列)ごとの値)
    //  @param y: 目的変数(2次元配列．線形モデルでは回帰しか行えないので列数は常に1)
    // ==========================================================
    pub fn fitnorm(&mut self, x: &U::Matrix, y: &U::Matrix) {
        self.norm = Vec::<U::MinMax>::new();

        // 目的変数の最大・最小
        self.norm.push(U::calcMinMax(&y[0]));

        // 説明変数の次元数--> n
        let n = x.len();

        // 説明変数の最大・最小
        for i in 0..n {
            self.norm.push(U::calcMinMax(&x[i]));
        }
    }

    // ==================================================================
    //  説明変数x, 目的変数yを正規化し、新しい説明変数、目的変数として返す
    //
    //  @param x: 説明変数(2次元配列. 変数(=列)ごとの値)
    //  @param y: 目的変数(2次元配列．線形モデルでは回帰しか行えないので列数は常に1)
    //
    //  @return 正規化後の説明変数、目的変数
    // ==================================================================
    pub fn normalize(&self, x: &U::Matrix, y: &U::Matrix) -> (U::Matrix, U::Matrix) {
        // 変数の値の範囲 --> ranges
        // ranges[0] .. 目的変数の範囲
        // ranges[1..] .. 説明変数の範囲
        let mut ranges = Vec::<f64>::new();

        for e in &self.norm {
            let mut range: f64 = e.max - e.min;
            if range == 0.0 {
                range = 1.0;
            }
            ranges.push(range);
        }

        // println!("ranges.len() = {}", ranges.len());
        // println!("nrow(x) = {}", x[0].len());

        // 説明変数を正規化
        let mut expVars = U::Matrix::new();

        let nExpVars = x.len();
        for jcol in 0..nExpVars {
            let xx = &x[jcol];
            let mut expVar = Vec::<f64>::new();
            let nrow = xx.len();    // 説明変数xxの行数 --> nrow
            for i in 0..nrow {
                expVar.push((xx[i] - self.norm[jcol+1].min) / ranges[jcol+1]);   // self.normは最初に目的変数が入っているので+1
            }
            expVars.push(expVar);
        }

        // 目的変数を正規化
        let mut objVars = U::Matrix::new();
        let mut objVar = Vec::<f64>::new();
        if y.len() > 0 {
            let nrow = y[0].len();
            for i in 0..nrow {
                objVar.push((y[0][i] - self.norm[0].min) / ranges[0]);
            }
        }
        objVars.push(objVar);

        (expVars, objVars)
    }

    // ============================================
    //  モデル作成
    //
    //  @param x: 説明変数(2次元配列. 変数(=列)ごとの値)
    //  @param y: 目的変数(2次元配列．線形モデルでは回帰しか行えないので列数は常に1)
    // ============================================
    pub fn fit(&mut self, x: &U::Matrix, y: &U::Matrix) {
        // 最初にデータに含まれる値の範囲を0以上1以下に正規化する
        self.fitnorm(x, y);

        // 正規化
        let (nx, ny) = self.normalize(x, y);

        // println!("**** nx ****");
        // U::printMat(&nx);

        // println!("**** ny ****");
        // U::printMat(&ny);
        
        let nExpVars = x.len(); // 説明変数の次元数
        let nrows = x[0].len(); // xの行数 = 観測値の個数

        self.beta = vec![0.0; nExpVars+1];    // 初期化．長さnExpVars+1の0ベクトル

        for _ in 0..self.epochs {
            for irow in 0..nrows {
                // p=説明変数を保持する行列(1行xn列)．predict()の引数の型と合わせるため行列にする
                let mut p: U::Matrix = U::Matrix::new();
                for jcol in 0..nExpVars {
                    let expVar: Vec<f64> = vec![nx[jcol][irow]]; // 要素1個だけのベクトル
                    p.push(expVar);
                }

                // q=目的変数
                let q: f64 = ny[0][irow];

                // 予測値計算 --> z
                let z: Vec<f64> = self.predict(&p, true);   // true=正規化計算を行う

                // 誤差率
                let err: f64 = (z[0] - q) * self.lr;

                // モデルを更新
                self.beta[0] -= err;    // 切片
                for jcol in 0..nExpVars {
                    let delta: f64 = nx[jcol][irow] * err;
                    self.beta[jcol+1] -= delta; // 係数
                }
            }
        }

        // debug
        // println!("self.beta={:?}", self.beta);  // 回帰係数
    }

    // ============================================================
    //  モデルを適用して予測値を計算
    // 
    //  @param x .. 説明変数のベクトル(m行xn列の行列) nは説明変数の次元数
    //  m=1 (predictからコールされた時)
    //  m>1（それ以外の時）
    //
    //  @normalized .. 説明変数xが正規化されていればtrue. そうでなければfalse
    //
    //  返り値は回帰式にxを当てはめた値(=予測値)
    // ============================================================
    pub fn predict(&self, x: &U::Matrix, normalized: bool) -> Vec<f64> {
        let nx: U::Matrix;
        if !normalized {
            // 正規化されていないので、正規化する
            let dummy = U::Matrix::new();
            nx = self.normalize(x, &dummy).0;
        } else {
            // 正規化の必要なし
            nx = x.clone();
        }

        let n = x.len(); // 説明変数の次元数
        let m = x[0].len(); // 説明変数の行数

        // 目的変数の最小値と値の範囲
        let objMin: f64 = self.norm[0].min;
        let objRange: f64 = self.norm[0].max - self.norm[0].min;

        let mut zs: Vec<f64> = vec![0.0; m];    // 予測値

        for irow in 0..m {
            // 回帰式に説明変数を当てはめて予測値を計算 ---> z
            let mut z: f64 = self.beta[0];
            for i in 0..n {
                z += nx[i][irow] * self.beta[i+1];
            }

            if !normalized {
                // 元のスケールに戻す
                z = z * objRange + objMin;
            }

            zs[irow] = z;
        }

        zs
    }
}