// ******************************************************************
//  ユーティリティモジュール
//
//  2020/9/23 konao
// ******************************************************************
#![allow(non_snake_case)]

use std::fs::File;
use std::io::{BufRead, BufReader};

// ========================================
//  CVSデータ
// ========================================
pub struct CSV {
    pub ncols: u32, // 列数
    pub nrows: u32, // 行数
    pub cols: Vec<Vec<f64>>,    // 列ベクトルの集合
    pub colNames: Vec<String>   // 列名
}

// カスタムエラー識別子
pub enum CSVError {
    IoErr,
    ParseErr
}

impl CSV {
    // static method（C++/C#などのclass methodに相当）
    pub fn new() -> CSV {
        return CSV {
            ncols: 0,
            nrows: 0,
            cols: vec![],
            colNames: vec![]
        };
    }

    pub fn read(&mut self, fpath: &str) -> Result<(), CSVError> {
        let resFile = File::open(fpath);
        if resFile.is_err() {
            // ファイル読み込み失敗
            return Err(CSVError::IoErr);
        }
        let file = resFile.unwrap();
        let reader = BufReader::new(file);
    
        self.cols = vec![];
        self.colNames = vec![];
    
        let mut nCols = 0;
        let mut nRows = 0;

        let mut bFirstLine: bool = true;
        // std::io::BufReadのlines()イテレータで行ごとに読む
        for (index, line) in reader.lines().enumerate() {
            if line.is_err() {
                return Err(CSVError::IoErr);
            }
            let line = line.unwrap();
            // println!("[{}] {}", index+1, line);
            // let elms = line.split(",");
            let elms = line.split(";");
            if bFirstLine {
                // 1行目（=列名の行）
                bFirstLine = false; // もうここは通らない
                for (_, elm) in elms.enumerate() {
                    self.colNames.push(elm.trim().to_string());
                    // println!("{}", elm.trim());
                }
                nCols = self.colNames.len() as u32;  // 列数
                for _ in 0..nCols {
                    let col: Vec<f64> = vec![];
                    self.cols.push(col); // 空の列を追加
                }
            } else {
                // 2行目以降
                for (ind, elm) in elms.enumerate() {
                    // colNames.push(elm.trim().to_string());
                    // println!("ind={}, elm={}", ind, elm);
                    let resVal = elm.trim().parse::<f64>();
                    if resVal.is_err() {
                        return Err(CSVError::ParseErr);
                    }
                    self.cols[ind].push(resVal.unwrap());
                }
                nRows = nRows + 1;
            }
        }

        self.nrows = nRows;
        self.ncols = nCols;
    
        Ok(())
    }

    // CSV構造体を表示
    pub fn print(&self) {
        let nrows = self.nrows;
        let ncols = self.ncols;
        println!("({}, {})", nrows, ncols);

        for colName in &self.colNames {
            print!("{} ", colName);
        }
        println!();

        for irow in 0..nrows {
            for jcol in 0..ncols {
                let col: &Vec<f64> = &self.cols[jcol as usize];
                let val = col[irow as usize];
                print!("{} ", val);
            }
            println!();
        }
    }

    // このCSVオブジェクトの一部の列からなる別のCSVオブジェクトを生成して返す
    //
    // (ex)
    // let mut csv = utils::CSV::new();
    // csv.read(filePath);
    // let csv2 = csv.clonePartial(3, 5);  // 3列目、4列目をクローンして返す
    pub fn clonePartial(&self, startCol: usize, endCol: usize) -> Self {
        let mut newCols = vec![];
        let mut newColNames = vec![];

        for i in startCol..endCol {
            newCols.push(self.cols[i].clone());
            newColNames.push(self.colNames[i].clone());
        }

        return CSV {
            ncols: (endCol-startCol) as u32,
            nrows: self.nrows,
            cols: newCols,
            colNames: newColNames
        };
    }
}

// =================================================
// ベクトルの平均値を返す
//
// @param v 数値ベクトル
//
// @return vの平均値
// =================================================
pub fn mean(v: &Vec<f64>) -> f64 {
    // vの合計 --> sum
    let sum = v.iter().fold(0.0, |total, e| total+e);

    // 要素数で割って平均を返す
    sum / (v.len() as f64)
}

// =================================================
//  最大、最小型
// =================================================
pub struct MinMax {
    pub min: f64,
    pub max: f64
}

// =================================================
// ベクトルの最大・最小値を返す
//
// @param v 数値ベクトル
//
// @return vの最大値、最小値
// =================================================
pub fn calcMinMax(v: &Vec<f64>) -> MinMax {
    let mut min: f64 = v[0];
    let mut max: f64 = v[0];
    let n = v.len();

    for i in 1..n {
        let e: f64 = v[i];
        if e < min {
            min = e;
        }
        if e > max {
            max = e;
        }
    }

    MinMax {
        min: min,
        max: max
    }
}

// =================================================
//  行列型
// =================================================
// Matrix[jcol][irow]の順で値が入っている
pub type Matrix = Vec<Vec<f64>>;

// =================================================
//  Matrix型の変数の内容を表示
// =================================================
pub fn printMat(m: &Matrix) {
    let ncol = m.len();
    let nrow = m[0].len();

    for irow in 0..nrow {
        for jcol in 0..ncol {
            print!("{}, ", m[jcol][irow]);
        }
        println!("");
    }
}
