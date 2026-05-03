pub struct SpxSmile {
    pub maturity: f64, // in year, ex: 7.0/365.0
    pub forward: f64,
    pub strikes: Vec<f64>, // Strike in absolute value 
    pub implied_vols: Vec<f64>,
    pub inv_spreads: Vec<f64>, // bid ask spread for all strikes
}

pub struct VixSmile {
    pub maturity: f64, 
    pub future_price: f64,
    pub strikes: Vec<f64>,
    pub call_prices: Vec<f64>,
    pub put_prices: Vec<f64>,
    pub inv_spreads: Vec<f64>,
}

pub struct MarketData {
    pub spot: f64,
    pub spx_smiles: Vec<SpxSmile>,
    pub vix_smiles: Vec<VixSmile>,
}

impl MarketData {
    pub fn example() -> Self {
        Self {
            spot: 4400.0,
            spx_smiles: vec![
                SpxSmile {
                    maturity: 7.0 / 365.0,
                    forward: 4401.0,
                    strikes: vec![4200.0, 4400.0, 4600.0],
                    implied_vols: vec![0.22, 0.18, 0.20],
                    inv_spreads: vec![100.0, 150.0, 100.0],
                },
                SpxSmile {
                    maturity: 21.0 / 365.0,
                    forward: 4403.0,
                    strikes: vec![4000.0, 4400.0, 4800.0],
                    implied_vols: vec![0.25, 0.19, 0.21],
                    inv_spreads: vec![80.0, 120.0, 80.0],
                },
            ],
            // Données VIX en unités décimales (0.22 = 22% de vol annualisée)
            // pour être cohérent avec le modèle (sigma_X in (0,2))
            vix_smiles: vec![
                VixSmile {
                    maturity: 7.0 / 365.0,
                    future_price: 0.22,
                    strikes: vec![0.18, 0.22, 0.26, 0.30],
                    call_prices: vec![0.045, 0.020, 0.005, 0.001],
                    put_prices:  vec![0.001, 0.020, 0.045, 0.081],
                    inv_spreads: vec![50.0, 80.0, 80.0, 50.0],
                },
                VixSmile {
                    maturity: 21.0 / 365.0,
                    future_price: 0.24,
                    strikes: vec![0.20, 0.24, 0.28, 0.32],
                    call_prices: vec![0.050, 0.025, 0.008, 0.002],
                    put_prices:  vec![0.002, 0.025, 0.058, 0.092],
                    inv_spreads: vec![50.0, 80.0, 80.0, 50.0],
                },
            ],
        }
    }
}