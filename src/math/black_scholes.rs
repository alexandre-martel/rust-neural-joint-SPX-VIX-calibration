use statrs::distribution::{ContinuousCDF, Normal};

pub enum OptionType {
    Call,
    Put,
}



fn standard_normal_cdf(x: f64) -> f64 {
    let n = Normal::new(0.0, 1.0).unwrap();
    n.cdf(x)
}


fn calculate_d(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> (f64, f64) {
    let d1 = ((s / k).ln() + (r + sigma.powi(2) / 2.0) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    (d1, d2)
}

pub fn bs_call(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    let (d1, d2) = calculate_d(s, k, t, r, sigma);
    s * standard_normal_cdf(d1) - k * (-r * t).exp() * standard_normal_cdf(d2)
}

pub fn bs_put(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    let (d1, d2) = calculate_d(s, k, t, r, sigma);
    k * (-r * t).exp() * standard_normal_cdf(-d2) - s * standard_normal_cdf(-d1)
}

pub fn bs_vega(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return 0.0;
    }
    let (d1, _) = calculate_d(s, k, t, r, sigma);
    s * t.sqrt() * (-0.5 * d1 * d1).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

// Implied vol via Newton-Raphson
pub fn implied_vol(price: f64, s: f64, k: f64, t: f64, r: f64, flag: OptionType, init_sigma: f64) -> f64 {
    let mut sigma = init_sigma;

    for _ in 0..50 {
        let p = match flag {
            OptionType::Call => bs_call(s, k, t, r, sigma),
            OptionType::Put => bs_put(s, k, t, r, sigma),
        };

        let diff = p - price;
        if diff.abs() < 1e-8 {
            return sigma;
        }

        let (d1, _) = calculate_d(s, k, t, r, sigma);
        // Calcul du Vega
        let vega = s * t.sqrt() * (-0.5 * d1.powi(2)).exp() / (2.0 * std::f64::consts::PI).sqrt();
        
        if vega < 1e-10 {
            break;
        }

        sigma -= diff / vega;
    }
    sigma
}