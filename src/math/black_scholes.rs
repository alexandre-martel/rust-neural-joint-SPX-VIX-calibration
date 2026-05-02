use special::Error;
use argmin::core::{CostFunction, Error as ArgminError, Executor};
use argmin::solver::neldermead::NelderMead;

pub enum OptionType {
    Call,
    Put,
}

// simulate N(0,1) with erf
fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + (x / std::f64::consts::SQRT_2).erf())
}

// We calculate d1 and d2
fn calculate_d(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> (f64, f64) {
    let d1 = ((s / k).log(std::f64::consts::E) + (r + sigma.powi(2)/2.0)*t)/ (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    (d1, d2)
}


// Call and Put with BS formula
pub fn bs_call(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64{
    let (d1, d2) = calculate_d(s, k, t, r, sigma);
    bs_call = s * standard_normal_cdf(d1) - k* (-r*t).exp() * standard_normal_cdf(d2);
    bs_call
}

pub fn bs_put(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64{
    let (d1, d2) = calculate_d(s, k, t, r, sigma);
    bs_call = k* (-r*t).exp() * standard_normal_cdf(-d2) - s * standard_normal_cdf(-d1)  ;
    bs_call
}

