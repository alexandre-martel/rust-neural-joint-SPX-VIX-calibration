use candle_core::Result;
use super::super::nn::network_grad::NetworkGrad;
use crate::market_data::{MarketData, SpxSmile, VixSmile};
use super::super::math::black_scholes::implied_vol;

// Price call SPX = mean((St - K)+)
pub fn compute_spx_price(x_paths: &[f64], strike: f64, forward: f64) -> f64 {
    let n = x_paths.len() as f64;

    x_paths.iter().map(|&xi| (forward * xi.exp() - strike).max(0.0)).sum::<f64>() / n
}

pub fn compute_spx_price_grad(x_paths: &[f64],dx_dtheta: &[NetworkGrad],strike: f64,forward: f64) -> Result<NetworkGrad> {
    let n = x_paths.len();
    let device = dx_dtheta[0].dw1.device();
    let mut grad = NetworkGrad::zeros(device)?;
    let mut _count = 0usize;

    for i in 0..n {
        let s_i = forward * x_paths[i].exp();

        if s_i >= strike {
            grad = grad.add(&dx_dtheta[i].scale(s_i)?)?;
            _count += 1;
        }
    }

    grad = grad.scale(1.0 / n as f64)?;
    Ok(grad)
}

// Price call VIX = mean((VIXi - K)+)
pub fn compute_vix_call_price(vix_paths: &[f64], strike: f64) -> f64 {
    let n = vix_paths.len() as f64;
    vix_paths.iter().map(|&v| (v - strike).max(0.0)).sum::<f64>() / n
}

pub fn compute_vix_call_grad(vix_paths: &[f64],vix_grads: &[NetworkGrad],strike: f64) -> Result<NetworkGrad> {
    let n = vix_paths.len();
    let device = vix_grads[0].dw1.device();
    let mut grad = NetworkGrad::zeros(device)?;

    for i in 0..n {
        if vix_paths[i] >= strike {
            grad = grad.add(&vix_grads[i])?;
        }
    }

    grad = grad.scale(1.0 / n as f64)?;
    Ok(grad)
}

// COmpute put VIX price = mean((K - VIXi)+)
pub fn compute_vix_put_price(vix_paths: &[f64],strike: f64) -> f64 {
    let n = vix_paths.len() as f64;
    vix_paths.iter().map(|&v| (strike - v).max(0.0)).sum::<f64>() / n
}


pub fn compute_vix_put_grad(vix_paths: &[f64],vix_grads: &[NetworkGrad],strike: f64,) -> Result<NetworkGrad> {
    let n = vix_paths.len();
    let device = vix_grads[0].dw1.device();
    let mut grad = NetworkGrad::zeros(device)?;

    for i in 0..n {
        if vix_paths[i] <= strike {

            grad = grad.add(&vix_grads[i].scale(-1.0)?)?;
        }
    }

    grad = grad.scale(1.0 / n as f64)?;
    Ok(grad)
}