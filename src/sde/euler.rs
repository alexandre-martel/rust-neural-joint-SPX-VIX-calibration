use candle_core::{Result, Tensor};
use super::super::nn::network_grad::NetworkGrad;
use super::super::nn::network::Network;

pub struct Path {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub r: Vec<f64>,
}
pub struct PathGrad {
    pub dx: Vec<NetworkGrad>,
    pub dy: Vec<NetworkGrad>,
    pub dr: Vec<NetworkGrad>,
}

pub fn euler_step(network: &Network, t: f64, x: f64, y: f64, r: f64, dt: f64, dw1: f64, dw2: f64) -> Result<(f64, f64, f64)> {

    let device = network.layer1.weight().device();
    let input = Tensor::from_slice(&[t as f32, x as f32, y as f32], (1, 3), device)?;
    let output = network.forward(&input)?;
    let out = output.flatten_all()?.to_vec1::<f32>()?;

    let (sigma_x, sigma_y, mu_y, rho) = (out[0] as f64, out[1] as f64, out[2] as f64, out[3] as f64);
    let sqrt_dt = dt.sqrt();

    let x_new = x - 0.5 * sigma_x.powi(2) * dt + sigma_x * dw1 * sqrt_dt;
    
    let dw_y = (rho * dw1 + (1.0 - rho.powi(2)).sqrt() * dw2) * sqrt_dt;
    let y_new = y + mu_y * dt + sigma_y * dw_y;
    
    let tau = 30.0 / 365.0;
    let r_new = r + (sigma_x.powi(2) * dt) / tau;

    Ok((x_new, y_new, r_new))
}