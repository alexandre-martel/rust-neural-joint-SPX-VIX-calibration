use candle_core::{Result, Tensor};
use candle_nn::VarMap;
use super::super::nn::neural_func::point_jacobian;
use super::super::nn::network_grad::NetworkGrad;
use super::super::nn::network::Network;

const TAU: f64 = 30.0 / 365.0;
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
    
    let r_new = r + (sigma_x.powi(2) * dt) / TAU;

    Ok((x_new, y_new, r_new))
}

pub fn euler_step_with_grad(varmap: &VarMap, network: &Network, t: f64, x: f64, y: f64, r: f64, 
                            dx_dtheta: &NetworkGrad, 
                            dy_dtheta: &NetworkGrad, 
                            dr_dtheta: &NetworkGrad,
                            dt: f64, dw1: f64, dw2: f64,) 
                            -> Result<(f64, f64, f64, NetworkGrad, NetworkGrad, NetworkGrad)> {

    
    let (sigma_x, sigma_y, mu_y, rho) = network.eval(t, x, y)?;
    
    
    let jac = point_jacobian(varmap, network, t, x, y)?;

    let x_new = x - 0.5 * sigma_x.powi(2) * dt + sigma_x * dw1 * dt.sqrt();
    let y_new = y + mu_y * dt + sigma_y * (rho * dw1 + (1.0 - rho.powi(2)).sqrt() * dw2) * dt.sqrt();
    let r_new = r + sigma_x.powi(2) * dt / TAU;


    let coeff_x = -sigma_x * dt + dw1 * dt.sqrt();
    let d_sigmax = NetworkGrad::from_vec(jac[0].clone())?;
    let dx_new = dx_dtheta.add(&d_sigmax.scale(coeff_x)?)?;


    let coeff_muy = dt;
    let coeff_sigmay = (rho * dw1 + (1.0 - rho.powi(2)).sqrt() * dw2) * dt.sqrt();
    let coeff_rho= sigma_y * (dw1 - rho / (1.0 - rho.powi(2)).sqrt() * dw2) * dt.sqrt();

    let d_muy    = NetworkGrad::from_vec(jac[2].clone())?;
    let d_sigmay = NetworkGrad::from_vec(jac[1].clone())?;
    let d_rho    = NetworkGrad::from_vec(jac[3].clone())?;

    let dy_new = dy_dtheta
        .add(&d_muy.scale(coeff_muy)?)?
        .add(&d_sigmay.scale(coeff_sigmay)?)?
        .add(&d_rho.scale(coeff_rho)?)?;

    let coeff_r  = 2.0 * sigma_x * dt / TAU;
    let d_sigmax2 = NetworkGrad::from_vec(jac[0].clone())?;
    let dr_new = dr_dtheta.add(&d_sigmax2.scale(coeff_r)?)?;

    Ok((x_new, y_new, r_new, dx_new, dy_new, dr_new))
}