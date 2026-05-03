use candle_core::Result;
use super::super::nn::network::Network;
use super::super::nn::network_grad::NetworkGrad;
use super::super::nn::adam::Adam;
use super::super::market_data::market_data::MarketData;
use super::super::sde::path::simulate_paths_with_grad;
use super::super::sde::euler::{Path, PathGrad};
use super::super::vix::vix::compute_vix;
use super::loss::*;
use super::super::math::black_scholes::{bs_vega, implied_vol, OptionType};

const TAU: f64 = 30.0 / 365.0;

fn extract_at_maturity(
    paths: &[(Path, PathGrad)],
    maturity: f64,
    dt: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<NetworkGrad>, Vec<NetworkGrad>, Vec<NetworkGrad>)> {
    let step_j = (maturity / dt).round() as usize;
    let tau_steps = (TAU / dt).round() as usize;
    let device = paths[0].1.dx[0].dw1.device();

    let n = paths.len();
    let mut x_out = Vec::with_capacity(n);
    let mut y_out = Vec::with_capacity(n);
    let mut r_out = Vec::with_capacity(n);
    let mut dx_out = Vec::with_capacity(n);
    let mut dy_out = Vec::with_capacity(n);
    let mut dr_out = Vec::with_capacity(n);

    for (path, grad) in paths {
        x_out.push(path.x[step_j]);
        y_out.push(path.y[step_j]);
        // R_t = integral de t à t+tau de sigma_X^2 / tau  (accumulé dans r)
        r_out.push(path.r[step_j + tau_steps] - path.r[step_j]);

        // Gradient de X à step_j: dx_history[step_j-1] correspond à x[step_j]
        let dx = if step_j > 0 {
            grad.dx[step_j - 1].clone()?
        } else {
            NetworkGrad::zeros(device)?
        };
        let dy = if step_j > 0 {
            grad.dy[step_j - 1].clone()?
        } else {
            NetworkGrad::zeros(device)?
        };

        // dR = dr[step_j + tau_steps] - dr[step_j]
        let dr_end = grad.dr[step_j + tau_steps - 1].clone()?;
        let dr_start = if step_j > 0 {
            grad.dr[step_j - 1].clone()?
        } else {
            NetworkGrad::zeros(device)?
        };
        let dr = dr_end.add(&dr_start.scale(-1.0)?)?;

        dx_out.push(dx);
        dy_out.push(dy);
        dr_out.push(dr);
    }

    Ok((x_out, y_out, r_out, dx_out, dy_out, dr_out))
}

// Extract (x, dx) for all paths at the matirity
fn extract_spx_at_maturity(
    paths: &[(Path, PathGrad)],
    maturity: f64,
    dt: f64,
) -> Result<(Vec<f64>, Vec<NetworkGrad>)> {
    let step_j = (maturity / dt).round() as usize;
    let device = paths[0].1.dx[0].dw1.device();

    let n = paths.len();
    let mut x_out = Vec::with_capacity(n);
    let mut dx_out = Vec::with_capacity(n);

    for (path, grad) in paths {
        x_out.push(path.x[step_j]);
        let dx = if step_j > 0 {
            grad.dx[step_j - 1].clone()?
        } else {
            NetworkGrad::zeros(device)?
        };
        dx_out.push(dx);
    }

    Ok((x_out, dx_out))
}

fn mean_grad(grads: &[NetworkGrad]) -> Result<NetworkGrad> {
    let n = grads.len();
    let device = grads[0].dw1.device();
    let mut acc = NetworkGrad::zeros(device)?;
    for g in grads {
        acc = acc.add(g)?;
    }
    acc.scale(1.0 / n as f64)
}

pub fn train_step(
    network: &Network,
    adam: &mut Adam,
    market: &MarketData,
    n_paths: usize,
    dt: f64,
    degree: usize,
) -> Result<f64> {
    // Nombre de pas nécessaires: couvrir max(maturité VIX + tau, maturité SPX)
    let tau_steps = (TAU / dt).round() as usize;
    let max_vix_mat = market.vix_smiles.iter().map(|s| s.maturity).fold(0.0_f64, f64::max);
    let max_spx_mat = market.spx_smiles.iter().map(|s| s.maturity).fold(0.0_f64, f64::max);
    let max_t = (max_vix_mat + TAU).max(max_spx_mat);
    // +tau_steps pour s'assurer que r[step + tau_steps] est valide
    let n_steps = (max_t / dt).ceil() as usize + tau_steps;

    let paths = simulate_paths_with_grad(network, n_paths, n_steps, dt)?;

    // --- VIX futures et options ---
    let mut vix_futures_model = Vec::new();
    let mut vix_future_grads  = Vec::new();
    let mut vix_calls_model   = Vec::new();
    let mut vix_call_grads    = Vec::new();
    let mut vix_puts_model    = Vec::new();
    let mut vix_put_grads     = Vec::new();

    for vix_smile in &market.vix_smiles {
        let (x, y, r, dx, dy, dr) =
            extract_at_maturity(&paths, vix_smile.maturity, dt)?;

        let (vix_vals, vix_grads) = compute_vix(&x, &y, &r, &dx, &dy, &dr, degree)?;

        let n = vix_vals.len() as f64;
        let fvix = vix_vals.iter().sum::<f64>() / n;
        vix_futures_model.push(fvix);
        vix_future_grads.push(mean_grad(&vix_grads)?);

        let zero_grad = NetworkGrad::zeros(vix_grads[0].dw1.device())?;
        for &strike in &vix_smile.strikes {
            if strike > vix_smile.future_price {
                vix_calls_model.push(compute_vix_call_price(&vix_vals, strike));
                vix_call_grads.push(compute_vix_call_grad(&vix_vals, &vix_grads, strike)?);
                vix_puts_model.push(0.0);
                vix_put_grads.push(zero_grad.clone()?);
            } else {
                vix_puts_model.push(compute_vix_put_price(&vix_vals, strike));
                vix_put_grads.push(compute_vix_put_grad(&vix_vals, &vix_grads, strike)?);
                vix_calls_model.push(0.0);
                vix_call_grads.push(zero_grad.clone()?);
            }
        }
    }

    // --- SPX implied vols ---
    let mut spx_model_ivs = Vec::new();
    let mut spx_iv_grads  = Vec::new();

    for spx_smile in &market.spx_smiles {
        let (x_paths, dx_paths) =
            extract_spx_at_maturity(&paths, spx_smile.maturity, dt)?;
        let forward = spx_smile.forward;

        for &strike in &spx_smile.strikes {
            let call_price = compute_spx_price(&x_paths, strike, forward);
            let iv = implied_vol(
                call_price, forward, strike,
                spx_smile.maturity, 0.0, OptionType::Call, 0.3,
            );
            spx_model_ivs.push(iv);

            // dIV/dθ = dC/dθ / vega_BS
            let call_grad = compute_spx_price_grad(&x_paths, &dx_paths, strike, forward)?;
            let vega = bs_vega(forward, strike, spx_smile.maturity, 0.0, iv);
            let iv_grad = call_grad.scale(1.0 / vega.max(1e-10))?;
            spx_iv_grads.push(iv_grad);
        }
    }

    // --- Loss et gradient ---
    let loss = compute_loss(
        &spx_model_ivs, &vix_futures_model, &vix_calls_model, &vix_puts_model,
        market, 30.0, 2.0, 3.0,
    );

    let total_grad = compute_loss_grad(
        &spx_iv_grads, &vix_future_grads, &vix_call_grads, &vix_put_grads,
        &spx_model_ivs, &vix_futures_model, &vix_calls_model, &vix_puts_model,
        market, 30.0, 2.0, 3.0,
    )?;

    adam.step(&total_grad)?;

    Ok(loss)
}
