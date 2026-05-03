use rand::Rng;
use candle_core::Result;
use candle_nn::optim::Optimizer;
use super::super::nn::network::Network;
use super::super::nn::network_grad::NetworkGrad;
use super::super::market_data::market_data::MarketData;
use super::super::sde::path::simulate_paths_with_grad;
use super::super::vix::vix::compute_vix;
use super::loss::*;
use super::super::math::black_scholes::implied_vol;
use candle_core::backprop::GradStore;

fn mean_grad(grads: &[NetworkGrad]) -> Result<NetworkGrad> {
    let n = grads.len();
    let device = grads[0].dw1.device();
    let mut acc = NetworkGrad::zeros(device)?;
    for g in grads {
        acc = acc.add(g)?;
    }
    acc.scale(1.0 / n as f64)
}

fn network_grad_to_candle(
    grad: &NetworkGrad,
    varmap: &candle_nn::VarMap,
) -> Result<candle_core::backprop::GradStore> {
    let mut store = GradStore::new();
    let vars = varmap.all_vars();
    // vars[0] = W1, vars[1] = b1, vars[2] = W2, vars[3] = b2
    store.insert(&vars[0], grad.dw1.clone());
    store.insert(&vars[1], grad.db1.clone());
    store.insert(&vars[2], grad.dw2.clone());
    store.insert(&vars[3], grad.db2.clone());
    Ok(store)
}


pub fn train_step(
    network: &mut Network,
    adam: &mut candle_nn::optim::AdamW,
    varmap: &candle_nn::VarMap,
    market: &MarketData,
    n_paths: usize,   
    dt: f64,          
    degree: usize, 
    rng: &mut impl Rng,
) -> Result<f64> {

    let paths = simulate_paths_with_grad(network, n_paths, dt, rng)?;


    let mut vix_futures_model = Vec::new();
    let mut vix_future_grads  = Vec::new();
    let mut vix_calls_model   = Vec::new();
    let mut vix_call_grads_   = Vec::new();
    let mut vix_puts_model    = Vec::new();
    let mut vix_put_grads_    = Vec::new();

    for (j, vix_smile) in market.vix_smiles.iter().enumerate() {
        let (x, y, r, dx, dy, dr) = extract_at_maturity(&paths, j);

        let (vix_vals, vix_grads)=compute_vix(&x, &y, &r, &dx, &dy, &dr, degree)?;

        // Future VIX = mean(VIXi)
        let n = vix_vals.len() as f64;
        let fvix = vix_vals.iter().sum::<f64>() / n;
        vix_futures_model.push(fvix);

        // dfVIX/dθ = mean(dVIXi/dθ)
        let fvix_grad = mean_grad(&vix_grads)?;
        vix_future_grads.push(fvix_grad);

        for (k, &strike) in vix_smile.strikes.iter().enumerate() {
            if strike > vix_smile.future_price {
                vix_calls_model.push(compute_vix_call_price(&vix_vals, strike));
                vix_call_grads_.push(compute_vix_call_grad(&vix_vals, &vix_grads, strike)?);
                vix_puts_model.push(0.0);
                vix_put_grads_.push(NetworkGrad::zeros(vix_grads[0].dw1.device())?);
            } else {
                vix_puts_model.push(compute_vix_put_price(&vix_vals, strike));
                vix_put_grads_.push(compute_vix_put_grad(&vix_vals, &vix_grads, strike)?);
                vix_calls_model.push(0.0);
                vix_call_grads_.push(NetworkGrad::zeros(vix_grads[0].dw1.device())?);
            }
        }
    }


    let mut spx_model_ivs  = Vec::new();
    let mut spx_iv_grads   = Vec::new();

    for (j, spx_smile) in market.spx_smiles.iter().enumerate() {
        let (x_paths, dx_paths) = extract_spx_at_maturity(&paths, j);
        let forward = spx_smile.forward;

        for &strike in &spx_smile.strikes {
   
            let call_price = compute_spx_price(&x_paths, strike, forward);
            let iv = implied_vol(call_price, forward, strike, spx_smile.maturity).unwrap_or(0.3); // fallback si non convergé
            spx_model_ivs.push(iv);

            // dIV/dθ = (dC/dθ) / vega_BS
            let call_grad = compute_spx_price_grad(&x_paths, &dx_paths, strike, forward)?;
            let vega = bs_vega(forward, strike, spx_smile.maturity, iv);
            let iv_grad = call_grad.scale(1.0 / vega.max(1e-10))?;
            spx_iv_grads.push(iv_grad);
        }
    }


    let loss = compute_loss(
        &spx_model_ivs,
        &vix_futures_model,
        &vix_calls_model,
        &vix_puts_model,
        market,
        30.0, 2.0, 3.0,
    );

    let total_grad = compute_loss_grad(
        &spx_iv_grads,
        &vix_future_grads,
        &vix_call_grads_,
        &vix_put_grads_,
        &spx_model_ivs,
        &vix_futures_model,
        &vix_calls_model,
        &vix_puts_model,
        market,
        30.0, 2.0, 3.0,
    )?;

    let grads = network_grad_to_candle(&total_grad, varmap)?;
    adam.step(&grads)?;

    Ok(loss)
}