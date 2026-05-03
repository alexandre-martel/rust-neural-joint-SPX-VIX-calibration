use candle_core::Result;
use super::super::nn::network_grad::NetworkGrad;
use super::super::market_data::market_data::MarketData;

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

pub fn compute_loss(
    spx_model_ivs: &[f64],
    vix_futures_model: &[f64],
    vix_calls_model: &[f64],
    vix_puts_model: &[f64],
    market: &MarketData,
    w_fvix: f64,  
    w_spx: f64,  
    w_vix: f64, 
) -> f64 {
    let mut loss = 0.0;
    let nv = market.vix_smiles.len() as f64;
    let ns = market.spx_smiles.len() as f64;

    //  futures VIX 
    for (j, vix_smile) in market.vix_smiles.iter().enumerate() {
        let ratio = vix_futures_model[j] / vix_smile.future_price - 1.0;
        loss += w_fvix / nv * ratio * ratio;
    }

    // SPX option
    let mut spx_idx = 0;
    for spx_smile in &market.spx_smiles {
        let total_delta: f64 = spx_smile.inv_spreads.iter().sum();

        for (k, &iv_mkt) in spx_smile.implied_vols.iter().enumerate() {
            let delta = spx_smile.inv_spreads[k] / total_delta;
            let ratio = spx_model_ivs[spx_idx] / iv_mkt - 1.0;
            loss += w_spx / ns * delta * ratio * ratio;
            spx_idx += 1;
        }
    }

    // VIX options
    let mut vix_idx = 0;
    for vix_smile in &market.vix_smiles {
        let total_delta: f64 = vix_smile.inv_spreads.iter().sum();
        for (k, &strike) in vix_smile.strikes.iter().enumerate() {
            let delta = vix_smile.inv_spreads[k] / total_delta;
            if strike > vix_smile.future_price {
                let ratio = vix_calls_model[vix_idx] / vix_smile.call_prices[k] - 1.0;
                loss += w_vix / nv * delta * ratio * ratio;
            } else {
                let ratio = vix_puts_model[vix_idx] / vix_smile.put_prices[k] - 1.0;
                loss += w_vix / nv * delta * ratio * ratio;
            }
            vix_idx += 1;
        }
    }

    loss
}

pub fn compute_loss_grad(
    spx_iv_grads: &[NetworkGrad],
    vix_future_grads: &[NetworkGrad],
    vix_call_grads: &[NetworkGrad],
    vix_put_grads: &[NetworkGrad],
    spx_model_ivs: &[f64],
    vix_futures_model: &[f64],
    vix_calls_model: &[f64],
    vix_puts_model: &[f64],
    market: &MarketData,
    w_fvix: f64,
    w_spx: f64,
    w_vix: f64,
) -> Result<NetworkGrad> {
    let device = vix_future_grads[0].dw1.device();
    let mut grad = NetworkGrad::zeros(device)?;
    let nv = market.vix_smiles.len() as f64;
    let ns = market.spx_smiles.len() as f64;

    // futures VIX
    for (j, vix_smile) in market.vix_smiles.iter().enumerate() {
        let ratio = vix_futures_model[j] / vix_smile.future_price - 1.0;
        let coeff = w_fvix / nv * 2.0 * ratio / vix_smile.future_price;
        grad = grad.add(&vix_future_grads[j].scale(coeff)?)?;
    }

    // SPX option
    let mut spx_idx = 0;
    for spx_smile in &market.spx_smiles {
        let total_delta: f64 = spx_smile.inv_spreads.iter().sum();
        for (k, &iv_mkt) in spx_smile.implied_vols.iter().enumerate() {
            let delta = spx_smile.inv_spreads[k] / total_delta;
            let ratio = spx_model_ivs[spx_idx] / iv_mkt - 1.0;
            let coeff = w_spx / ns * 2.0 * delta * ratio / iv_mkt;
            grad = grad.add(&spx_iv_grads[spx_idx].scale(coeff)?)?;
            spx_idx += 1;
        }
    }

    // VIX options
    let mut vix_idx = 0;
    for vix_smile in &market.vix_smiles {
        let total_delta: f64 = vix_smile.inv_spreads.iter().sum();
        for (k, &strike) in vix_smile.strikes.iter().enumerate() {
            let delta = vix_smile.inv_spreads[k] / total_delta;
            if strike > vix_smile.future_price {
                let price_mkt = vix_smile.call_prices[k];
                let ratio = vix_calls_model[vix_idx] / price_mkt - 1.0;
                let coeff = w_vix / nv * 2.0 * delta * ratio / price_mkt;
                grad = grad.add(&vix_call_grads[vix_idx].scale(coeff)?)?;
            } else {
                let price_mkt = vix_smile.put_prices[k];
                let ratio = vix_puts_model[vix_idx] / price_mkt - 1.0;
                let coeff = w_vix / nv * 2.0 * delta * ratio / price_mkt;
                grad = grad.add(&vix_put_grads[vix_idx].scale(coeff)?)?;
            }
            vix_idx += 1;
        }
    }

    Ok(grad)
}