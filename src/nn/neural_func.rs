use candle_core::{Result, Tensor};


pub fn sde_coefs(raw: &Tensor) -> Result<(f64, f64, f64, f64)> {
    let raw = raw.squeeze(0)?;

    let phi1 = raw.get(0)?.to_scalar::<f32>()? as f64;
    let phi2 = raw.get(1)?.to_scalar::<f32>()? as f64;
    let phi3 = raw.get(2)?.to_scalar::<f32>()? as f64;
    let phi4 = raw.get(3)?.to_scalar::<f32>()? as f64;

    let sigma_x = 1.0 + phi1.tanh();
    let sigma_y = 1.0 + phi2.tanh();
    let mu_y = phi3;
    let rho = phi4.tanh();

    Ok((sigma_x, sigma_y, mu_y, rho))
}