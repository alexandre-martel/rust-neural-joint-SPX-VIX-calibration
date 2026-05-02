use candle_core::{Device, Result, Tensor};
use candle_nn::VarMap;
use super::network::Network;

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

pub fn point_jacobian(varmap: &VarMap, network: &Network, t: f64, x: f64, y: f64) -> Result<[Vec<Tensor>; 4]> {
    let all_vars = varmap.all_vars();
    let device = all_vars.first().map(|v| v.device()).unwrap_or(&Device::Cpu);

    let mut jacobian: [Vec<Tensor>; 4] = [vec![], vec![], vec![], vec![]];

    for i in 0..4 {

        let input = Tensor::from_slice(&[t as f32, x as f32, y as f32], (1, 3), device)?;
        let output = network.forward(&input)?;
        let out_i = output.get(0)?.get(i)?;

        let grad_store = out_i.backward()?;

        let mut grads = Vec::with_capacity(all_vars.len());
        for var in all_vars.iter() {

            if let Some(grad) = grad_store.get(var.as_tensor()) {
                grads.push(grad.copy()?);
            }
        }
        jacobian[i] = grads;
    }

    Ok(jacobian)
}