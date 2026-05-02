use super::euler::{euler_step, Path};
use candle_core::Result;
use candle_nn::VarMap;
use rand::{Rng, rng};
use rand_distr::StandardNormal;
use rayon::prelude::*;

use crate::{nn::{network::Network, network_grad::NetworkGrad}, sde::euler::{PathGrad, euler_step_with_grad}};


pub fn simulate_paths(network: &Network, n_paths: usize, n_steps: usize, dt: f64) -> Vec<Path> {
    (0..n_paths).into_par_iter().map(|_| {
        let mut x = vec![0.0; n_steps + 1];
        let mut y = vec![0.0; n_steps + 1];
        let mut r = vec![0.0; n_steps + 1];
        
        let mut rng = rand::rng();
        
        for s in 0..n_steps {
            let dw1: f64 = rng.sample(StandardNormal);
            let dw2: f64 = rng.sample(StandardNormal);
            let t = (s as f64) * dt;
            
            let (xn, yn, rn) = euler_step(network, t, x[s], y[s], r[s], dt, dw1, dw2).unwrap();
            x[s+1] = xn; y[s+1] = yn; r[s+1] = rn;
        }
        Path {x, y, r}
    }).collect()
}

pub fn simulate_paths_with_grad(varmap: &VarMap, network: &Network, n_paths: usize, n_steps: usize, dt: f64) -> Result<Vec<(Path, PathGrad)>> {
    let device = network.layer1.weight().device();

    (0..n_paths).into_par_iter().map(|_| {
        let mut x_vals = vec![0.0; n_steps + 1];
        let mut y_vals = vec![0.0; n_steps + 1];
        let mut r_vals = vec![0.0; n_steps + 1];

        let mut curr_dx = NetworkGrad::zeros(device)?;
        let mut curr_dy = NetworkGrad::zeros(device)?;
        let mut curr_dr = NetworkGrad::zeros(device)?;

        let mut dx_history = Vec::with_capacity(n_steps + 1);
        let mut dy_history = Vec::with_capacity(n_steps + 1);
        let mut dr_history = Vec::with_capacity(n_steps + 1);

        let mut rng = rng();

        for s in 0..n_steps {
            let t = (s as f64) * dt;
            let dw1: f64 = rng.sample(StandardNormal);
            let dw2: f64 = rng.sample(StandardNormal);

            let (next_x, next_y, next_r, next_dx, next_dy, next_dr) = euler_step_with_grad(
                varmap,
                network,
                t, x_vals[s], y_vals[s], r_vals[s],
                &curr_dx, &curr_dy, &curr_dr,
                dt, dw1, dw2
            )?;

            x_vals[s+1] = next_x;
            y_vals[s+1] = next_y;
            r_vals[s+1] = next_r;

            curr_dx = next_dx;
            curr_dy = next_dy;
            curr_dr = next_dr;


            dx_history.push(curr_dx.clone()?); 
            dy_history.push(curr_dy.clone()?);
            dr_history.push(curr_dr.clone()?);
        }

        Ok((
            Path { x: x_vals, y: y_vals, r: r_vals },
            PathGrad { dx: dx_history, dy: dy_history, dr: dr_history }
        ))
    }).collect::<Result<Vec<_>>>()
}