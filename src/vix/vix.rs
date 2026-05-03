use candle_core::Result;
use ndarray::Array2;
use super::super::nn::network_grad::NetworkGrad;
use super::qr::{solve_lstsq, solve_lstsq_grad};

pub fn build_polynomial_matrix(x: &[f64], y: &[f64], degree: usize) -> Array2<f64> {
    let n = x.len();
    let m = (degree + 1) * (degree + 2) / 2;

    let mut a = Array2::<f64>::zeros((n, m));

    for i in 0..n {
        let xi = x[i];
        let yi = y[i];
        let mut col = 0;

        for k in 0..=degree {
            let x_pow = xi.powi(k as i32);

            for l in 0..=(degree - k) {
                a[[i, col]] = x_pow * yi.powi(l as i32);
                col += 1;
            }
        }
    }
    a
}

pub fn build_polynomial_matrix_grad(x: &[f64], y: &[f64], dx_dtheta: &[NetworkGrad], dy_dtheta: &[NetworkGrad], degree: usize,) -> Result<Vec<Vec<NetworkGrad>>>{
    let n = x.len();
    let m = (degree + 1) * (degree + 2) / 2;
    let device = dx_dtheta[0].dw1.device();
    let mut matrix_grad = Vec::with_capacity(n);

    for i in 0..n {
        let xi = x[i];
        let yi = y[i];
        let dxi = &dx_dtheta[i];
        let dyi = &dy_dtheta[i];
        let mut row = Vec::with_capacity(m);

        for k in 0..=degree {
            for l in 0..=(degree - k) {

                let grad_monome = if k == 0 && l == 0 {
                    NetworkGrad::zeros(device)?
                } else if k == 0 {
                    let coeff = (l as f64) * yi.powi(l as i32 - 1);
                    dyi.scale(coeff)?
                } else if l == 0 {

                    let coeff = (k as f64) * xi.powi(k as i32 - 1);
                    dxi.scale(coeff)?
                } else {

                    let coeff_x = (k as f64) * xi.powi(k as i32 - 1) * yi.powi(l as i32);
                    let coeff_y = (l as f64) * xi.powi(k as i32) * yi.powi(l as i32 - 1);
                    dxi.scale(coeff_x)?.add(&dyi.scale(coeff_y)?)?
                };

                row.push(grad_monome);
            }
        }

        matrix_grad.push(row);
    }

    Ok(matrix_grad)
}


pub fn compute_vix(x: &[f64],y: &[f64],r: &[f64],dx_dtheta: &[NetworkGrad],dy_dtheta: &[NetworkGrad],dr_dtheta: &[NetworkGrad],degree: usize) -> Result<(Vec<f64>, Vec<NetworkGrad>)> {
    let n = x.len();
    let device = dx_dtheta[0].dw1.device();


    let a = build_polynomial_matrix(x, y, degree);
    let a_grad = build_polynomial_matrix_grad(x, y, dx_dtheta, dy_dtheta, degree)?;

    let (alpha, q, r_mat) = solve_lstsq(&a, r);
    let d_alpha = solve_lstsq_grad(&q, &r_mat, &alpha, &a_grad, dr_dtheta)?;

    let n_cols = alpha.len();
    let mut vix_values = Vec::with_capacity(n);
    let mut vix_grads = Vec::with_capacity(n);

    for i in 0..n {

        let vix2_i: f64 = (0..n_cols).map(|j| alpha[j] * a[[i, j]]).sum();
        let vix2_i = vix2_i.max(1e-10); // garde-fou numérique

        let mut d_vix2_i = NetworkGrad::zeros(device)?;

        for j in 0..n_cols {
            let t1 = d_alpha[j].scale(a[[i, j]])?;
            let t2 = a_grad[i][j].scale(alpha[j])?;
            d_vix2_i = d_vix2_i.add(&t1)?.add(&t2)?;
        }

        let vix_i = vix2_i.sqrt();
        let d_vix_i = d_vix2_i.scale(1.0 / (2.0 * vix_i))?;

        vix_values.push(vix_i);
        vix_grads.push(d_vix_i);
    }

    Ok((vix_values, vix_grads))
}