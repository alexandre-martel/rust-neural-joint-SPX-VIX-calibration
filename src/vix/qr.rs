use ndarray::{Array1, Array2, s};
use candle_core::Result;
use super::super::nn::network_grad::NetworkGrad;

fn qr_gram_schmidt(a: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let n = a.nrows();
    let m = a.ncols();
    let mut q = Array2::<f64>::zeros((n, m));
    let mut r = Array2::<f64>::zeros((m, m));

    for j in 0..m {
        // Colonne j de A
        let mut v = a.column(j).to_owned();

        for i in 0..j {
            let q_i = q.column(i);
            let proj = q_i.dot(&v);
            r[[i, j]] = proj;
            v = v - proj * &q_i;
        }

        let norm = v.dot(&v).sqrt();
        r[[j, j]] = norm;
        q.column_mut(j).assign(&(v / norm));
    }

    (q, r)
}

pub fn solve_lstsq(a: &Array2<f64>, r: &[f64]) -> (Array1<f64>, Array2<f64>, Array2<f64>) {
    let r_vec = Array1::from_vec(r.to_vec());

    let (q, r_mat) = qr_gram_schmidt(a);
    let qt_r = q.t().dot(&r_vec);

    let m = r_mat.ncols();
    let r_upper = r_mat.slice(s![0..m, 0..m]).to_owned();
    let qt_r_m = qt_r.slice(s![0..m]).to_owned();

    let alpha = back_substitution(&r_upper, &qt_r_m);

    (alpha, q.to_owned(), r_mat)
}

fn back_substitution(r: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let m = r.nrows();
    let mut x = Array1::zeros(m);

    for i in (0..m).rev() {
        let mut sum = 0.0;

        for j in (i + 1)..m {sum += r[[i, j]] * x[j];}
        x[i] = (b[i] - sum) / r[[i, i]];
    }
    x
}


pub fn solve_lstsq_grad(q: &Array2<f64>, r_mat: &Array2<f64>, alpha: &Array1<f64>, a_grad: &[Vec<NetworkGrad>], r_grad: &[NetworkGrad]) -> Result<Vec<NetworkGrad>> {
    let n_paths = q.nrows();   // N
    let n_cols = r_mat.ncols(); // m
    let device = r_grad[0].dw1.device();

    // 1. B_i = ∂R_i/∂θ  -  Σ_j (∂A_{i,j}/∂θ · α*_j)
    let mut b_grads = Vec::with_capacity(n_paths);
    for i in 0..n_paths {
        let mut da_alpha_i = NetworkGrad::zeros(device)?;
        for j in 0..n_cols {
            da_alpha_i = da_alpha_i.add(&a_grad[i][j].scale(alpha[j])?)?;
        }
        // B_i = r_grad[i] - da_alpha_i
        b_grads.push(r_grad[i].add(&da_alpha_i.scale(-1.0)?)?);
    }

    // 2. Y_j = Σ_i Q[i,j] · B_i    (produit Qᵀ · B, chaque Y_j est un NetworkGrad)
    let mut y_grads = Vec::with_capacity(n_cols);
    for j in 0..n_cols {
        let mut y_j = NetworkGrad::zeros(device)?;
        for i in 0..n_paths {
            y_j = y_j.add(&b_grads[i].scale(q[[i, j]])?)?;
        }
        y_grads.push(y_j);
    }

    // 3. Back-substitution : R_mat · ∂α* = Y
    // Pour chaque j de m-1 à 0 :
    // ∂α*_j = (Y_j - Σ_{k>j} R[j,k] · ∂α*_k) / R[j,j]
    let mut d_alpha = (0..n_cols)
        .map(|_| NetworkGrad::zeros(device))
        .collect::<Result<Vec<_>>>()?;

    for j in (0..n_cols).rev() {
        let mut sum = NetworkGrad::zeros(device)?;
        for k in (j + 1)..n_cols {
            sum = sum.add(&d_alpha[k].scale(r_mat[[j, k]])?)?;
        }
        d_alpha[j] = y_grads[j].add(&sum.scale(-1.0)?)?.scale(1.0 / r_mat[[j, j]])?;
    }

    Ok(d_alpha)
}