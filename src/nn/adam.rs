use candle_core::{DType, Result, Tensor, Var};
use candle_nn::VarMap;
use super::network_grad::NetworkGrad;

pub struct Adam {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    // Fixed order: l1.weight[16,3], l1.bias[16], l2.weight[4,16], l2.bias[4]
    // Matches NetworkGrad fields: dw1, db1, dw2, db2
    vars: [Var; 4],
    m: [Tensor; 4],
    v: [Tensor; 4],
    t: i32,
}

impl Adam {
    pub fn new(varmap: &VarMap, lr: f64) -> Result<Self> {
        let data = varmap.data().lock().unwrap();

        let names = ["l1.weight", "l1.bias", "l2.weight", "l2.bias"];
        let vars: Vec<Var> = names.iter().map(|&name| {
            data.get(name)
                .unwrap_or_else(|| panic!("var '{}' not found in varmap", name))
                .clone()
        }).collect();
        drop(data);

        let device = vars[0].device();
        let m = [
            Tensor::zeros(vars[0].shape(), DType::F32, device)?,
            Tensor::zeros(vars[1].shape(), DType::F32, device)?,
            Tensor::zeros(vars[2].shape(), DType::F32, device)?,
            Tensor::zeros(vars[3].shape(), DType::F32, device)?,
        ];
        let v = [
            Tensor::zeros(vars[0].shape(), DType::F32, device)?,
            Tensor::zeros(vars[1].shape(), DType::F32, device)?,
            Tensor::zeros(vars[2].shape(), DType::F32, device)?,
            Tensor::zeros(vars[3].shape(), DType::F32, device)?,
        ];
        let vars = [vars[0].clone(), vars[1].clone(), vars[2].clone(), vars[3].clone()];
        Ok(Self { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, vars, m, v, t: 0 })
    }

    pub fn step(&mut self, grad: &NetworkGrad) -> Result<()> {
        self.t += 1;
        // NetworkGrad order: dw1=l1.weight, db1=l1.bias, dw2=l2.weight, db2=l2.bias
        let grads = [&grad.dw1, &grad.db1, &grad.dw2, &grad.db2];

        let bc1 = 1.0 - self.beta1.powi(self.t);
        let bc2 = 1.0 - self.beta2.powi(self.t);

        for i in 0..4 {
            let g = grads[i];

            let new_m = (&self.m[i].affine(self.beta1, 0.0)? + &g.affine(1.0 - self.beta1, 0.0)?)?;
            let g_sq = g.mul(g)?;
            let new_v = (&self.v[i].affine(self.beta2, 0.0)? + &g_sq.affine(1.0 - self.beta2, 0.0)?)?;

            self.m[i] = new_m;
            self.v[i] = new_v;

            let m_hat = self.m[i].affine(1.0 / bc1, 0.0)?;
            let v_hat = self.v[i].affine(1.0 / bc2, 0.0)?;

            let denom = v_hat.sqrt()?.affine(1.0, self.eps)?;
            let update = m_hat.div(&denom)?.affine(self.lr, 0.0)?;

            let new_val = (&*self.vars[i] - &update)?;
            self.vars[i].set(&new_val)?;
        }
        Ok(())
    }
}
