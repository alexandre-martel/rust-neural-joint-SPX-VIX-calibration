use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use candle_core::{Result};
use candle_nn::VarMap;

pub fn build_optimizer(varmap: &VarMap) -> Result<AdamW> {
    AdamW::new(varmap.all_vars(), ParamsAdamW {
        lr: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
    })
}