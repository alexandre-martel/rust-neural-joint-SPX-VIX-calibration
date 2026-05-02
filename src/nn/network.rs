use candle_core::{Device, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder, VarMap};

pub struct Network {
    layer1: Linear,  
    layer2: Linear,  
}

impl Network {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let layer1 = linear(3, 16, vb.pp("l1"))?;
        let layer2 = linear(16, 4, vb.pp("l2"))?;
    }
}