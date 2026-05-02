use candle_core::{Device, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use super::neural_func::sde_coefs;

pub struct Network {
    pub layer1: Linear,  
    pub layer2: Linear,  
}

impl Network {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let layer1 = linear(3, 16, vb.pp("l1"))?;
        let layer2 = linear(16, 4, vb.pp("l2"))?;

        Ok(Self { layer1, layer2 })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let xs = self.layer1.forward(input)?;
        let xs = xs.tanh()?;
        self.layer2.forward(&xs)
    }

    pub fn eval(&self, t: f64, x: f64, y: f64) -> Result<(f64, f64, f64, f64)> {
        let input = Tensor::new(&[[t as f32, x as f32, y as f32]], &Device::Cpu)?;
        let raw = self.forward(&input)?;
        sde_coefs(&raw)
    }
}


