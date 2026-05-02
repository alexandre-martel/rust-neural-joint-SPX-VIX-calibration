use candle_core::{Device, Result, Tensor};

pub struct NetworkGrad {
    pub dw1: Tensor,   //[16, 3]
    pub db1: Tensor,   //[16]
    pub dw2: Tensor,   //[4, 16]
    pub db2: Tensor,   //[4]
}

impl NetworkGrad {
    pub fn zeros(device: &Device) -> Result<Self> {
        
    }
}

