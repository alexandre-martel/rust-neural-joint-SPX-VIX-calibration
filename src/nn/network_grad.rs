use candle_core::{Device, Result, Tensor};

pub struct NetworkGrad {
    pub dw1: Tensor,   //[16, 3]
    pub db1: Tensor,   //[16]
    pub dw2: Tensor,   //[4, 16]
    pub db2: Tensor,   //[4]
}

impl NetworkGrad {
    pub fn zeros(device: &Device) -> Result<Self> {
        Ok(Self {
            dw1: Tensor::zeros((16,3), candle_core::DType::F32, device)?,
            db1: Tensor::zeros(16, candle_core::DType::F32, device)?,
            dw2: Tensor::zeros((4,16), candle_core::DType::F32, device)?,
            db2: Tensor::zeros(4, candle_core::DType::F32, device)?,
        })
    }

    pub fn scale(&self, scalar: f64) -> Result<Self> {
        Ok(Self {
            dw1: self.dw1.affine(scalar, 0.0)?,
            db1: self.db1.affine(scalar, 0.0)?,
            dw2: self.dw2.affine(scalar, 0.0)?,
            db2: self.db2.affine(scalar, 0.0)?,
        })
    }

    pub fn add(&self, other: &Self) -> Result<Self> {
        Ok(Self {
            dw1: (&self.dw1 + &other.dw1)?,
            db1: (&self.db1 + &other.db1)?,
            dw2: (&self.dw2 + &other.dw2)?,
            db2: (&self.db2 + &other.db2)?,
        })
    }

    pub fn from_vec(grads: Vec<Tensor>) -> Result<Self> {
        Ok(Self {
            dw1: grads[0].copy()?,
            db1: grads[1].copy()?,
            dw2: grads[2].copy()?,
            db2: grads[3].copy()?,
        })
    }

    pub fn clone(&self) -> Result<Self> {
    Ok(Self {
        dw1: self.dw1.copy()?,
        db1: self.db1.copy()?,
        dw2: self.dw2.copy()?,
        db2: self.db2.copy()?,
    })
}

}

