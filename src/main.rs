use rust_neural_joint_SPX_VIX_calibration::nn::network::Network;
use rust_neural_joint_SPX_VIX_calibration::nn::adam::Adam;
use rust_neural_joint_SPX_VIX_calibration::calibration::train::train_step;
use rust_neural_joint_SPX_VIX_calibration::market_data::market_data::MarketData;
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};

fn main() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let network = Network::new(vb)?;
    let mut adam = Adam::new(&varmap, 0.001)?;

    let market = MarketData::example();


    let n_paths = 512;
    let dt = 0.5 / 365.0;
    let degree = 8;
    let n_steps = 100;


    for step in 0..n_steps {
        match train_step(&network, &mut adam, &market, n_paths, dt, degree) {
            Ok(loss) => println!("Step {:>3}: loss = {:.6}", step + 1, loss),
            Err(e)   => { eprintln!("Erreur at {}: {:?}", step + 1, e); break; }
        }
    }


    Ok(())
}
