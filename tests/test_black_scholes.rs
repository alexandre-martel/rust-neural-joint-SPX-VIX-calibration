use rust_neural_joint_SPX_VIX_calibration::math::black_scholes::{bs_call, bs_put, implied_vol, OptionType};

#[test]
fn test_bs_call_price() {

    let s = 100.0;
    let k = 100.0;
    let t = 1.0;
    let r = 0.05;
    let sigma = 0.2;

    let price = bs_call(s, k, t, r, sigma);
    assert!((price - 10.45058).abs() < 1e-4);
}

#[test]
fn test_bs_put_price() {
    let s = 100.0;
    let k = 100.0;
    let t = 1.0;
    let r = 0.05;
    let sigma = 0.2;

    let price = bs_put(s, k, t, r, sigma);
    assert!((price - 5.57352).abs() < 1e-4);
}

#[test]
fn test_implied_vol_convergence() {
    let s = 100.0;
    let k = 100.0;
    let t = 1.0;
    let r = 0.05;
    let market_price = 10.45058; 

    let iv = implied_vol(market_price, s, k, t, r, OptionType::Call, 0.5);
    assert!((iv - 0.2).abs() < 1e-4);
}

#[test]
fn test_implied_vol_convergence_with_different_init_sigma() {
    let s = 100.0;
    let k = 90.0;
    let t = 0.25;
    let r = 0.02;
    let target_sigma = 0.35;
    
    let price = bs_call(s, k, t, r, target_sigma);
    let iv = implied_vol(price, s, k, t, r, OptionType::Call, 0.1);
    
    assert!((iv - 0.35).abs() < 1e-6);
}
