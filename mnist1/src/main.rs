use std::iter::zip;

use mnist::{MnistBuilder, NormalizedMnist};
use network::Network;

mod network;
mod util;

fn main() {
    let NormalizedMnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .training_set_length(50_000)
        .test_set_length(10_000)
        .label_format_one_hot()
        .finalize()
        .normalize();

    let regular_data = |img: Vec<f32>, lbl: Vec<u8>| {
        zip(img.chunks(28 * 28), lbl.chunks(10))
            .map(|(img, lbl)| {
                (
                    img.iter().map(|pixel| *pixel as f64).collect::<Vec<_>>(),
                    lbl.iter().cloned().collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>()
    };

    let training_data = regular_data(trn_img, trn_lbl);
    let test_data = regular_data(tst_img, tst_lbl);

    let mut net = Network::new(vec![784, 40, 10]).unwrap();
    net.sdg(&training_data, 30, 10, 3.0, Some(&test_data));
}
