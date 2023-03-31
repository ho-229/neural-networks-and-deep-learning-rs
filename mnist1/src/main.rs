use std::iter::zip;

use mnist::{Mnist, MnistBuilder};
use network::Network;

mod network;
mod util;

fn main() {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .training_set_length(50_000)
        .test_set_length(10_000)
        .finalize();

    let regular_data = |img: Vec<u8>, lbl: Vec<u8>| {
        zip(img.chunks(28 * 28), lbl)
            .map(|(img, lbl)| {
                (
                    img.iter()
                        .map(|pixel| *pixel as f64 / 255.)
                        .collect::<Vec<_>>(),
                    lbl,
                )
            })
            .collect::<Vec<_>>()
    };

    let training_data = regular_data(trn_img, trn_lbl);
    let test_data = regular_data(tst_img, tst_lbl);

    let mut net = Network::new(vec![784, 30, 10]).unwrap();
    net.sdg(&training_data, 30, 10, 3.0, Some(&test_data));
}
