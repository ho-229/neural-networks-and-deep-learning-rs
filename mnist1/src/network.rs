use std::iter::zip;

use thiserror::Error;

use crate::util::*;

#[derive(Debug, Error)]
pub enum Error {
    #[error("layer sizes must greater than 2")]
    InvalidLayerSizes,
    #[error("invalid input")]
    InvalidInput,
}

pub type Result<T> = std::result::Result<T, Error>;
pub type Sample = (
    Vec<f64>, // image
    u8,       // label
);

#[derive(Debug, Clone)]
pub struct Network {
    input_len: usize,
    biases: Array2D<f64>,
    weights: Array3D<f64>,
}

impl Network {
    pub fn new(mut layer_sizes: Vec<usize>) -> Result<Self> {
        layer_sizes.retain(|size| *size > 0);
        if layer_sizes.len() < 3 {
            Err(Error::InvalidLayerSizes)?;
        }

        let biases = layer_sizes[1..]
            .iter()
            .map(|count| random_array(*count))
            .collect::<Array2D<f64>>();

        let weights = zip(
            layer_sizes[..layer_sizes.len() - 1].iter(),
            layer_sizes[1..].iter(),
        )
        .map(|(col, row)| (0..*row).map(|_| random_array(*col)).collect())
        .collect::<Array3D<f64>>();

        Ok(Self {
            input_len: layer_sizes[0],
            biases,
            weights,
        })
    }

    pub fn layer_count(&self) -> usize {
        if self.biases.len() != self.weights.len() {
            unreachable!();
        }

        self.biases.len() + 1
    }

    pub fn feed_forward(&self, mut input: Array1D<f64>) -> Result<Array1D<f64>> {
        if input.len() != self.input_len {
            Err(Error::InvalidInput)?;
        }

        for (b, w) in zip(self.biases.iter(), self.weights.iter()) {
            input = d1_add_d1(d2_dot_d1(&w, &input), b.iter().cloned())
                .map(|val| sigmoid(val))
                .collect();
        }

        Ok(input)
    }

    pub fn sdg(
        &mut self,
        training_data: &Vec<Sample>,
        epochs: u32,
        mini_batch_size: usize,
        eta: f64,
        test_data: Option<&Vec<Sample>>,
    ) {
        for j in 0..epochs {
            training_data
                .chunks(mini_batch_size)
                .for_each(|mini_batch| self.update_mini_batch(mini_batch, eta));

            if let Some(test_data) = test_data {
                println!(
                    "Epoch {j} : {} / {}",
                    self.evaluate(test_data),
                    test_data.len()
                );
            } else {
                println!("Epoch {j} complete");
            }
        }
    }

    fn update_mini_batch(&mut self, mini_batch: &[Sample], eta: f64) {
        let mut nabla_b = d2_same_shape(&self.biases);
        let mut nabla_w = d3_same_shape(&self.weights);

        for (img, lbl) in mini_batch {
            let (delta_nabla_b, delta_nabla_w) = self.backprop(img, lbl);

            zip(nabla_b.iter_mut(), delta_nabla_b)
                .for_each(|(nb, dnb)| d1_add_assign_d1(nb.iter_mut(), dnb.into_iter()));
            zip(nabla_w.iter_mut(), delta_nabla_w).for_each(|(nw, dnw)| {
                d2_add_assign_d2(
                    nw.iter_mut().map(|inner| inner.iter_mut()),
                    dnw.into_iter().map(|inner| inner.into_iter()),
                );
            });
        }

        zip(self.weights.iter_mut(), nabla_w).for_each(|(w, nw)| {
            d2_sub_assign_d2(
                w.iter_mut().map(|inner| inner.iter_mut()),
                d0_dot_d2(eta / mini_batch.len() as f64, &nw),
            )
        });
        zip(self.biases.iter_mut(), nabla_b).for_each(|(b, nb)| {
            d1_sub_assign_d1(b.iter_mut(), d0_dot_d1(eta / mini_batch.len() as f64, &nb))
        });
    }

    fn backprop(&self, img: &Vec<f64>, lbl: &u8) -> (Array2D<f64>, Array3D<f64>) {
        let mut nabla_b = d2_same_shape(&self.biases);
        let mut nabla_w = d3_same_shape(&self.weights);

        let mut activation = img.clone();
        let mut activations = vec![img.clone()];
        let mut zs = Vec::new();

        for (b, w) in zip(self.biases.iter(), self.weights.iter()) {
            let z = d1_add_d1(d2_dot_d1(w, &activation), b.iter().cloned()).collect::<Vec<_>>();
            zs.push(z);
            activation = zs.last().unwrap().iter().map(|val| sigmoid(*val)).collect();
            activations.push(activation.clone());
        }

        let mut delta = d1_mul_d1(
            d1_sub_d1(
                activations.last().unwrap().iter().cloned(),
                digit2one_hot::<f64>(*lbl),
            ),
            zs.last().unwrap().iter().map(|val| sigmoid_prime(*val)),
        )
        .collect::<Vec<_>>();

        *nabla_b.last_mut().unwrap() = delta.clone();
        *nabla_w.last_mut().unwrap() = delta
            .iter()
            .map(|val| d0_dot_d1(*val, &activations[activations.len() - 2]).collect())
            .collect();

        for l in 2..self.layer_count() {
            let sp = zs[zs.len() - l].iter().map(|val| sigmoid_prime(*val));
            delta = d1_mul_d1(
                d2_dot_d1(
                    &d2_transpose(&self.weights[self.weights.len() - l + 1]),
                    &delta,
                ),
                sp,
            )
            .collect();

            let len = nabla_b.len();
            nabla_b[len - l] = delta.clone();

            let len = nabla_w.len();
            nabla_w[len - l] = delta
                .iter()
                .map(|val| d0_dot_d1(*val, &activations[activations.len() - l - 1]).collect())
                .collect();
        }

        (nabla_b, nabla_w)
    }

    fn evaluate(&self, test_data: &Vec<Sample>) -> u32 {
        test_data
            .iter()
            .map(|(img, lbl)| {
                (one_hot2digit(&self.feed_forward(img.clone()).unwrap()) == *lbl) as u32
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::Network;

    #[test]
    fn test_network() {
        let network = Network::new(vec![2, 3, 1]).unwrap();

        println!("{:#?}", network);
        assert_eq!(3, network.layer_count());

        Network::new(vec![2, 0, 1]).unwrap_err();

        let network = Network::new(vec![2, 3, 0, 1]).unwrap();

        println!("{:#?}", network);
        assert_eq!(3, network.layer_count());

        let res = network.feed_forward(vec![0., 1.]).unwrap();
        println!("{res:#?}");
    }
}
