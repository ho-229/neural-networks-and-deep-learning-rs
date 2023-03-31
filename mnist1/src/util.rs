use std::{
    iter::zip,
    ops::{AddAssign, SubAssign},
};

use rand::{thread_rng, Rng};
use rand_distr::{
    num_traits::{cast, Float, Num, NumCast},
    Distribution, StandardNormal,
};

pub type Array1D<T> = Vec<T>;
pub type Array2D<T> = Vec<Vec<T>>;
pub type Array3D<T> = Vec<Vec<Vec<T>>>;

#[inline]
pub fn sigmoid<F: Float>(f: F) -> F {
    // use std::f64::consts::E;
    // let e = F::from(E).unwrap();
    // F::one() / (F::one() + e.powf(-f))

    // fast sigmoid
    F::from(0.5).unwrap() * (f / (F::one() + f.abs()) + F::one())
}

#[inline]
pub fn sigmoid_prime<F: Float>(f: F) -> F {
    let res = sigmoid(f);
    res * (F::one() - res)
}

pub fn d2_dot_d1<'a, F: Float>(
    a: &'a Array2D<F>,
    b: &'a Array1D<F>,
) -> impl Iterator<Item = F> + 'a {
    a.iter().map(|row| {
        if row.len() != b.len() {
            panic!(
                "shapes ({}, {}) and ({},) not aligned",
                a.len(),
                row.len(),
                b.len()
            )
        }

        zip(row.iter(), b.iter())
            .map(|(a, b)| a.mul(*b))
            .fold(F::zero(), |sum, val| sum.add(val))
    })
}

#[inline]
pub fn d0_dot_d1<'a, F: Float>(a: F, b: &'a Array1D<F>) -> impl Iterator<Item = F> + 'a {
    b.iter().map(move |val| a.mul(*val))
}

#[inline]
pub fn d0_dot_d2<'a, F: Float>(
    a: F,
    b: &'a Array2D<F>,
) -> impl Iterator<Item = impl Iterator<Item = F> + 'a> + 'a {
    b.iter()
        .map(move |val| val.iter().map(move |val| a.mul(*val)))
}

#[inline]
pub fn d1_add_d1<'a, F: Float>(
    a: impl Iterator<Item = F> + 'a,
    b: impl Iterator<Item = F> + 'a,
) -> impl Iterator<Item = F> + 'a {
    zip(a, b).map(|(a, b)| a.add(b))
}

#[inline]
pub fn d1_sub_d1<'a, F: Float>(
    a: impl Iterator<Item = F> + 'a,
    b: impl Iterator<Item = F> + 'a,
) -> impl Iterator<Item = F> + 'a {
    zip(a, b).map(|(a, b)| a.sub(b))
}

#[inline]
pub fn d1_mul_d1<'a, F: Float>(
    a: impl Iterator<Item = F> + 'a,
    b: impl Iterator<Item = F> + 'a,
) -> impl Iterator<Item = F> + 'a {
    zip(a, b).map(|(a, b)| a.mul(b))
}

#[inline]
pub fn d1_add_assign_d1<'a, F: Float + AddAssign + 'a>(
    a: impl Iterator<Item = &'a mut F> + 'a,
    b: impl Iterator<Item = F> + 'a,
) {
    zip(a, b).for_each(|(a, b)| a.add_assign(b));
}

#[inline]
pub fn d2_add_assign_d2<'a, F: Float + AddAssign + 'a>(
    a: impl Iterator<Item = impl Iterator<Item = &'a mut F> + 'a> + 'a,
    b: impl Iterator<Item = impl Iterator<Item = F> + 'a> + 'a,
) {
    zip(a, b).for_each(|(a, b)| d1_add_assign_d1(a, b));
}

#[inline]
pub fn d1_sub_assign_d1<'a, F: Float + SubAssign + 'a>(
    a: impl Iterator<Item = &'a mut F> + 'a,
    b: impl Iterator<Item = F> + 'a,
) {
    zip(a, b).for_each(|(a, b)| a.sub_assign(b));
}

#[inline]
pub fn d2_sub_assign_d2<'a, F: Float + SubAssign + 'a>(
    a: impl Iterator<Item = impl Iterator<Item = &'a mut F> + 'a> + 'a,
    b: impl Iterator<Item = impl Iterator<Item = F> + 'a> + 'a,
) {
    zip(a, b).for_each(|(a, b)| d1_sub_assign_d1(a, b));
}

#[inline]
pub fn d2_same_shape<F: Float>(a: &Array2D<F>) -> Array2D<F> {
    a.iter().map(|a| vec![F::zero(); a.len()]).collect()
}

#[inline]
pub fn d3_same_shape<F: Float>(a: &Array3D<F>) -> Array3D<F> {
    a.iter().map(|a| d2_same_shape(a)).collect()
}

pub fn d2_transpose<N: Copy>(a: &Array2D<N>) -> Array2D<N> {
    (0..a.first().map(|row| row.len()).unwrap_or(0))
        .map(|i| {
            a.iter()
                .map(|row| *row.get(i).expect("shapes not aligned"))
                .collect()
        })
        .collect()
}

#[inline]
pub fn random_array<B, I>(size: usize) -> B
where
    I: Sized,
    B: FromIterator<I>,
    StandardNormal: Distribution<I>,
{
    thread_rng()
        .sample_iter(StandardNormal)
        .take(size)
        .collect()
}

#[inline]
pub fn one_hot2digit<N: Num + PartialOrd + Copy>(a: &Array1D<N>) -> u8 {
    a.iter()
        .enumerate()
        .fold((0, N::zero()), |(max_i, max_val), (i, val)| {
            match *val > max_val {
                true => (i as u8, *val),
                false => (max_i, max_val),
            }
        })
        .0
}

#[inline]
pub fn digit2one_hot<N: NumCast>(n: u8) -> impl Iterator<Item = N> {
    (0..10u8).map(move |i| cast((i == n) as u8).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_d2_transpose() {
        let a = vec![vec![1, 2, 6], vec![3, 4, 7]];

        assert_eq!(d2_transpose(&a), vec![vec![1, 3], vec![2, 4], vec![6, 7]]);
    }
}
