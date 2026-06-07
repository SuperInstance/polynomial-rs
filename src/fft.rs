//! FFT-based polynomial multiplication.
//!
//! Implements the Cooley-Tukey radix-2 FFT algorithm for multiplying
//! two polynomials in O(n log n) time.

use crate::poly::Polynomial;

/// A complex number for internal FFT computation.
#[derive(Clone, Copy)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self {
        Complex { re, im }
    }

    fn zero() -> Self {
        Complex { re: 0.0, im: 0.0 }
    }

    fn mul(self, other: Complex) -> Complex {
        Complex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    fn add(self, other: Complex) -> Complex {
        Complex {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    fn sub(self, other: Complex) -> Complex {
        Complex {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

/// Multiplies two polynomials using FFT.
///
/// Pads both polynomials to the next power of two, computes the DFT of each,
/// multiplies point-wise, then inverts.
pub fn fft_mul(a: &Polynomial, b: &Polynomial) -> Polynomial {
    if a.is_zero() || b.is_zero() {
        return Polynomial::zero();
    }

    let result_len = a.coeffs().len() + b.coeffs().len() - 1;
    let n = next_power_of_two(result_len);

    // Pad and convert to complex
    let mut fa = vec![Complex::zero(); n];
    let mut fb = vec![Complex::zero(); n];
    for (i, &c) in a.coeffs().iter().enumerate() {
        fa[i] = Complex::new(c, 0.0);
    }
    for (i, &c) in b.coeffs().iter().enumerate() {
        fb[i] = Complex::new(c, 0.0);
    }

    // Forward FFT
    fft_in_place(&mut fa, false);
    fft_in_place(&mut fb, false);

    // Pointwise multiply
    for i in 0..n {
        fa[i] = fa[i].mul(fb[i]);
    }

    // Inverse FFT
    fft_in_place(&mut fa, true);

    // Extract real parts
    let result: Vec<f64> = fa.iter().take(result_len).map(|c| c.re / n as f64).collect();
    Polynomial::from_coeffs(&result)
}

/// In-place Cooley-Tukey FFT.
///
/// When `inverse` is `true`, computes the inverse DFT.
fn fft_in_place(a: &mut [Complex], inverse: bool) {
    let n = a.len();
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            a.swap(i, j);
        }
    }

    // Cooley-Tukey butterfly
    let mut len = 2;
    while len <= n {
        let ang = 2.0 * std::f64::consts::PI / len as f64 * if inverse { -1.0 } else { 1.0 };
        let wlen = Complex::new(ang.cos(), ang.sin());
        for i in (0..n).step_by(len) {
            let mut w = Complex::new(1.0, 0.0);
            for j in 0..len / 2 {
                let u = a[i + j];
                let v = a[i + j + len / 2].mul(w);
                a[i + j] = u.add(v);
                a[i + j + len / 2] = u.sub(v);
                w = w.mul(wlen);
            }
        }
        len *= 2;
    }
}

/// Returns the smallest power of 2 >= `n`.
fn next_power_of_two(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p *= 2;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(8), 8);
    }
}
