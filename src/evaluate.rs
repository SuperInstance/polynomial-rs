//! Polynomial evaluation using Horner's method.

use crate::poly::Polynomial;

/// Evaluates the polynomial at point `x` using Horner's method.
///
/// Computes p(x) = a₀ + a₁x + a₂x² + ... + aₙxⁿ
/// as (...((aₙx + aₙ₋₁)x + aₙ₋₂)x + ... + a₁)x + a₀
/// in O(n) operations.
pub fn eval(p: &Polynomial, x: f64) -> f64 {
    let coeffs = p.coeffs();
    if coeffs.is_empty() {
        return 0.0;
    }
    // Horner's method: iterate from highest to lowest degree
    let mut result = 0.0;
    for &c in coeffs.iter().rev() {
        result = result * x + c;
    }
    result
}

/// Evaluates the polynomial at multiple points.
pub fn eval_multi(p: &Polynomial, points: &[f64]) -> Vec<f64> {
    points.iter().map(|&x| eval(p, x)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_multi() {
        let p = Polynomial::from_coeffs(&[1.0, 0.0, 1.0]); // 1 + x²
        let values = eval_multi(&p, &[0.0, 1.0, 2.0, 3.0]);
        assert!((values[0] - 1.0).abs() < 1e-10);
        assert!((values[1] - 2.0).abs() < 1e-10);
        assert!((values[2] - 5.0).abs() < 1e-10);
        assert!((values[3] - 10.0).abs() < 1e-10);
    }
}
