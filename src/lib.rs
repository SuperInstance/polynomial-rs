//! # polynomial-rs
//!
//! Polynomial arithmetic library with FFT-based multiplication, evaluation at points,
//! and root finding via Newton's method.
//!
//! ## Modules
//! - [`poly`] - Core `Polynomial` type and constructors
//! - [`arithmetic`] - Add, subtract, multiply, divide polynomials
//! - [`fft`] - FFT-based multiplication for large polynomials
//! - [`evaluate`] - Horner's method evaluation and multi-point
//! - [`roots`] - Newton's method root finding

pub mod arithmetic;
pub mod evaluate;
pub mod fft;
pub mod poly;
pub mod roots;

pub use poly::Polynomial;

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create polynomial from coefficients (highest degree first → no, we store lowest first)
    fn poly(coeffs: &[f64]) -> Polynomial {
        Polynomial::from_coeffs(coeffs)
    }

    // ── Arithmetic tests ─────────────────────────────────────

    #[test]
    fn test_add_constant() {
        let a = poly(&[1.0, 2.0, 3.0]); // 1 + 2x + 3x²
        let b = poly(&[10.0]);           // 10
        let sum = &a + &b;
        assert_eq!(sum, poly(&[11.0, 2.0, 3.0]));
    }

    #[test]
    fn test_add_polynomials() {
        // (1 + 2x) + (3 + 4x + 5x²) = 4 + 6x + 5x²
        let a = poly(&[1.0, 2.0]);
        let b = poly(&[3.0, 4.0, 5.0]);
        assert_eq!(&a + &b, poly(&[4.0, 6.0, 5.0]));
    }

    #[test]
    fn test_subtract_polynomials() {
        let a = poly(&[5.0, 3.0, 2.0]);
        let b = poly(&[1.0, 1.0, 2.0]);
        assert_eq!(&a - &b, poly(&[4.0, 2.0]));
    }

    #[test]
    fn test_multiply_by_zero() {
        let a = poly(&[1.0, 2.0, 3.0]);
        let b = poly(&[0.0]);
        assert_eq!(&a * &b, poly(&[0.0]));
    }

    #[test]
    fn test_multiply_by_constant() {
        let a = poly(&[1.0, 2.0, 3.0]);
        let b = poly(&[2.0]);
        assert_eq!(&a * &b, poly(&[2.0, 4.0, 6.0]));
    }

    #[test]
    fn test_multiply_linear() {
        // (1 + x) * (1 - x) = 1 - x²
        let a = poly(&[1.0, 1.0]);
        let b = poly(&[1.0, -1.0]);
        assert_eq!(&a * &b, poly(&[1.0, 0.0, -1.0]));
    }

    #[test]
    fn test_multiply_quadratic() {
        // (1 + 2x + 3x²) * (4 + 5x) = 4 + 13x + 22x² + 15x³
        let a = poly(&[1.0, 2.0, 3.0]);
        let b = poly(&[4.0, 5.0]);
        assert_eq!(&a * &b, poly(&[4.0, 13.0, 22.0, 15.0]));
    }

    #[test]
    fn test_divide_simple() {
        // (x² - 1) / (x - 1) = (x + 1)
        let num = poly(&[-1.0, 0.0, 1.0]);
        let den = poly(&[-1.0, 1.0]);
        let (q, r) = num.divide(&den);
        assert_eq!(q, poly(&[1.0, 1.0]));
        assert!(r.is_zero() || r.degree() == 0 && r.coeffs()[0].abs() < 1e-10);
    }

    #[test]
    fn test_divide_with_remainder() {
        // (2x² + 3x + 1) / (x + 1) = 2x + 1, remainder 0
        let num = poly(&[1.0, 3.0, 2.0]);
        let den = poly(&[1.0, 1.0]);
        let (q, _r) = num.divide(&den);
        assert_eq!(q, poly(&[1.0, 2.0]));
    }

    // ── FFT tests ─────────────────────────────────────────────

    #[test]
    fn test_fft_matches_naive_small() {
        let a = poly(&[1.0, 2.0, 3.0]);
        let b = poly(&[4.0, 5.0]);
        let naive = arithmetic::mul(&a, &b);
        let fft_result = fft::fft_mul(&a, &b);
        assert!(naive.approx_eq(&fft_result, 1e-8));
    }

    #[test]
    fn test_fft_matches_naive_medium() {
        let a = poly(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = poly(&[6.0, 7.0, 8.0, 9.0]);
        let naive = arithmetic::mul(&a, &b);
        let fft_result = fft::fft_mul(&a, &b);
        assert!(naive.approx_eq(&fft_result, 1e-6));
    }

    #[test]
    fn test_fft_identity() {
        let a = poly(&[1.0, 2.0, 3.0]);
        let one = poly(&[1.0]);
        let result = fft::fft_mul(&a, &one);
        assert!(a.approx_eq(&result, 1e-8));
    }

    #[test]
    fn test_fft_zero() {
        let a = poly(&[1.0, 2.0, 3.0]);
        let zero = poly(&[0.0]);
        let result = fft::fft_mul(&a, &zero);
        assert!(result.is_zero());
    }

    // ── Evaluation tests ──────────────────────────────────────

    #[test]
    fn test_eval_constant() {
        let p = poly(&[5.0]);
        assert!((evaluate::eval(&p, 100.0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_linear() {
        // p(x) = 3 + 2x
        let p = poly(&[3.0, 2.0]);
        assert!((evaluate::eval(&p, 4.0) - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_quadratic() {
        // p(x) = 1 + 0x + 1x² (x² + 1)
        let p = poly(&[1.0, 0.0, 1.0]);
        assert!((evaluate::eval(&p, 3.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_negative() {
        // p(x) = 1 - x²
        let p = poly(&[1.0, 0.0, -1.0]);
        assert!((evaluate::eval(&p, 5.0) - (-24.0)).abs() < 1e-10);
    }

    #[test]
    fn test_eval_zero() {
        let p = poly(&[5.0, 3.0, 2.0]);
        assert!((evaluate::eval(&p, 0.0) - 5.0).abs() < 1e-10);
    }

    // ── Roots tests ───────────────────────────────────────────

    #[test]
    fn test_root_linear() {
        // p(x) = -2 + x, root at x=2
        let p = poly(&[-2.0, 1.0]);
        let root = roots::newton(&p, 0.0, 100, 1e-10);
        assert!(root.is_some());
        assert!((root.unwrap() - 2.0).abs() < 1e-8);
    }

    #[test]
    fn test_root_quadratic() {
        // p(x) = x² - 4, roots at ±2
        let p = poly(&[-4.0, 0.0, 1.0]);
        let root = roots::newton(&p, 3.0, 100, 1e-10);
        assert!(root.is_some());
        let r = root.unwrap();
        assert!((r - 2.0).abs() < 1e-6 || (r + 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_root_cubic() {
        // p(x) = x³ - 1, root at x=1
        let p = poly(&[-1.0, 0.0, 0.0, 1.0]);
        let root = roots::newton(&p, 2.0, 100, 1e-10);
        assert!(root.is_some());
        assert!((root.unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_root_quadratic_negative() {
        // p(x) = x² - 4, starting from -3 should find -2
        let p = poly(&[-4.0, 0.0, 1.0]);
        let root = roots::newton(&p, -3.0, 100, 1e-10);
        assert!(root.is_some());
        assert!((root.unwrap() + 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_root_known_quadratic() {
        // x² + 5x + 6 = (x+2)(x+3), roots at -2 and -3
        let p = poly(&[6.0, 5.0, 1.0]);
        let r1 = roots::newton(&p, -1.0, 100, 1e-10);
        let r2 = roots::newton(&p, -4.0, 100, 1e-10);
        assert!(r1.is_some());
        assert!(r2.is_some());
        assert!((r1.unwrap() + 2.0).abs() < 1e-6);
        assert!((r2.unwrap() + 3.0).abs() < 1e-6);
    }

    // ── Polynomial property tests ─────────────────────────────

    #[test]
    fn test_degree() {
        assert_eq!(poly(&[0.0]).degree(), 0);
        assert_eq!(poly(&[1.0]).degree(), 0);
        assert_eq!(poly(&[1.0, 2.0]).degree(), 1);
        assert_eq!(poly(&[1.0, 0.0, 3.0]).degree(), 2);
    }

    #[test]
    fn test_leading_coefficient() {
        let p = poly(&[1.0, 2.0, 5.0]);
        assert!((p.leading_coeff() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_zero_polynomial() {
        let z = poly(&[0.0]);
        assert!(z.is_zero());
        let z2 = poly(&[0.0, 0.0, 0.0]);
        assert!(z2.is_zero());
    }

    #[test]
    fn test_derivative() {
        // d/dx (3 + 2x + x²) = 2 + 2x
        let p = poly(&[3.0, 2.0, 1.0]);
        let d = p.derivative();
        assert_eq!(d, poly(&[2.0, 2.0]));
    }

    #[test]
    fn test_scale() {
        let p = poly(&[1.0, 2.0, 3.0]);
        let scaled = p.scale(2.0);
        assert_eq!(scaled, poly(&[2.0, 4.0, 6.0]));
    }
}
