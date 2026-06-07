//! Root finding using Newton's method.
//!
//! Finds a root of `p(x) = 0` by iterating:
//! `x_{n+1} = x_n - p(x_n) / p'(x_n)`

use crate::evaluate;
use crate::poly::Polynomial;

/// Finds a root of `p` using Newton's method.
///
/// Starts from `x0`, iterates up to `max_iter` times, converges when
/// `|x_{n+1} - x_n| < tol`.
///
/// Returns `None` if the derivative vanishes or convergence fails.
pub fn newton(p: &Polynomial, x0: f64, max_iter: usize, tol: f64) -> Option<f64> {
    let dp = p.derivative();
    let mut x = x0;

    for _ in 0..max_iter {
        let px = evaluate::eval(p, x);
        let dpx = evaluate::eval(&dp, x);

        if dpx.abs() < 1e-30 {
            // Derivative too small, try a small perturbation
            return None;
        }

        let x_new = x - px / dpx;

        if (x_new - x).abs() < tol {
            return Some(x_new);
        }
        x = x_new;
    }

    // Check if we're close enough
    if evaluate::eval(p, x).abs() < tol * 100.0 {
        Some(x)
    } else {
        None
    }
}

/// Finds all real roots by attempting Newton's method from a grid of starting points.
///
/// Returns deduplicated roots sorted in ascending order.
pub fn find_roots(p: &Polynomial, range: (f64, f64), steps: usize) -> Vec<f64> {
    let mut roots = Vec::new();
    let (lo, hi) = range;
    let step = (hi - lo) / steps as f64;

    for i in 0..=steps {
        let x0 = lo + step * i as f64;
        if let Some(r) = newton(p, x0, 100, 1e-10) {
            // Check if this root is new (not a duplicate)
            if !roots.iter().any(|existing: &f64| (existing - r).abs() < 1e-6) {
                // Verify it's actually a root
                if evaluate::eval(p, r).abs() < 1e-4 {
                    roots.push(r);
                }
            }
        }
    }

    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    roots
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_roots_quadratic() {
        // (x-1)(x-3) = x² - 4x + 3
        let p = Polynomial::from_coeffs(&[3.0, -4.0, 1.0]);
        let roots = find_roots(&p, (-10.0, 10.0), 40);
        assert!(roots.len() >= 2);
        assert!((roots[0] - 1.0).abs() < 1e-4);
        assert!((roots[1] - 3.0).abs() < 1e-4);
    }
}
