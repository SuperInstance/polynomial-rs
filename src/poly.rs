//! Core [`Polynomial`] type and constructors.
//!
//! Polynomials are stored as a `Vec<f64>` of coefficients where index `i`
//! represents the coefficient of x^i.

use std::fmt;
use std::ops::{Add, Sub, Mul};

/// A polynomial with `f64` coefficients.
///
/// Coefficients are stored in ascending degree order:
/// `coeffs[i]` is the coefficient of x^i.
#[derive(Clone, Debug)]
pub struct Polynomial {
    /// Coefficients in ascending degree order. May have trailing zeros.
    coeffs: Vec<f64>,
}

impl Polynomial {
    /// Creates a polynomial from a slice of coefficients (ascending degree order).
    ///
    /// # Examples
    /// ```
    /// use polynomial_rs::Polynomial;
    /// // 3 + 2x + x²
    /// let p = Polynomial::from_coeffs(&[3.0, 2.0, 1.0]);
    /// assert_eq!(p.coeffs(), &[3.0, 2.0, 1.0]);
    /// ```
    pub fn from_coeffs(coeffs: &[f64]) -> Self {
        let mut p = Polynomial { coeffs: coeffs.to_vec() };
        p.trim();
        p
    }

    /// Creates the zero polynomial.
    pub fn zero() -> Self {
        Polynomial { coeffs: vec![0.0] }
    }

    /// Returns the coefficients as a slice.
    pub fn coeffs(&self) -> &[f64] {
        &self.coeffs
    }

    /// Returns the degree of the polynomial.
    /// The zero polynomial has degree 0.
    pub fn degree(&self) -> usize {
        if self.is_zero() {
            return 0;
        }
        self.coeffs.len() - 1
    }

    /// Returns the leading coefficient.
    pub fn leading_coeff(&self) -> f64 {
        if self.is_zero() {
            return 0.0;
        }
        *self.coeffs.last().unwrap()
    }

    /// Returns `true` if this is the zero polynomial.
    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|&c| c.abs() < 1e-14)
    }

    /// Removes trailing near-zero coefficients.
    fn trim(&mut self) {
        while self.coeffs.len() > 1 && self.coeffs.last().map(|c| c.abs() < 1e-14).unwrap_or(false) {
            self.coeffs.pop();
        }
    }

    /// Computes the formal derivative.
    ///
    /// d/dx (a₀ + a₁x + a₂x² + ...) = a₁ + 2a₂x + 3a₃x² + ...
    pub fn derivative(&self) -> Self {
        if self.coeffs.len() <= 1 {
            return Self::zero();
        }
        let deriv: Vec<f64> = self.coeffs[1..]
            .iter()
            .enumerate()
            .map(|(i, &c)| c * (i + 1) as f64)
            .collect();
        Polynomial::from_coeffs(&deriv)
    }

    /// Multiplies all coefficients by a scalar.
    pub fn scale(&self, factor: f64) -> Self {
        let coeffs: Vec<f64> = self.coeffs.iter().map(|&c| c * factor).collect();
        Polynomial::from_coeffs(&coeffs)
    }

    /// Divides `self` by `divisor`, returning `(quotient, remainder)`.
    ///
    /// Uses synthetic (long) division.
    pub fn divide(&self, divisor: &Polynomial) -> (Polynomial, Polynomial) {
        crate::arithmetic::div(self, divisor)
    }

    /// Checks approximate equality with a given tolerance.
    pub fn approx_eq(&self, other: &Polynomial, tol: f64) -> bool {
        let max_len = self.coeffs.len().max(other.coeffs.len());
        for i in 0..max_len {
            let a = self.coeffs.get(i).copied().unwrap_or(0.0);
            let b = other.coeffs.get(i).copied().unwrap_or(0.0);
            if (a - b).abs() > tol {
                return false;
            }
        }
        true
    }
}

impl PartialEq for Polynomial {
    fn eq(&self, other: &Self) -> bool {
        let max_len = self.coeffs.len().max(other.coeffs.len());
        for i in 0..max_len {
            let a = self.coeffs.get(i).copied().unwrap_or(0.0);
            let b = other.coeffs.get(i).copied().unwrap_or(0.0);
            if (a - b).abs() > 1e-14 {
                return false;
            }
        }
        true
    }
}
impl Eq for Polynomial {}

impl fmt::Display for Polynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }
        let mut terms = Vec::new();
        for (i, &c) in self.coeffs.iter().enumerate() {
            if c.abs() < 1e-14 {
                continue;
            }
            let term = match i {
                0 => format!("{c}"),
                1 => format!("{c}x"),
                _ => format!("{c}x^{i}"),
            };
            terms.push(term);
        }
        if terms.is_empty() {
            write!(f, "0")
        } else {
            write!(f, "{}", terms.join(" + "))
        }
    }
}

impl Add for &Polynomial {
    type Output = Polynomial;
    fn add(self, other: &Polynomial) -> Polynomial {
        crate::arithmetic::add(self, other)
    }
}

impl Add for Polynomial {
    type Output = Polynomial;
    fn add(self, other: Polynomial) -> Polynomial {
        crate::arithmetic::add(&self, &other)
    }
}

impl Sub for &Polynomial {
    type Output = Polynomial;
    fn sub(self, other: &Polynomial) -> Polynomial {
        crate::arithmetic::sub(self, other)
    }
}

impl Sub for Polynomial {
    type Output = Polynomial;
    fn sub(self, other: Polynomial) -> Polynomial {
        crate::arithmetic::sub(&self, &other)
    }
}

impl Mul for &Polynomial {
    type Output = Polynomial;
    fn mul(self, other: &Polynomial) -> Polynomial {
        crate::arithmetic::mul(self, other)
    }
}

impl Mul for Polynomial {
    type Output = Polynomial;
    fn mul(self, other: Polynomial) -> Polynomial {
        crate::arithmetic::mul(&self, &other)
    }
}
