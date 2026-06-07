//! Polynomial arithmetic: add, subtract, multiply, divide.
//!
//! Uses naive O(n*m) multiplication. For large polynomials, see [`fft`](crate::fft).

use crate::poly::Polynomial;

/// Adds two polynomials.
pub fn add(a: &Polynomial, b: &Polynomial) -> Polynomial {
    let max_len = a.coeffs().len().max(b.coeffs().len());
    let mut result = vec![0.0; max_len];
    for (i, &c) in a.coeffs().iter().enumerate() {
        result[i] += c;
    }
    for (i, &c) in b.coeffs().iter().enumerate() {
        result[i] += c;
    }
    Polynomial::from_coeffs(&result)
}

/// Subtracts `b` from `a`.
pub fn sub(a: &Polynomial, b: &Polynomial) -> Polynomial {
    let max_len = a.coeffs().len().max(b.coeffs().len());
    let mut result = vec![0.0; max_len];
    for (i, &c) in a.coeffs().iter().enumerate() {
        result[i] += c;
    }
    for (i, &c) in b.coeffs().iter().enumerate() {
        result[i] -= c;
    }
    Polynomial::from_coeffs(&result)
}

/// Multiplies two polynomials using the naive O(n*m) algorithm.
pub fn mul(a: &Polynomial, b: &Polynomial) -> Polynomial {
    if a.is_zero() || b.is_zero() {
        return Polynomial::zero();
    }
    let mut result = vec![0.0; a.coeffs().len() + b.coeffs().len() - 1];
    for (i, &ca) in a.coeffs().iter().enumerate() {
        for (j, &cb) in b.coeffs().iter().enumerate() {
            result[i + j] += ca * cb;
        }
    }
    Polynomial::from_coeffs(&result)
}

/// Divides `numerator` by `denominator`, returning `(quotient, remainder)`.
///
/// Implements polynomial long division.
pub fn div(numerator: &Polynomial, denominator: &Polynomial) -> (Polynomial, Polynomial) {
    assert!(!denominator.is_zero(), "division by zero polynomial");

    let mut remainder = numerator.clone();
    let denom_lc = denominator.leading_coeff();
    let denom_deg = denominator.degree();

    if numerator.degree() < denom_deg {
        return (Polynomial::zero(), numerator.clone());
    }

    let mut quotient_coeffs = vec![0.0; numerator.degree() - denom_deg + 1];

    while !remainder.is_zero() && remainder.degree() >= denom_deg {
        let diff = remainder.degree() - denom_deg;
        let coeff = remainder.leading_coeff() / denom_lc;
        quotient_coeffs[diff] = coeff;

        // Subtract coeff * x^diff * denominator from remainder
        let mut sub_coeffs = vec![0.0; diff + denominator.coeffs().len()];
        for (i, &c) in denominator.coeffs().iter().enumerate() {
            sub_coeffs[i + diff] = c * coeff;
        }
        let sub_poly = Polynomial::from_coeffs(&sub_coeffs);
        remainder = sub(&remainder, &sub_poly);
    }

    (Polynomial::from_coeffs(&quotient_coeffs), remainder)
}
