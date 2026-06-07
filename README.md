# polynomial-rs

Polynomial arithmetic library in pure Rust.

## Features
- Add, subtract, multiply, divide polynomials
- FFT-based multiplication (Cooley-Tukey)
- Horner's method evaluation
- Newton's method root finding
- Zero external dependencies

## Usage
```rust
use polynomial_rs::Polynomial;

let p = Polynomial::from_coeffs(&[-1.0, 0.0, 1.0]); // x² - 1
let q = Polynomial::from_coeffs(&[1.0, 1.0]);       // 1 + x

let product = &p * &q;
let value = polynomial_rs::evaluate::eval(&p, 3.0); // 8.0
```

License: MIT OR Apache-2.0
