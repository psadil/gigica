use faer::{Mat, MatRef};
use ndarray::{Array1, Array2, Axis};
use numpy::{PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "gig_ica_fit_rust")]
fn gig_ica_fit<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    references: PyReadonlyArray2<'py, f64>,
    alpha: f64,
    whiten: bool,
    max_iter: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let x_arr = x.as_array();
    let references_arr = references.as_array();

    let n_samples = x_arr.shape()[0];
    let n_features = x_arr.shape()[1];
    let n_components = references_arr.shape()[0];

    // 1. Center Data
    let x_mean = x_arr.mean_axis(Axis(1)).unwrap();
    let mut x_centered = x_arr.to_owned();
    for i in 0..n_samples {
        let m = x_mean[i];
        for j in 0..n_features {
            x_centered[[i, j]] -= m;
        }
    }

    // 2. Whitening
    let rank = n_samples.min(n_features);
    let mat_x_centered = array_to_mat(x_centered.view());

    let (mat_white_owned, n_samples_eff) = if whiten {
        let svd_x = mat_x_centered.thin_svd().unwrap();
        let s_diag = svd_x.S();
        let u = svd_x.U();

        let sqrt_n = (n_features as f64).sqrt();
        let mut k_mat = Mat::<f64>::zeros(n_samples, rank);

        for i in 0..rank {
            let val = s_diag[i];
            let inv_s = if val > 1e-12 { 1.0 / val } else { 0.0 };
            let scale = inv_s * sqrt_n;
            for r in 0..n_samples {
                k_mat[(r, i)] = u[(r, i)] * scale;
            }
        }

        let mat_white = k_mat.transpose() * &mat_x_centered; // (rank, n_features)
        (mat_white, rank)
    } else {
        (mat_x_centered.clone(), n_samples)
    };

    let mat_x_white = mat_white_owned.as_ref();

    // Precompute Pseudo-Inverse of X_white using Faer SVD
    // pinv(X_white) = V * S^-1 * U^T
    let svd_xw = mat_x_white.thin_svd().unwrap();
    let s_xw = svd_xw.S();
    let u_xw = svd_xw.U();
    let v_xw = svd_xw.V();

    let min_dim = n_samples_eff.min(n_features);
    let mut s_inv_u_t = Mat::<f64>::zeros(min_dim, n_samples_eff);

    for i in 0..min_dim {
        let val = s_xw[i];
        let inv_val = if val > 1e-12 { 1.0 / val } else { 0.0 };
        for r in 0..n_samples_eff {
            s_inv_u_t[(i, r)] = u_xw[(r, i)] * inv_val;
        }
    }

    let mat_pinv = v_xw * s_inv_u_t;

    let mut components = Array2::<f64>::zeros((n_components, n_features));

    let e_g_v = 0.3745672075;

    for k in 0..n_components {
        let ref_row = references_arr.row(k);
        let ref_mean = ref_row.mean().unwrap();
        let ref_std = ref_row.std(1.0);
        let ref_norm = if ref_std > 1e-15 {
            (&ref_row - ref_mean) / ref_std
        } else {
            ref_row.to_owned()
        };

        // ref_norm as Mat
        let mat_ref_norm =
            MatRef::from_column_major_slice(ref_norm.as_slice().unwrap(), n_features, 1);
        let mat_x_ref = mat_x_white * mat_ref_norm;

        let mat_ref = Mat::from_fn(n_features, 1, |r, _| ref_norm[r]);
        let w_init_mat_res = mat_pinv.transpose() * &mat_ref;

        let mut w = Array1::from_shape_fn(n_samples_eff, |i| w_init_mat_res[(i, 0)]);
        let w_norm = w.mapv(|a| a * a).sum().sqrt();
        w /= w_norm;

        // y_est = mat_x_white.T * w
        let mat_w = MatRef::from_column_major_slice(w.as_slice().unwrap(), n_samples_eff, 1);
        let mat_y_est = mat_x_white.transpose() * mat_w;
        let mut y_est = Array1::from_vec(mat_y_est.col_as_slice(0).to_vec());

        let e_gy = y_est.mapv(|v| log_cosh(v)).mean().unwrap();
        let neg_j = (e_gy - e_g_v).powi(2);

        let val_f = (&y_est * &ref_norm).mean().unwrap();

        let initial_f_clamped = val_f.max(0.0).min(0.999);
        let ci = if neg_j > 1e-15 {
            (std::f64::consts::FRAC_PI_2 * initial_f_clamped).tan() / neg_j
        } else {
            1.0
        };

        let mut converged = false;
        for _iter in 0..max_iter {
            let w_old = w.clone();

            // Forward pass
            let mat_w = MatRef::from_column_major_slice(w.as_slice().unwrap(), n_samples_eff, 1);
            let mat_y_est = mat_x_white.transpose() * mat_w;
            // Map back to Array for element-wise ops
            // y_est needs to be updated!
            y_est = Array1::from_vec(mat_y_est.col_as_slice(0).to_vec());

            let gy_tanh = y_est.mapv(|v| v.tanh());
            let e_gy = y_est.mapv(|v| log_cosh(v)).mean().unwrap();

            let gamma = e_gy - e_g_v;
            let val_j = gamma.powi(2);

            // x_gy = x_white * gy_tanh
            let mat_gy_tanh =
                MatRef::from_column_major_slice(gy_tanh.as_slice().unwrap(), n_features, 1);
            let mat_x_gy = mat_x_white * mat_gy_tanh;
            // Convert to Array
            let x_gy_arr = Array1::from_vec(mat_x_gy.col_as_slice(0).to_vec());

            let grad_j = &x_gy_arr * (2.0 * gamma / (n_features as f64));

            let dk_dj = (2.0 / std::f64::consts::PI) * ci / (1.0 + (ci * val_j).powi(2));
            let grad_k = &grad_j * dk_dj;

            // grad_f = x_white * ref_norm
            let grad_f = Array1::from_vec(mat_x_ref.col_as_slice(0).to_vec()) / (n_features as f64);

            let grad_c = grad_k * alpha + grad_f * (1.0 - alpha);

            let grad_norm = grad_c.mapv(|v| v.powi(2)).sum().sqrt();
            if grad_norm < 1e-15 {
                converged = true;
                break;
            }
            let direction = &grad_c / grad_norm;

            let mut mu = 1.0;
            let rho = 0.5;
            let beta = 0.02;

            let val_k = (2.0 / std::f64::consts::PI) * (ci * val_j).atan();
            let c_current = alpha * val_k + (1.0 - alpha) * val_f;

            let mut w_new = w.clone();
            let mut improved = false;

            for _ls in 0..10 {
                let w_try_un = &w + &direction * mu;
                let w_try_norm = w_try_un.mapv(|v| v * v).sum().sqrt();
                let w_try = w_try_un / w_try_norm;

                // y_try = x_white.t * w_try
                let mat_w_try =
                    MatRef::from_column_major_slice(w_try.as_slice().unwrap(), n_samples_eff, 1);
                let mat_y_try = mat_x_white.transpose() * mat_w_try;
                let y_try_arr = Array1::from_vec(mat_y_try.col_as_slice(0).to_vec()); // Avoid overwriting y_try variable name confusion

                let gamma_try = y_try_arr.mapv(|v| log_cosh(v)).mean().unwrap() - e_g_v;
                let j_try = gamma_try.powi(2);
                let k_try = (2.0 / std::f64::consts::PI) * (ci * j_try).atan();
                let f_try = (&y_try_arr * &ref_norm).mean().unwrap();
                let c_try = alpha * k_try + (1.0 - alpha) * f_try;

                if c_try > c_current + beta * mu * grad_norm {
                    w_new = w_try;
                    improved = true;
                    break;
                } else {
                    mu *= rho;
                }
            }

            if improved {
                w = w_new;
            }

            let dot = w.dot(&w_old).abs();
            if 1.0 - dot < tol {
                converged = true;
                break;
            }
        }

        if !converged {
            if let Ok(logging) = py.import("logging") {
                let _ = logging.call_method1(
                    "warning",
                    (format!(
                        "GIG-ICA did not converge for component {} after {} iterations.",
                        k, max_iter
                    ),),
                );
            }
        }

        // y_final = x_white.t().dot(&w);
        // Reuse Faer logic
        let mat_w = MatRef::from_column_major_slice(w.as_slice().unwrap(), n_samples_eff, 1);
        let mat_y_final = mat_x_white.transpose() * mat_w;

        for j in 0..n_features {
            components[[k, j]] = mat_y_final[(j, 0)];
        }
    }

    Ok(components.to_pyarray(py).into_any())
}

fn log_cosh(x: f64) -> f64 {
    let x_abs = x.abs();
    if x_abs > 20.0 {
        x_abs - std::f64::consts::LN_2
    } else {
        x.cosh().ln()
    }
}

fn array_to_mat(arr: ndarray::ArrayView2<f64>) -> Mat<f64> {
    let (rows, cols) = arr.dim();
    if let Some(slice) = arr.as_slice() {
        // Array2 is Row-Major. MatExpects Col-Major.
        // Slice represents A(rows, cols) row-major.
        // Equivalent to B(cols, rows) col-major where B = A.T.
        // So we create MatRef(cols, rows) wrapping slice, then transpose.
        use faer::MatRef;
        MatRef::from_column_major_slice(slice, cols, rows)
            .transpose()
            .to_owned()
    } else {
        Mat::from_fn(rows, cols, |r, c| arr[[r, c]])
    }
}

#[pymodule]
mod _gigica {
    #[pymodule_export]
    use super::gig_ica_fit;
}
