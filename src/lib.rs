use faer::stats::NanHandling;
use faer::{stats, Col, Mat, MatMut, MatRef};
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
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
    // Convert inputs to faer Matrices
    // Data is usually (n_samples, n_features) => (n_time, n_voxel)
    let n_samples = x.shape()[0];
    let n_features = x.shape()[1];

    let mut mat_x_centered = array_to_mat(x.as_array());

    // 1. Center Data
    // Calculate row means using faer::stats
    // We want mean of each row (collapsing columns) -> Result is Col vector (n_samples, 1).
    let mut row_means = Col::<f64>::zeros(n_samples);
    stats::col_mean(
        row_means.as_mut(),
        mat_x_centered.as_ref(),
        NanHandling::Propagate,
    );

    // Subtract row means
    // There isn't a direct broadcasting "sub_col" in faer yet for in-place modification of Mat by Col?
    // We can iterate, but let's check if we can improve this loop too.
    for r in 0..n_samples {
        let m = row_means[r];
        for c in 0..n_features {
            mat_x_centered[(r, c)] -= m;
        }
    }

    // 2. Whitening
    let rank = n_samples.min(n_features);

    let (mat_white_owned, n_samples_eff) = if whiten {
        let svd_x = mat_x_centered.thin_svd().unwrap();
        let s_diag = svd_x.S(); // Diagonal singular values
        let u = svd_x.U();

        let sqrt_n = (n_features as f64).sqrt();
        let mut k_mat = Mat::<f64>::zeros(n_samples, rank);

        // Accessing diagonal S
        for i in 0..rank {
            let val = s_diag[i];
            let inv_s = if val > 1e-12 { 1.0 / val } else { 0.0 };
            let scale_val = inv_s * sqrt_n;

            // k_mat[:, i] = u[:, i] * scale
            for r in 0..n_samples {
                k_mat[(r, i)] = u[(r, i)] * scale_val;
            }
        }

        // mat_white = k_mat.T * mat_x_centered
        // (rank, n_samples) * (n_samples, n_features) -> (rank, n_features)
        let mat_white = k_mat.transpose() * &mat_x_centered;
        (mat_white, rank)
    } else {
        (mat_x_centered, n_samples)
    };

    let mat_x_white = mat_white_owned.as_ref(); // (n_samples_eff, n_features)

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
    let mat_pinv_t = mat_pinv.transpose();

    let e_g_v = 0.3745672075;

    // Precompute stats for references
    // mat_references is (n_components, n_features).
    // We want stats for each row (component).
    let n_components = references.shape()[0];
    let mat_references = array_to_mat(references.as_array());
    let mut ref_means = Col::<f64>::zeros(n_components);
    stats::col_mean(
        ref_means.as_mut(),
        mat_references.as_ref(),
        NanHandling::Propagate,
    );

    let mut ref_stds = Col::<f64>::zeros(n_components);
    stats::col_varm(
        ref_stds.as_mut(),
        mat_references.as_ref(),
        ref_means.as_ref(),
        NanHandling::Propagate,
    );
    for r in 0..n_components {
        let ref_var = ref_stds[r];
        ref_stds[r] = if ref_var > 0.0 { ref_var.sqrt() } else { 0.0 };
    }

    let mut components = Mat::<f64>::zeros(n_components, n_features);
    for k in 0..n_components {
        let ref_row = mat_references.row(k); // RowRef
        let ref_mean = ref_means[k];
        let ref_std = ref_stds[k];

        // Normalize reference
        let ref_norm = if ref_std > 1e-15 {
            Col::<f64>::from_fn(n_features, |i| (ref_row[i] - ref_mean) / ref_std)
        } else {
            Col::<f64>::from_fn(n_features, |i| ref_row[i])
        };

        // mat_x_ref = mat_x_white * mat_ref_norm
        // (n_samples_eff, n_features) * (n_features, 1) -> (n_samples_eff, 1)
        let mat_x_ref = mat_x_white * &ref_norm;

        // w_init = pinv.T * ref_norm
        let w_init = mat_pinv_t * &ref_norm;

        // normalize w
        let w_norm = w_init.norm_l2();
        let mut w = w_init / w_norm;

        // y_est = mat_x_white.T * w
        let mut y_est = mat_x_white.transpose() * &w;

        // Calculate neg_j
        let mut e_gy_sum = 0.0;
        for i in 0..n_features {
            e_gy_sum += log_cosh(y_est[i]);
        }
        let e_gy = e_gy_sum / (n_features as f64);
        let neg_j = (e_gy - e_g_v).powi(2);

        // val_f = mean(y_est * ref_norm)
        let val_f = (y_est.transpose() * &ref_norm) / (n_features as f64);

        let initial_f_clamped = val_f.max(0.0).min(0.999);
        let ci = if neg_j > 1e-15 {
            (std::f64::consts::FRAC_PI_2 * initial_f_clamped).tan() / neg_j
        } else {
            1.0
        };

        let mut converged = false;

        for _iter in 0..max_iter {
            let w_old = w.clone();

            y_est = mat_x_white.transpose() * &w;

            // gy_tanh
            let e_gy =
                Col::<f64>::from_fn(n_features, |i| log_cosh(y_est[i])).sum() / (n_features as f64);
            let gy_tanh = Col::<f64>::from_fn(n_features, |i| y_est[i].tanh());
            let gamma = e_gy - e_g_v;
            let val_j = gamma * gamma;

            // mat_x_gy = mat_x_white * gy_tanh
            let x_gy = mat_x_white * &gy_tanh;

            let gj_scale = 2.0 * gamma / (n_features as f64);
            let dk_dj = std::f64::consts::FRAC_2_PI * ci / (1.0 + (ci * val_j).powi(2));

            let coeff_gy = gj_scale * dk_dj * alpha;
            let coeff_ref = (1.0 - alpha) / (n_features as f64);

            // grad_c = x_gy * coeff_gy + mat_x_ref * coeff_ref
            let grad_c = x_gy * coeff_gy + &mat_x_ref * coeff_ref;

            let grad_norm = grad_c.norm_l2();
            if grad_norm < 1e-15 {
                converged = true;
                break;
            }

            let direction = grad_c / grad_norm;

            // Line Search
            let mut mu = 1.0;
            let rho = 0.5;
            let beta = 0.02;

            let val_k = (2.0 / std::f64::consts::PI) * (ci * val_j).atan();
            let c_current = alpha * val_k + (1.0 - alpha) * val_f;

            let mut w_new = w.clone();
            let mut improved = false;

            for _ls in 0..10 {
                // w_try_un = w + direction * mu
                let w_try_un = &w + &direction * mu;
                let w_try_norm = w_try_un.norm_l2();
                let w_try = w_try_un / w_try_norm;

                let y_try = mat_x_white.transpose() * &w_try; // (n_features, 1)

                let scan_e_gy = Col::<f64>::from_fn(n_features, |i| log_cosh(y_try[i])).sum();

                let gamma_try = (scan_e_gy / (n_features as f64)) - e_g_v;
                let j_try = gamma_try * gamma_try;
                let k_try = (2.0 / std::f64::consts::PI) * (ci * j_try).atan();
                let f_try = (y_try.transpose() * &ref_norm) / (n_features as f64);
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

            let dot = w.transpose() * &w_old;
            if 1.0 - dot.abs() < tol {
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

        // Save component
        let y_final = mat_x_white.transpose() * &w;
        for j in 0..n_features {
            components[(k, j)] = y_final[j];
        }
    }

    // Create output Python array directly (C-order)
    let output = PyArray2::zeros(py, (n_components, n_features), false);

    // Get mutable ndarray view
    // We need to use `try_readwrite` or `readwrite`
    let mut out_view = output.readwrite();
    let mut out_array = out_view.as_array_mut();

    // Optimize copy using faer slice methods
    // out_array is C-order (Row-Major) and contiguous.
    let slice = out_array
        .as_slice_mut()
        .expect("Output array must be contiguous");

    let mut target = MatMut::from_row_major_slice_mut(slice, n_components, n_features);
    target.copy_from(&components);

    Ok(output.into_any())
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
    let slice = arr.as_slice().unwrap();
    MatRef::from_row_major_slice(slice, rows, cols).to_owned()
}

#[pymodule]
mod _gigica {
    #[pymodule_export]
    use super::gig_ica_fit;
}
