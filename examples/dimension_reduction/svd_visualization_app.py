from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import io
import base64
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import TSNE

matplotlib.use("Agg")  # Use non-interactive backend

app = Flask(__name__)


def load_bars_data():
    """Load the bars dataset"""
    try:
        df = pd.read_csv(
            "../../datasets/ml_05/bars.csv",
            header=None,
        )
        return df.values
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create a simple synthetic bars dataset if file not found
        return create_synthetic_bars()


def create_synthetic_bars():
    """Create a synthetic bars dataset for demonstration"""
    np.random.seed(42)
    data = []

    # Create different bar patterns
    patterns = [
        # Horizontal bars
        np.array([255] * 8 + [0] * 56),  # Top row
        np.array([0] * 8 + [255] * 8 + [0] * 48),  # Second row
        np.array([0] * 16 + [255] * 8 + [0] * 40),  # Third row
        # Vertical bars
        np.array([255, 0, 0, 0, 0, 0, 0, 0] * 8),  # Left column
        np.array([0, 255, 0, 0, 0, 0, 0, 0] * 8),  # Second column
        # Mixed patterns
        np.array([255] * 16 + [0] * 16 + [255] * 16 + [0] * 16),
        np.array(
            [0] * 8
            + [255] * 8
            + [0] * 8
            + [255] * 8
            + [0] * 8
            + [255] * 8
            + [0] * 8
            + [255] * 8
        ),
    ]

    # Generate variations
    for pattern in patterns:
        for _ in range(50):  # 50 variations of each pattern
            noise = np.random.normal(0, 10, 64)
            noisy_pattern = np.clip(pattern + noise, 0, 255)
            data.append(noisy_pattern)

    return np.array(data)


def matrix_to_image_base64(matrix, title="Image"):
    """Convert a matrix to a base64 encoded image"""
    fig, ax = plt.subplots(figsize=(4, 4))

    # Reshape to 8x8 if it's a vector
    if matrix.ndim == 1:
        matrix = matrix.reshape(8, 8)

    ax.imshow(matrix, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return image_base64


def compute_svd_step_by_step(data, n_components=None):
    """Compute SVD and return step-by-step results"""
    if n_components is None:
        n_components = min(data.shape)

    # Center the data (subtract mean)
    data_centered = data - np.mean(data, axis=0)

    # Compute SVD
    U, s, Vt = np.linalg.svd(data_centered, full_matrices=False)

    # Truncate to n_components
    U_truncated = U[:, :n_components]
    s_truncated = s[:n_components]
    Vt_truncated = Vt[:n_components, :]

    # Compute reconstruction
    reconstruction = U_truncated @ np.diag(
        s_truncated
    ) @ Vt_truncated + np.mean(data, axis=0)

    return {
        "U": U_truncated,
        "s": s_truncated,
        "Vt": Vt_truncated,
        "reconstruction": reconstruction,
        "original": data,
        "centered": data_centered,
        "explained_variance": s_truncated**2 / np.sum(s**2),
        "cumulative_variance": np.cumsum(s_truncated**2) / np.sum(s**2),
    }


def compute_nnmf_step_by_step(data, n_components=None, max_iter=200):
    """Compute NNMF and return step-by-step results"""
    if n_components is None:
        n_components = min(data.shape)

    # Ensure data is non-negative (clip negative values to 0)
    data_non_negative = np.maximum(data, 0)

    # Initialize NNMF
    nmf = NMF(n_components=n_components, max_iter=max_iter, random_state=42)

    # Fit the model
    W = nmf.fit_transform(data_non_negative)
    H = nmf.components_

    # Compute reconstruction
    reconstruction = W @ H

    # Compute explained variance (for NNMF, we use reconstruction quality)
    total_variance = np.sum(data_non_negative**2)
    reconstruction_variance = np.sum(reconstruction**2)
    explained_variance_ratio = reconstruction_variance / total_variance

    return {
        "W": W,
        "H": H,
        "reconstruction": reconstruction,
        "original": data_non_negative,
        "explained_variance_ratio": explained_variance_ratio,
        "reconstruction_error": nmf.reconstruction_err_,
        "n_iter": nmf.n_iter_,
    }


@app.route("/")
def index():
    """Home page with dimension reduction overview"""
    return render_template("index.html")


@app.route("/svd")
def svd_page():
    """SVD visualization page"""
    return render_template("svd.html")


@app.route("/api/data")
def get_data():
    """Get the bars dataset"""
    data = load_bars_data()
    return jsonify(
        {
            "data": data.tolist(),
            "shape": data.shape,
            "description": "Bars dataset - 8x8 pixel images representing bar patterns",
        }
    )


@app.route("/api/svd", methods=["POST"])
def compute_svd():
    """Compute SVD reconstruction"""
    try:
        data = request.json.get("data")
        n_components = request.json.get("n_components", 10)

        if not data:
            return jsonify({"error": "No data provided"}), 400

        data_array = np.array(data)
        svd_results = compute_svd_step_by_step(data_array, n_components)

        # Convert matrices to base64 images for visualization
        results = {
            "n_components": n_components,
            "explained_variance": svd_results["explained_variance"].tolist(),
            "cumulative_variance": svd_results["cumulative_variance"].tolist(),
            "singular_values": svd_results["s"].tolist(),
            "reconstruction_error": np.mean(
                (svd_results["original"] - svd_results["reconstruction"]) ** 2
            ),
            "compression_ratio": (
                n_components * (data_array.shape[0] + data_array.shape[1])
            )
            / (data_array.shape[0] * data_array.shape[1]),
        }

        # Generate sample visualizations
        sample_indices = [0, 1, 2] if data_array.shape[0] >= 3 else [0]

        for i, idx in enumerate(sample_indices):
            if idx < data_array.shape[0]:
                results[f"original_{i}"] = matrix_to_image_base64(
                    svd_results["original"][idx], f"Original {i+1}"
                )
                results[f"reconstructed_{i}"] = matrix_to_image_base64(
                    svd_results["reconstruction"][idx], f"Reconstructed {i+1}"
                )

        # Show first few principal components
        for i in range(min(4, n_components)):
            # Scale the principal component for better visualization
            pc_scaled = svd_results["Vt"][i].copy()
            # Normalize to 0-255 range for better visualization
            pc_min, pc_max = pc_scaled.min(), pc_scaled.max()
            if pc_max > pc_min:
                pc_scaled = (
                    (pc_scaled - pc_min) / (pc_max - pc_min) * 255
                ).astype(np.uint8)
            else:
                pc_scaled = np.zeros_like(pc_scaled, dtype=np.uint8)

            results[f"pc_{i}"] = matrix_to_image_base64(pc_scaled, f"PC {i+1}")

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/nnmf", methods=["POST"])
def compute_nnmf():
    """Compute NNMF reconstruction"""
    try:
        data = request.json.get("data")
        n_components = request.json.get("n_components", 10)
        max_iter = request.json.get("max_iter", 200)

        if not data:
            return jsonify({"error": "No data provided"}), 400

        data_array = np.array(data)
        nnmf_results = compute_nnmf_step_by_step(
            data_array, n_components, max_iter
        )

        # Convert matrices to base64 images for visualization
        results = {
            "n_components": n_components,
            "explained_variance_ratio": nnmf_results[
                "explained_variance_ratio"
            ],
            "reconstruction_error": nnmf_results["reconstruction_error"],
            "n_iter": nnmf_results["n_iter"],
            "compression_ratio": (
                n_components * (data_array.shape[0] + data_array.shape[1])
            )
            / (data_array.shape[0] * data_array.shape[1]),
        }

        # Generate sample visualizations
        sample_indices = [0, 1, 2] if data_array.shape[0] >= 3 else [0]

        for i, idx in enumerate(sample_indices):
            if idx < data_array.shape[0]:
                results[f"original_{i}"] = matrix_to_image_base64(
                    nnmf_results["original"][idx], f"Original {i+1}"
                )
                results[f"reconstructed_{i}"] = matrix_to_image_base64(
                    nnmf_results["reconstruction"][idx], f"Reconstructed {i+1}"
                )

        # Show first few basis components (H matrix rows)
        for i in range(min(4, n_components)):
            # Scale the basis component for better visualization
            basis_scaled = nnmf_results["H"][i].copy()
            # Normalize to 0-255 range for better visualization
            basis_min, basis_max = basis_scaled.min(), basis_scaled.max()
            if basis_max > basis_min:
                basis_scaled = (
                    (basis_scaled - basis_min) / (basis_max - basis_min) * 255
                ).astype(np.uint8)
            else:
                basis_scaled = np.zeros_like(basis_scaled, dtype=np.uint8)

            results[f"basis_{i}"] = matrix_to_image_base64(
                basis_scaled, f"Basis {i+1}"
            )

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/svd-math")
def svd_math_page():
    """SVD mathematical explanation page"""
    return render_template("svd_math.html")


@app.route("/nnmf")
def nnmf_page():
    """NNMF visualization page"""
    return render_template("nnmf.html")


@app.route("/nnmf-math")
def nnmf_math_page():
    """NNMF mathematical explanation page"""
    return render_template("nnmf_math.html")


@app.route("/pca")
def pca_page():
    """PCA visualization and explanation page"""
    return render_template("pca.html")


@app.route("/tsne")
def tsne_page():
    """t-SNE visualization and explanation page"""
    return render_template("tsne.html")


@app.route("/api/nnmf-example")
def get_nnmf_example():
    """Return detailed 2x2 NNMF example with all calculations"""
    # Create a simple 2x2 example matrix
    A = np.array([[3, 1], [1, 3]])

    # Initialize with specific values for reproducible example
    np.random.seed(42)
    W0 = np.array([[0.5, 0.3], [0.7, 0.4]])
    H0 = np.array([[0.6, 0.8], [0.2, 0.9]])

    # Calculate actual matrix products for detailed steps
    W0T = W0.T
    H0T = H0.T

    # W^T A calculation
    WTA = W0T @ A
    WTA_00 = W0T[0][0] * A[0][0] + W0T[0][1] * A[1][0]
    WTA_01 = W0T[0][0] * A[0][1] + W0T[0][1] * A[1][1]
    WTA_10 = W0T[1][0] * A[0][0] + W0T[1][1] * A[1][0]
    WTA_11 = W0T[1][0] * A[0][1] + W0T[1][1] * A[1][1]

    # W^T W calculation
    WTW = W0T @ W0
    WTW_00 = W0T[0][0] * W0[0][0] + W0T[0][1] * W0[1][0]
    WTW_01 = W0T[0][0] * W0[0][1] + W0T[0][1] * W0[1][1]
    WTW_10 = W0T[1][0] * W0[0][0] + W0T[1][1] * W0[1][0]
    WTW_11 = W0T[1][0] * W0[0][1] + W0T[1][1] * W0[1][1]

    # W^T W H calculation
    WTWH = WTW @ H0
    WTWH_00 = WTW[0][0] * H0[0][0] + WTW[0][1] * H0[1][0]
    WTWH_01 = WTW[0][0] * H0[0][1] + WTW[0][1] * H0[1][1]
    WTWH_10 = WTW[1][0] * H0[0][0] + WTW[1][1] * H0[1][0]
    WTWH_11 = WTW[1][0] * H0[0][1] + WTW[1][1] * H0[1][1]

    # Update H using multiplicative rule
    H1 = H0 * (WTA / WTWH)

    # A H^T calculation (using updated H)
    AHT = A @ H1.T
    AHT_00 = A[0][0] * H1[0][0] + A[0][1] * H1[1][0]
    AHT_01 = A[0][0] * H1[0][1] + A[0][1] * H1[1][1]
    AHT_10 = A[1][0] * H1[0][0] + A[1][1] * H1[1][0]
    AHT_11 = A[1][0] * H1[0][1] + A[1][1] * H1[1][1]

    # W H H^T calculation
    WHHT = W0 @ H1 @ H1.T
    WHHT_00 = (
        W0[0][0] * H1[0][0] * H1[0][0]
        + W0[0][0] * H1[0][1] * H1[1][0]
        + W0[0][1] * H1[1][0] * H1[0][0]
        + W0[0][1] * H1[1][1] * H1[1][0]
    )
    WHHT_01 = (
        W0[0][0] * H1[0][0] * H1[0][1]
        + W0[0][0] * H1[0][1] * H1[1][1]
        + W0[0][1] * H1[1][0] * H1[0][1]
        + W0[0][1] * H1[1][1] * H1[1][1]
    )
    WHHT_10 = (
        W0[1][0] * H1[0][0] * H1[0][0]
        + W0[1][0] * H1[0][1] * H1[1][0]
        + W0[1][1] * H1[1][0] * H1[0][0]
        + W0[1][1] * H1[1][1] * H1[1][0]
    )
    WHHT_11 = (
        W0[1][0] * H1[0][0] * H1[0][1]
        + W0[1][0] * H1[0][1] * H1[1][1]
        + W0[1][1] * H1[1][0] * H1[0][1]
        + W0[1][1] * H1[1][1] * H1[1][1]
    )

    # Update W using multiplicative rule
    W1 = W0 * (AHT / WHHT)

    # Calculate reconstruction error
    reconstruction = W1 @ H1
    error = np.linalg.norm(A - reconstruction, "fro")

    # Step-by-step calculations
    steps = {
        "step1": {
            "title": "Initialize Matrices",
            "description": "Start with random non-negative matrices W and H",
            "initialization": f"W^{{(0)}} = \\begin{{bmatrix}} {W0[0][0]} & {W0[0][1]} \\\\ {W0[1][0]} & {W0[1][1]} \\end{{bmatrix}}, \\quad H^{{(0)}} = \\begin{{bmatrix}} {H0[0][0]} & {H0[0][1]} \\\\ {H0[1][0]} & {H0[1][1]} \\end{{bmatrix}}",
            "check_non_negative": "All elements ≥ 0 ✓",
            "note": "These are randomly initialized non-negative values",
        },
        "step2": {
            "title": "First Iteration - Update H",
            "description": "Update H using multiplicative rule: H_{kj} ← H_{kj} × (W^T A)_{kj} / (W^T W H)_{kj}",
            "calculate_WT": f"W^T = \\begin{{bmatrix}} {W0[0][0]} & {W0[1][0]} \\\\ {W0[0][1]} & {W0[1][1]} \\end{{bmatrix}}",
            "calculate_WTA": f"W^T A = \\begin{{bmatrix}} {W0[0][0]} & {W0[1][0]} \\\\ {W0[0][1]} & {W0[1][1]} \\end{{bmatrix}} \\begin{{bmatrix}} {A[0][0]} & {A[0][1]} \\\\ {A[1][0]} & {A[1][1]} \\end{{bmatrix}} = \\begin{{bmatrix}} {WTA_00:.2f} & {WTA_01:.2f} \\\\ {WTA_10:.2f} & {WTA_11:.2f} \\end{{bmatrix}}",
            "calculate_WTW": f"W^T W = \\begin{{bmatrix}} {WTW_00:.2f} & {WTW_01:.2f} \\\\ {WTW_10:.2f} & {WTW_11:.2f} \\end{{bmatrix}}",
            "calculate_WTWH": f"W^T W H = \\begin{{bmatrix}} {WTW_00:.2f} & {WTW_01:.2f} \\\\ {WTW_10:.2f} & {WTW_11:.2f} \\end{{bmatrix}} \\begin{{bmatrix}} {H0[0][0]} & {H0[0][1]} \\\\ {H0[1][0]} & {H0[1][1]} \\end{{bmatrix}} = \\begin{{bmatrix}} {WTWH_00:.2f} & {WTWH_01:.2f} \\\\ {WTWH_10:.2f} & {WTWH_11:.2f} \\end{{bmatrix}}",
            "update_H": f"H^{{(1)}} = H^{{(0)}} \\odot \\frac{{W^T A}}{{W^T W H}} = \\begin{{bmatrix}} {H0[0][0]} & {H0[0][1]} \\\\ {H0[1][0]} & {H0[1][1]} \\end{{bmatrix}} \\odot \\begin{{bmatrix}} \\frac{{{WTA_00:.2f}}}{{{WTWH_00:.2f}}} & \\frac{{{WTA_01:.2f}}}{{{WTWH_01:.2f}}} \\\\ \\frac{{{WTA_10:.2f}}}{{{WTWH_10:.2f}}} & \\frac{{{WTA_11:.2f}}}{{{WTWH_11:.2f}}} \\end{{bmatrix}} = \\begin{{bmatrix}} {H1[0][0]:.3f} & {H1[0][1]:.3f} \\\\ {H1[1][0]:.3f} & {H1[1][1]:.3f} \\end{{bmatrix}}",
        },
        "step3": {
            "title": "First Iteration - Update W",
            "description": "Update W using multiplicative rule: W_{ik} ← W_{ik} × (A H^T)_{ik} / (W H H^T)_{ik}",
            "calculate_HT": f"H^T = \\begin{{bmatrix}} {H1[0][0]:.3f} & {H1[1][0]:.3f} \\\\ {H1[0][1]:.3f} & {H1[1][1]:.3f} \\end{{bmatrix}}",
            "calculate_AHT": f"A H^T = \\begin{{bmatrix}} {A[0][0]} & {A[0][1]} \\\\ {A[1][0]} & {A[1][1]} \\end{{bmatrix}} \\begin{{bmatrix}} {H1[0][0]:.3f} & {H1[1][0]:.3f} \\\\ {H1[0][1]:.3f} & {H1[1][1]:.3f} \\end{{bmatrix}} = \\begin{{bmatrix}} {AHT_00:.3f} & {AHT_01:.3f} \\\\ {AHT_10:.3f} & {AHT_11:.3f} \\end{{bmatrix}}",
            "calculate_WHHT": f"W H H^T = \\begin{{bmatrix}} {W0[0][0]} & {W0[0][1]} \\\\ {W0[1][0]} & {W0[1][1]} \\end{{bmatrix}} \\begin{{bmatrix}} {H1[0][0]:.3f} & {H1[0][1]:.3f} \\\\ {H1[1][0]:.3f} & {H1[1][1]:.3f} \\end{{bmatrix}} \\begin{{bmatrix}} {H1[0][0]:.3f} & {H1[1][0]:.3f} \\\\ {H1[0][1]:.3f} & {H1[1][1]:.3f} \\end{{bmatrix}}",
            "update_W": f"W^{{(1)}} = W^{{(0)}} \\odot \\frac{{A H^T}}{{W H H^T}} = \\begin{{bmatrix}} {W0[0][0]} & {W0[0][1]} \\\\ {W0[1][0]} & {W0[1][1]} \\end{{bmatrix}} \\odot \\begin{{bmatrix}} \\frac{{{AHT_00:.3f}}}{{{WHHT_00:.3f}}} & \\frac{{{AHT_01:.3f}}}{{{WHHT_01:.3f}}} \\\\ \\frac{{{AHT_10:.3f}}}{{{WHHT_10:.3f}}} & \\frac{{{AHT_11:.3f}}}{{{WHHT_11:.3f}}} \\end{{bmatrix}} = \\begin{{bmatrix}} {W1[0][0]:.3f} & {W1[0][1]:.3f} \\\\ {W1[1][0]:.3f} & {W1[1][1]:.3f} \\end{{bmatrix}}",
        },
        "step4": {
            "title": "Compute Reconstruction Error",
            "description": "Calculate Frobenius norm: ||A - WH||_F",
            "calculate_WH": f"W^{{(1)}} H^{{(1)}} = \\begin{{bmatrix}} {W1[0][0]:.3f} & {W1[0][1]:.3f} \\\\ {W1[1][0]:.3f} & {W1[1][1]:.3f} \\end{{bmatrix}} \\begin{{bmatrix}} {H1[0][0]:.3f} & {H1[0][1]:.3f} \\\\ {H1[1][0]:.3f} & {H1[1][1]:.3f} \\end{{bmatrix}} = \\begin{{bmatrix}} {reconstruction[0][0]:.3f} & {reconstruction[0][1]:.3f} \\\\ {reconstruction[1][0]:.3f} & {reconstruction[1][1]:.3f} \\end{{bmatrix}}",
            "error_calculation": f"\\text{{Error}} = \\sqrt{{\\sum_{{i,j}} (A_{{ij}} - (WH)_{{ij}})^2}} = \\sqrt{{({A[0][0]} - {reconstruction[0][0]:.3f})^2 + ({A[0][1]} - {reconstruction[0][1]:.3f})^2 + ({A[1][0]} - {reconstruction[1][0]:.3f})^2 + ({A[1][1]} - {reconstruction[1][1]:.3f})^2}} = {error:.3f}",
        },
    }

    return jsonify(
        {
            "matrix_A": A.tolist(),
            "initial_W": W0.tolist(),
            "initial_H": H0.tolist(),
            "updated_W": W1.tolist(),
            "updated_H": H1.tolist(),
            "reconstruction": reconstruction.tolist(),
            "error": error,
            "detailed_steps": steps,
        }
    )


@app.route("/api/math")
def get_math_explanation():
    """Return mathematical explanation of SVD"""
    return jsonify(
        {
            "title": "Singular Value Decomposition (SVD)",
            "formula": "A = U Σ V^T",
            "explanation": {
                "overview": (
                    "SVD decomposes any matrix A into three matrices: "
                    "U (left singular vectors), Σ (singular values), "
                    "and V^T (right singular vectors transposed)."
                ),
                "matrices": {
                    "U": (
                        "U is an m×m orthogonal matrix containing left singular "
                        "vectors. Each column represents a direction in the "
                        "original data space."
                    ),
                    "Sigma": (
                        "Σ is an m×n diagonal matrix containing singular "
                        "values (σ₁ ≥ σ₂ ≥ ... ≥ σᵣ ≥ 0). These represent "
                        'the "importance" of each component.'
                    ),
                    "Vt": (
                        "V^T is an n×n orthogonal matrix containing right "
                        "singular vectors transposed. Each row represents a "
                        "direction in the feature space."
                    ),
                },
                "reconstruction": (
                    "To reconstruct the original matrix: "
                    "A ≈ Uₖ Σₖ Vₖ^T, where k is the number of "
                    "components used."
                ),
                "compression": (
                    "SVD enables compression by keeping only the most "
                    "important components (largest singular values)."
                ),
                "applications": [
                    "Dimensionality reduction",
                    "Data compression",
                    "Noise reduction",
                    "Principal Component Analysis (PCA)",
                    "Recommendation systems",
                    "Image processing",
                ],
            },
        }
    )


@app.route("/api/math-example")
def get_math_example():
    """Return detailed 2x2 SVD example with all calculations"""
    # Create a simple 2x2 example matrix
    A = np.array([[3, 1], [1, 3]])

    # Step 1: Compute A^T * A
    ATA = A.T @ A

    # Step 2: Find eigenvalues and eigenvectors of A^T * A
    eigenvals, eigenvecs = np.linalg.eigh(ATA)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    # Step 3: Compute singular values
    singular_values = np.sqrt(eigenvals)

    # Step 4: Compute V (right singular vectors)
    V = eigenvecs

    # Step 5: Compute U (left singular vectors)
    U = A @ V @ np.diag(1 / singular_values)

    # Step 6: Verify reconstruction
    reconstruction = U @ np.diag(singular_values) @ V.T

    # Calculate each element manually for detailed explanation
    ATA_00 = A[0][0] * A[0][0] + A[1][0] * A[1][0]  # 3*3 + 1*1 = 10
    ATA_01 = A[0][0] * A[0][1] + A[1][0] * A[1][1]  # 3*1 + 1*3 = 6
    ATA_10 = A[0][1] * A[0][0] + A[1][1] * A[1][0]  # 1*3 + 3*1 = 6
    ATA_11 = A[0][1] * A[0][1] + A[1][1] * A[1][1]  # 1*1 + 3*3 = 10

    # Calculate quadratic formula coefficients manually
    a = 1
    b = -(ATA[0][0] + ATA[1][1])  # -20
    c = ATA[0][0] * ATA[1][1] - ATA[0][1] * ATA[1][0]  # 100 - 36 = 64

    # Solve quadratic formula: λ = (-b ± √(b² - 4ac)) / 2a
    discriminant = b**2 - 4 * a * c
    sqrt_discriminant = np.sqrt(discriminant)
    lambda1 = (-b + sqrt_discriminant) / (2 * a)
    lambda2 = (-b - sqrt_discriminant) / (2 * a)

    # Detailed step-by-step calculations
    steps = {
        "step1": {
            "title": "Step 1: Compute A^T A",
            "description": "First, we compute the transpose of A and multiply it by A",
            "step1a": f"A^T = \\begin{{bmatrix}} {A[0][0]} & {A[1][0]} \\\\ {A[0][1]} & {A[1][1]} \\end{{bmatrix}} = \\begin{{bmatrix}} {A[0][0]} & {A[1][0]} \\\\ {A[0][1]} & {A[1][1]} \\end{{bmatrix}}",
            "step1b": f"A^T A = \\begin{{bmatrix}} {A[0][0]} & {A[1][0]} \\\\ {A[0][1]} & {A[1][1]} \\end{{bmatrix}} \\begin{{bmatrix}} {A[0][0]} & {A[0][1]} \\\\ {A[1][0]} & {A[1][1]} \\end{{bmatrix}}",
            "step1c": f"= \\begin{{bmatrix}} {A[0][0]} \\cdot {A[0][0]} + {A[1][0]} \\cdot {A[1][0]} & {A[0][0]} \\cdot {A[0][1]} + {A[1][0]} \\cdot {A[1][1]} \\\\ {A[0][1]} \\cdot {A[0][0]} + {A[1][1]} \\cdot {A[1][0]} & {A[0][1]} \\cdot {A[0][1]} + {A[1][1]} \\cdot {A[1][1]} \\end{{bmatrix}}",
            "step1d": f"= \\begin{{bmatrix}} {A[0][0]*A[0][0]} + {A[1][0]*A[1][0]} & {A[0][0]*A[0][1]} + {A[1][0]*A[1][1]} \\\\ {A[0][1]*A[0][0]} + {A[1][1]*A[1][0]} & {A[0][1]*A[0][1]} + {A[1][1]*A[1][1]} \\end{{bmatrix}}",
            "step1e": f"= \\begin{{bmatrix}} {ATA_00} & {ATA_01} \\\\ {ATA_10} & {ATA_11} \\end{{bmatrix}}",
        },
        "step2": {
            "title": "Step 2: Find Eigenvalues",
            "description": "Solve the characteristic equation det(A^T A - λI) = 0",
            "step2a": f"\\det(A^T A - \\lambda I) = \\det\\begin{{bmatrix}} {ATA[0][0]} - \\lambda & {ATA[0][1]} \\\\ {ATA[1][0]} & {ATA[1][1]} - \\lambda \\end{{bmatrix}} = 0",
            "step2b": f"({ATA[0][0]} - \\lambda)({ATA[1][1]} - \\lambda) - ({ATA[0][1]})({ATA[1][0]}) = 0",
            "step2c": f"({ATA[0][0]} - \\lambda)({ATA[1][1]} - \\lambda) - {ATA[0][1] * ATA[1][0]} = 0",
            "step2d": f"\\lambda^2 - ({ATA[0][0]} + {ATA[1][1]})\\lambda + ({ATA[0][0]} \\cdot {ATA[1][1]} - {ATA[0][1]} \\cdot {ATA[1][0]}) = 0",
            "step2e": f"\\lambda^2 - {ATA[0][0] + ATA[1][1]}\\lambda + {ATA[0][0] * ATA[1][1] - ATA[0][1] * ATA[1][0]} = 0",
            "step2f": f"\\lambda^2 - {b}\\lambda + {c} = 0",
            "step2g": f"\\text{{Using quadratic formula: }} \\lambda = \\frac{{-({b}) \\pm \\sqrt{{({b})^2 - 4 \\cdot {a} \\cdot {c}}}}}{{2 \\cdot {a}}}",
            "step2h": f"\\lambda = \\frac{{{b} \\pm \\sqrt{{{b**2} - {4*a*c}}}}}{{2}} = \\frac{{{b} \\pm \\sqrt{{{discriminant}}}}}{{2}}",
            "step2i": f"\\lambda_1 = \\frac{{{b} + {sqrt_discriminant:.4f}}}{{2}} = {lambda1:.4f}",
            "step2j": f"\\lambda_2 = \\frac{{{b} - {sqrt_discriminant:.4f}}}{{2}} = {lambda2:.4f}",
        },
        "step3": {
            "title": "Find Eigenvectors",
            "description": "For each eigenvalue, solve (A^T A - λI)v = 0",
            "eigenvector1": {
                "equation": f"(A^T A - \\lambda_1 I)v_1 = 0",
                "matrix": f"\\begin{{bmatrix}} {ATA[0][0]} - {eigenvals[0]:.4f} & {ATA[0][1]} \\\\ {ATA[1][0]} & {ATA[1][1]} - {eigenvals[0]:.4f} \\end{{bmatrix}} \\begin{{bmatrix}} x \\\\ y \\end{{bmatrix}} = \\begin{{bmatrix}} 0 \\\\ 0 \\end{{bmatrix}}",
                "solution": f"v_1 = \\begin{{bmatrix}} {eigenvecs[0][0]:.4f} \\\\ {eigenvecs[1][0]:.4f} \\end{{bmatrix}}",
            },
            "eigenvector2": {
                "equation": f"(A^T A - \\lambda_2 I)v_2 = 0",
                "matrix": f"\\begin{{bmatrix}} {ATA[0][0]} - {eigenvals[1]:.4f} & {ATA[0][1]} \\\\ {ATA[1][0]} & {ATA[1][1]} - {eigenvals[1]:.4f} \\end{{bmatrix}} \\begin{{bmatrix}} x \\\\ y \\end{{bmatrix}} = \\begin{{bmatrix}} 0 \\\\ 0 \\end{{bmatrix}}",
                "solution": f"v_2 = \\begin{{bmatrix}} {eigenvecs[0][1]:.4f} \\\\ {eigenvecs[1][1]:.4f} \\end{{bmatrix}}",
            },
        },
        "step4": {
            "title": "Compute Singular Values",
            "description": "Singular values are the square roots of eigenvalues",
            "calculation": f"\\sigma_1 = \\sqrt{{\\lambda_1}} = \\sqrt{{{eigenvals[0]:.4f}}} = {singular_values[0]:.4f}",
            "calculation2": f"\\sigma_2 = \\sqrt{{\\lambda_2}} = \\sqrt{{{eigenvals[1]:.4f}}} = {singular_values[1]:.4f}",
            "sigma_matrix": f"\\Sigma = \\begin{{bmatrix}} {singular_values[0]:.4f} & 0 \\\\ 0 & {singular_values[1]:.4f} \\end{{bmatrix}}",
        },
        "step5": {
            "title": "Construct V Matrix",
            "description": "V matrix contains the eigenvectors as columns",
            "calculation": f"V = \\begin{{bmatrix}} {eigenvecs[0][0]:.4f} & {eigenvecs[0][1]:.4f} \\\\ {eigenvecs[1][0]:.4f} & {eigenvecs[1][1]:.4f} \\end{{bmatrix}}",
        },
        "step6": {
            "title": "Construct U Matrix",
            "description": "U = A V Σ^(-1)",
            "calculation": f"U = A V \\Sigma^{{-1}} = \\begin{{bmatrix}} {A[0][0]} & {A[0][1]} \\\\ {A[1][0]} & {A[1][1]} \\end{{bmatrix}} \\begin{{bmatrix}} {eigenvecs[0][0]:.4f} & {eigenvecs[0][1]:.4f} \\\\ {eigenvecs[1][0]:.4f} & {eigenvecs[1][1]:.4f} \\end{{bmatrix}} \\begin{{bmatrix}} {1/singular_values[0]:.4f} & 0 \\\\ 0 & {1/singular_values[1]:.4f} \\end{{bmatrix}}",
            "result": f"U = \\begin{{bmatrix}} {U[0][0]:.4f} & {U[0][1]:.4f} \\\\ {U[1][0]:.4f} & {U[1][1]:.4f} \\end{{bmatrix}}",
        },
        "step7": {
            "title": "Verify Reconstruction",
            "description": "Check that A = U Σ V^T",
            "calculation": f"A = U \\Sigma V^T = \\begin{{bmatrix}} {U[0][0]:.4f} & {U[0][1]:.4f} \\\\ {U[1][0]:.4f} & {U[1][1]:.4f} \\end{{bmatrix}} \\begin{{bmatrix}} {singular_values[0]:.4f} & 0 \\\\ 0 & {singular_values[1]:.4f} \\end{{bmatrix}} \\begin{{bmatrix}} {eigenvecs[0][0]:.4f} & {eigenvecs[1][0]:.4f} \\\\ {eigenvecs[0][1]:.4f} & {eigenvecs[1][1]:.4f} \\end{{bmatrix}}",
            "result": f"= \\begin{{bmatrix}} {reconstruction[0][0]:.4f} & {reconstruction[0][1]:.4f} \\\\ {reconstruction[1][0]:.4f} & {reconstruction[1][1]:.4f} \\end{{bmatrix}}",
            "error": f"\\text{{Error}} = ||A - U\\Sigma V^T|| = {np.linalg.norm(A - reconstruction):.10f}",
        },
    }

    return jsonify(
        {
            "matrix_A": A.tolist(),
            "matrix_ATA": ATA.tolist(),
            "eigenvalues": eigenvals.tolist(),
            "eigenvectors": eigenvecs.tolist(),
            "singular_values": singular_values.tolist(),
            "matrix_V": V.tolist(),
            "matrix_U": U.tolist(),
            "reconstruction": reconstruction.tolist(),
            "reconstruction_error": np.linalg.norm(A - reconstruction),
            "detailed_steps": steps,
        }
    )


@app.route("/api/pca-visualization")
def get_pca_visualization():
    """Generate PCA visualization on bars dataset"""
    try:
        # Load data
        data = load_bars_data()
        if data is None:
            return jsonify({"error": "Could not load bars dataset"}), 500

        # Apply PCA with different numbers of components
        pca_results = {}

        for n_components in [1, 2, 3, 5, 10]:
            pca = PCA(n_components=n_components)
            pca_data = pca.fit_transform(data)

            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"PCA with {n_components} Components", fontsize=16)

            # Original data (first 4 samples)
            for i in range(min(4, len(data))):
                row = i // 2
                col = i % 2
                axes[row, col].imshow(data[i].reshape(8, 8), cmap="gray")
                axes[row, col].set_title(f"Original Sample {i+1}")
                axes[row, col].axis("off")

            plt.tight_layout()

            # Convert to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()

            pca_results[f"{n_components}_components"] = {
                "image": img_base64,
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "cumulative_variance": np.cumsum(
                    pca.explained_variance_ratio_
                ).tolist(),
                "total_variance_explained": np.sum(
                    pca.explained_variance_ratio_
                ),
            }

        return jsonify(pca_results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/tsne-visualization")
def get_tsne_visualization():
    """Generate t-SNE visualization on bars dataset"""
    try:
        # Load data
        data = load_bars_data()
        if data is None:
            return jsonify({"error": "Could not load bars dataset"}), 500

        # Take a subset for t-SNE (it's computationally expensive)
        subset_size = min(200, len(data))
        data_subset = data[:subset_size]

        # Apply t-SNE with different perplexity values
        tsne_results = {}

        perplexities = [5, 15, 30, 50]

        for perplexity in perplexities:
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            tsne_data = tsne.fit_transform(data_subset)

            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Scatter plot of t-SNE embedding
            scatter = ax1.scatter(
                tsne_data[:, 0],
                tsne_data[:, 1],
                c=range(len(tsne_data)),
                cmap="viridis",
                alpha=0.7,
            )
            ax1.set_title(f"t-SNE Embedding (Perplexity = {perplexity})")
            ax1.set_xlabel("t-SNE Component 1")
            ax1.set_ylabel("t-SNE Component 2")
            plt.colorbar(scatter, ax=ax1)

            # Show some original samples
            sample_indices = np.linspace(0, len(data_subset) - 1, 8, dtype=int)
            for i, idx in enumerate(sample_indices):
                row = i // 4
                col = i % 4
                if i < 4:
                    ax2.imshow(data_subset[idx].reshape(8, 8), cmap="gray")
                    ax2.set_title(f"Sample {idx+1}")
                    ax2.axis("off")

            plt.tight_layout()

            # Convert to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()

            tsne_results[f"perplexity_{perplexity}"] = {
                "image": img_base64,
                "perplexity": perplexity,
                "n_samples": subset_size,
            }

        return jsonify(tsne_results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/pca-animation")
def get_pca_animation():
    """Generate animated PCA visualization showing A = UΣV^T matrices"""
    try:
        # Load data
        data = load_bars_data()
        if data is None:
            return jsonify({"error": "Could not load bars dataset"}), 500

        # Take first 4 samples for visualization
        sample_data = data[:4]

        # Apply PCA with maximum components
        pca_full = PCA(n_components=min(64, data.shape[1]))
        pca_full.fit(data)

        # Get the full matrices
        U_full = pca_full.components_.T  # Principal components (eigenvectors)
        sigma_full = pca_full.singular_values_  # Singular values
        V_full = (
            pca_full.components_
        )  # Principal components (same as U for PCA)

        # Create animation frames for different numbers of components
        animation_frames = []

        for n_components in range(1, min(21, len(sigma_full)) + 1):
            # Get subset of matrices
            U_k = U_full[:, :n_components]
            sigma_k = sigma_full[:n_components]
            V_k = V_full[:n_components, :]

            # Reconstruct the data
            reconstructed = U_k @ np.diag(sigma_k) @ V_k

            # Create visualization
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.suptitle(
                f"PCA Animation: A = UΣV^T (k={n_components} components)",
                fontsize=16,
                fontweight="bold",
            )

            # Original data A (first row)
            for i in range(4):
                axes[0, i].imshow(
                    sample_data[i].reshape(8, 8), cmap="gray", vmin=0, vmax=1
                )
                axes[0, i].set_title(f"A[{i}] (Original)", fontsize=12)
                axes[0, i].axis("off")

            # Reconstructed data A_reconstructed (second row)
            for i in range(4):
                axes[1, i].imshow(
                    reconstructed[i].reshape(8, 8), cmap="gray", vmin=0, vmax=1
                )
                axes[1, i].set_title(f"A_reconstructed[{i}]", fontsize=12)
                axes[1, i].axis("off")

            # Add matrix visualization on the right
            fig.text(
                0.85,
                0.7,
                f"U (k×k):\n{sigma_k.shape[0]}×{sigma_k.shape[0]}",
                fontsize=10,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
            )
            fig.text(
                0.85,
                0.5,
                f"Σ (k×k):\n{sigma_k.shape[0]}×{sigma_k.shape[0]}",
                fontsize=10,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"),
            )
            fig.text(
                0.85,
                0.3,
                f"V^T (k×d):\n{sigma_k.shape[0]}×64",
                fontsize=10,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"),
            )

            plt.tight_layout()

            # Convert to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()

            # Calculate statistics
            mse = np.mean((sample_data - reconstructed) ** 2)
            explained_variance = np.sum(
                pca_full.explained_variance_ratio_[:n_components]
            )

            animation_frames.append(
                {
                    "frame": n_components,
                    "image": img_base64,
                    "n_components": n_components,
                    "mse": float(mse),
                    "explained_variance": float(explained_variance),
                    "singular_values": sigma_k.tolist(),
                }
            )

        return jsonify(
            {
                "frames": animation_frames,
                "total_frames": len(animation_frames),
                "max_components": len(sigma_full),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/svd-animation", methods=["GET", "POST"])
def get_svd_animation():
    """Generate animated SVD visualization showing A = UΣV^T matrices"""
    try:
        # Get parameters from request
        if request.method == "POST":
            params = request.get_json()
            max_components = params.get("max_components", 16)
            animation_steps = params.get("animation_steps", 16)
        else:
            # For GET requests, use all 64 components
            max_components = 64
            animation_steps = 64

        # Load data
        data = load_bars_data()
        if data is None:
            return jsonify({"error": "Could not load bars dataset"}), 500

        # Use full dataset for SVD animation to allow all 64 components
        svd_sample_data = data[:4]  # For SVD visualization only

        # Apply SVD with maximum components on the full dataset
        U_full, sigma_full, Vt_full = np.linalg.svd(data, full_matrices=False)

        # Create animation frames for different numbers of components
        animation_frames = []

        # Calculate component values based on steps
        max_possible = min(max_components, len(sigma_full))
        component_values = np.linspace(
            1, max_possible, animation_steps, dtype=int
        )
        component_values = np.unique(component_values)  # Remove duplicates
        component_values = component_values.astype(
            int
        )  # Ensure Python int type

        for n_components in component_values:
            # Get subset of matrices
            U_k = U_full[:, :n_components]
            sigma_k = sigma_full[:n_components]
            Vt_k = Vt_full[:n_components, :]

            # Reconstruct the data
            reconstructed = U_k @ np.diag(sigma_k) @ Vt_k

            # For visualization, use sample data
            svd_sample_reconstructed = (
                reconstructed[:4]
                if reconstructed.shape[0] > 4
                else reconstructed
            )

            # Create visualization showing A = UΣV^T matrices + reconstruction
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))
            fig.suptitle(
                f"SVD Animation: A = UΣV^T (k={n_components} components)",
                fontsize=16,
                fontweight="bold",
            )

            # Matrix A (Original) - show first sample
            axes[0].imshow(
                svd_sample_data[0].reshape(8, 8), cmap="gray", vmin=0, vmax=1
            )
            axes[0].set_title(
                f"A (Original)\n8×8", fontsize=12, fontweight="bold"
            )
            axes[0].axis("off")

            # Matrix U (Left singular vectors) - show first 4 rows
            U_display = U_k[:4] if U_k.shape[0] > 4 else U_k
            axes[1].imshow(U_display, cmap="Blues", aspect="auto")
            axes[1].set_title(
                f"U (Left singular vectors)\n{U_display.shape[0]}×{U_display.shape[1]}",
                fontsize=12,
                fontweight="bold",
                color="blue",
            )
            axes[1].axis("off")

            # Matrix Σ (Singular values) - show as diagonal matrix
            sigma_matrix = np.zeros((n_components, n_components))
            np.fill_diagonal(sigma_matrix, sigma_k)
            im = axes[2].imshow(sigma_matrix, cmap="Greens", aspect="auto")
            axes[2].set_title(
                f"Σ (Singular values)\n{n_components}×{n_components}",
                fontsize=12,
                fontweight="bold",
                color="green",
            )
            axes[2].axis("off")

            # Add singular values as text
            for i in range(min(n_components, 8)):  # Show max 8 values
                axes[2].text(
                    i,
                    i,
                    f"{sigma_k[i]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )

            # Matrix V^T (Right singular vectors transposed)
            axes[3].imshow(Vt_k, cmap="Reds", aspect="auto")
            axes[3].set_title(
                f"V^T (Right singular vectors)\n{Vt_k.shape[0]}×{Vt_k.shape[1]}",
                fontsize=12,
                fontweight="bold",
                color="red",
            )
            axes[3].axis("off")

            # Matrix A_reconstructed (Reconstruction)
            # Ensure values are in [0,1] range for proper visualization
            recon_values = svd_sample_reconstructed[0].reshape(8, 8)
            recon_values = np.clip(recon_values, 0, 1)  # Clip to [0,1] range

            axes[4].imshow(recon_values, cmap="gray", vmin=0, vmax=1)
            axes[4].set_title(
                f"A_reconstructed\n8×8 (k={n_components})",
                fontsize=12,
                fontweight="bold",
                color="purple",
            )
            axes[4].axis("off")

            plt.tight_layout()

            # Convert to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()

            # Calculate statistics
            mse = np.mean((svd_sample_data - svd_sample_reconstructed) ** 2)
            explained_variance = np.sum(sigma_k**2) / np.sum(sigma_full**2)

            animation_frames.append(
                {
                    "frame": int(n_components),
                    "image": img_base64,
                    "n_components": int(n_components),
                    "mse": float(mse),
                    "explained_variance": float(explained_variance),
                    "singular_values": sigma_k.tolist(),
                }
            )

        return jsonify(
            {
                "frames": animation_frames,
                "total_frames": len(animation_frames),
                "max_components": max_possible,
                "animation_steps": animation_steps,
                "component_values": [int(x) for x in component_values.tolist()],
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/nnmf-animation", methods=["GET", "POST"])
def get_nnmf_animation():
    """Generate animated NNMF visualization showing A ≈ W×H matrices"""
    try:
        # Get parameters from request
        if request.method == "POST":
            params = request.get_json()
            max_components = params.get("max_components", 16)
            animation_steps = params.get("animation_steps", 16)
        else:
            max_components = 16
            animation_steps = 16

        # Load data
        data = load_bars_data()
        if data is None:
            return jsonify({"error": "Could not load bars dataset"}), 500

        # Use separate sample data for NNMF
        nnmf_sample_data = data[:4]  # For NNMF visualization only

        # Create animation frames for different numbers of components
        animation_frames = []

        # Calculate component values based on steps
        # For NNMF, we can use more components than samples by using the full dataset
        if max_components > nnmf_sample_data.shape[0]:
            # Use the full dataset for NNMF to allow more components
            full_data = data
            max_possible = min(
                max_components, min(full_data.shape[0], full_data.shape[1])
            )
        else:
            max_possible = min(
                max_components,
                min(nnmf_sample_data.shape[0], nnmf_sample_data.shape[1]),
            )

        component_values = np.linspace(
            1, max_possible, animation_steps, dtype=int
        )
        component_values = np.unique(component_values)  # Remove duplicates
        component_values = component_values.astype(
            int
        )  # Ensure Python int type

        for n_components in component_values:
            # Apply NNMF
            # Use full dataset if we need more components than samples
            if max_components > nnmf_sample_data.shape[0]:
                nmf_data = full_data
            else:
                nmf_data = nnmf_sample_data

            nmf = NMF(
                n_components=n_components,
                max_iter=500,
                random_state=42,
                init="nndsvda",
                solver="mu",
                beta_loss="frobenius",
            )
            W = nmf.fit_transform(nmf_data)
            H = nmf.components_

            # Reconstruct the data
            reconstructed = W @ H

            # For visualization, use sample data
            nnmf_sample_reconstructed = (
                reconstructed[:4]
                if reconstructed.shape[0] > 4
                else reconstructed
            )

            # Create visualization showing A ≈ W×H matrices + reconstruction
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))
            fig.suptitle(
                f"NNMF Animation: A ≈ W×H (k={n_components} components)",
                fontsize=16,
                fontweight="bold",
            )

            # Matrix A (Original) - show first sample
            axes[0].imshow(
                nnmf_sample_data[0].reshape(8, 8), cmap="gray", vmin=0, vmax=1
            )
            axes[0].set_title(
                f"A (Original)\n8×8", fontsize=12, fontweight="bold"
            )
            axes[0].axis("off")

            # Matrix W (Coefficient matrix) - show only first 4 rows for visualization
            W_display = W[:4] if W.shape[0] > 4 else W
            axes[1].imshow(W_display, cmap="Blues", aspect="auto")
            axes[1].set_title(
                f"W (Coefficients)\n{W_display.shape[0]}×{W_display.shape[1]}",
                fontsize=12,
                fontweight="bold",
                color="blue",
            )
            axes[1].axis("off")

            # Matrix H (Basis matrix)
            axes[2].imshow(H, cmap="Greens", aspect="auto")
            axes[2].set_title(
                f"H (Basis)\n{H.shape[0]}×{H.shape[1]}",
                fontsize=12,
                fontweight="bold",
                color="green",
            )
            axes[2].axis("off")

            # Matrix W×H (Product) - show only first 4 rows for visualization
            product_matrix = W @ H
            product_display = (
                product_matrix[:4]
                if product_matrix.shape[0] > 4
                else product_matrix
            )
            axes[3].imshow(product_display, cmap="Reds", aspect="auto")
            axes[3].set_title(
                f"W×H (Product)\n{product_display.shape[0]}×{product_display.shape[1]}",
                fontsize=12,
                fontweight="bold",
                color="red",
            )
            axes[3].axis("off")

            # Matrix A_reconstructed (Reconstruction)
            # Ensure values are in [0,1] range for proper visualization
            recon_values = nnmf_sample_reconstructed[0].reshape(8, 8)
            recon_values = np.clip(recon_values, 0, 1)  # Clip to [0,1] range

            axes[4].imshow(recon_values, cmap="gray", vmin=0, vmax=1)
            axes[4].set_title(
                f"A_reconstructed\n8×8 (k={n_components})",
                fontsize=12,
                fontweight="bold",
                color="purple",
            )
            axes[4].axis("off")

            plt.tight_layout()

            # Convert to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()

            # Calculate statistics
            reconstruction_error = np.linalg.norm(
                nnmf_sample_data - nnmf_sample_reconstructed, "fro"
            )
            explained_variance = (
                np.linalg.norm(nnmf_sample_reconstructed, "fro") ** 2
                / np.linalg.norm(nnmf_sample_data, "fro") ** 2
            )

            animation_frames.append(
                {
                    "frame": int(n_components),
                    "image": img_base64,
                    "n_components": int(n_components),
                    "reconstruction_error": float(reconstruction_error),
                    "explained_variance": float(explained_variance),
                    "iterations_converged": int(nmf.n_iter_),
                }
            )

        return jsonify(
            {
                "frames": animation_frames,
                "total_frames": len(animation_frames),
                "max_components": max_possible,
                "animation_steps": animation_steps,
                "component_values": [int(x) for x in component_values.tolist()],
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/pca-random-example")
def get_pca_random_example():
    """Generate PCA visualization on random dataset"""
    try:
        # Generate random dataset
        np.random.seed(42)  # For reproducibility
        n_samples = 100
        n_features = 20

        # Create correlated data
        X = np.random.randn(n_samples, n_features)
        # Add some correlation structure
        X[:, 1] = X[:, 0] + 0.5 * np.random.randn(n_samples)
        X[:, 2] = X[:, 0] - 0.3 * X[:, 1] + 0.2 * np.random.randn(n_samples)

        # Apply PCA with different numbers of components
        pca_results = {}

        for n_components in [1, 2, 3, 5, 10]:
            pca = PCA(n_components=n_components)
            pca_data = pca.fit_transform(X)

            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"PCA with {n_components} Components", fontsize=16)

            # Original data scatter plots
            axes[0, 0].scatter(X[:, 0], X[:, 1], alpha=0.6, c="blue")
            axes[0, 0].set_title("Original Data: Feature 1 vs Feature 2")
            axes[0, 0].set_xlabel("Feature 1")
            axes[0, 0].set_ylabel("Feature 2")

            axes[0, 1].scatter(X[:, 0], X[:, 2], alpha=0.6, c="green")
            axes[0, 1].set_title("Original Data: Feature 1 vs Feature 3")
            axes[0, 1].set_xlabel("Feature 1")
            axes[0, 1].set_ylabel("Feature 3")

            # PCA transformed data
            if n_components >= 2:
                axes[1, 0].scatter(
                    pca_data[:, 0], pca_data[:, 1], alpha=0.6, c="red"
                )
                axes[1, 0].set_title(f"PCA Data: PC1 vs PC2")
                axes[1, 0].set_xlabel("PC1")
                axes[1, 0].set_ylabel("PC2")
            else:
                axes[1, 0].scatter(
                    pca_data[:, 0],
                    np.zeros_like(pca_data[:, 0]),
                    alpha=0.6,
                    c="red",
                )
                axes[1, 0].set_title(f"PCA Data: PC1")
                axes[1, 0].set_xlabel("PC1")
                axes[1, 0].set_ylabel("PC2 (zero)")

            # Explained variance plot
            axes[1, 1].bar(
                range(1, n_components + 1),
                pca.explained_variance_ratio_[:n_components],
            )
            axes[1, 1].set_title("Explained Variance Ratio")
            axes[1, 1].set_xlabel("Principal Component")
            axes[1, 1].set_ylabel("Explained Variance Ratio")

            plt.tight_layout()

            # Convert to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()

            pca_results[f"{n_components}_components"] = {
                "image": img_base64,
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "cumulative_variance": np.cumsum(
                    pca.explained_variance_ratio_
                ).tolist(),
                "total_variance_explained": np.sum(
                    pca.explained_variance_ratio_
                ),
            }

        return jsonify(pca_results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/tsne-random-example")
def get_tsne_random_example():
    """Generate t-SNE visualization on random dataset"""
    try:
        # Generate random dataset with clusters
        np.random.seed(42)  # For reproducibility
        n_samples = 150

        # Create 3 clusters
        cluster1 = np.random.multivariate_normal(
            [2, 2], [[0.5, 0.2], [0.2, 0.5]], 50
        )
        cluster2 = np.random.multivariate_normal(
            [-2, -2], [[0.5, -0.2], [-0.2, 0.5]], 50
        )
        cluster3 = np.random.multivariate_normal(
            [0, 3], [[0.3, 0], [0, 0.3]], 50
        )

        # Combine clusters
        X = np.vstack([cluster1, cluster2, cluster3])
        labels = np.hstack([np.zeros(50), np.ones(50), np.full(50, 2)])

        # Add noise dimensions
        noise = np.random.randn(n_samples, 8)
        X = np.hstack([X, noise])

        # Apply t-SNE with different perplexity values
        tsne_results = {}

        perplexities = [5, 15, 30, 50]

        for perplexity in perplexities:
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            tsne_data = tsne.fit_transform(X)

            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Original data (first 2 dimensions)
            scatter1 = ax1.scatter(
                X[:, 0], X[:, 1], c=labels, cmap="viridis", alpha=0.7
            )
            ax1.set_title(f"Original Data (Perplexity = {perplexity})")
            ax1.set_xlabel("Feature 1")
            ax1.set_ylabel("Feature 2")
            plt.colorbar(scatter1, ax=ax1)

            # t-SNE embedding
            scatter2 = ax2.scatter(
                tsne_data[:, 0],
                tsne_data[:, 1],
                c=labels,
                cmap="viridis",
                alpha=0.7,
            )
            ax2.set_title(f"t-SNE Embedding (Perplexity = {perplexity})")
            ax2.set_xlabel("t-SNE Component 1")
            ax2.set_ylabel("t-SNE Component 2")
            plt.colorbar(scatter2, ax=ax2)

            plt.tight_layout()

            # Convert to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()

            tsne_results[f"perplexity_{perplexity}"] = {
                "image": img_base64,
                "perplexity": perplexity,
                "n_samples": n_samples,
            }

        return jsonify(tsne_results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
