import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs, make_circles
import matplotlib.patches as patches


class ClusteringVisualizer:
    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 12))
        self.fig.suptitle(
            "K-means and DBSCAN Clustering Visualization", fontsize=16
        )

        # Generate sample data
        self.generate_data()

        # K-means variables
        self.kmeans_iteration = 0
        self.kmeans_centroids = None
        self.kmeans_labels = None
        self.kmeans_history = []

        # DBSCAN variables
        self.dbscan_eps = 0.5
        self.dbscan_min_samples = 5

        # Colors for visualization
        self.colors = [
            "red",
            "blue",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
        ]

        self.setup_plots()
        self.setup_buttons()

    def generate_data(self):
        """Generate synthetic 2D datasets"""
        # For K-means: well-separated clusters
        self.kmeans_data, _ = make_blobs(
            n_samples=300,
            centers=4,
            cluster_std=1.5,
            random_state=42,
            center_box=(-10, 10),
        )

        # For DBSCAN: more complex structure
        self.dbscan_data, _ = make_circles(
            n_samples=200, noise=0.1, factor=0.3, random_state=42
        )
        self.dbscan_data *= 5  # Scale up for better visualization

    def setup_plots(self):
        """Setup the four subplots"""
        # K-means plots
        self.ax_kmeans = self.axes[0, 0]
        self.ax_kmeans.set_title("K-means: Data Points")
        self.ax_kmeans.scatter(
            self.kmeans_data[:, 0],
            self.kmeans_data[:, 1],
            c="lightblue",
            alpha=0.6,
            s=50,
        )
        self.ax_kmeans.set_xlabel("X")
        self.ax_kmeans.set_ylabel("Y")
        self.ax_kmeans.grid(True, alpha=0.3)

        self.ax_kmeans_iter = self.axes[0, 1]
        self.ax_kmeans_iter.set_title("K-means: Iterations")
        self.ax_kmeans_iter.set_xlabel("X")
        self.ax_kmeans_iter.set_ylabel("Y")
        self.ax_kmeans_iter.grid(True, alpha=0.3)

        # DBSCAN plots
        self.ax_dbscan = self.axes[1, 0]
        self.ax_dbscan.set_title("DBSCAN: Data Points")
        self.ax_dbscan.scatter(
            self.dbscan_data[:, 0],
            self.dbscan_data[:, 1],
            c="lightblue",
            alpha=0.6,
            s=50,
        )
        self.ax_dbscan.set_xlabel("X")
        self.ax_dbscan.set_ylabel("Y")
        self.ax_dbscan.grid(True, alpha=0.3)

        self.ax_dbscan_result = self.axes[1, 1]
        self.ax_dbscan_result.set_title("DBSCAN: Clustering Result")
        self.ax_dbscan_result.set_xlabel("X")
        self.ax_dbscan_result.set_ylabel("Y")
        self.ax_dbscan_result.grid(True, alpha=0.3)

    def setup_buttons(self):
        """Setup interactive buttons"""
        # K-means buttons
        ax_kmeans_btn = plt.axes([0.02, 0.7, 0.1, 0.04])
        self.btn_kmeans_init = Button(ax_kmeans_btn, "Init K-means")
        self.btn_kmeans_init.on_clicked(self.init_kmeans)

        ax_kmeans_next = plt.axes([0.13, 0.7, 0.1, 0.04])
        self.btn_kmeans_next = Button(ax_kmeans_next, "Next Iteration")
        self.btn_kmeans_next.on_clicked(self.next_kmeans_iteration)

        ax_kmeans_reset = plt.axes([0.24, 0.7, 0.1, 0.04])
        self.btn_kmeans_reset = Button(ax_kmeans_reset, "Reset")
        self.btn_kmeans_reset.on_clicked(self.reset_kmeans)

        # DBSCAN buttons
        ax_dbscan_btn = plt.axes([0.02, 0.3, 0.1, 0.04])
        self.btn_dbscan_run = Button(ax_dbscan_btn, "Run DBSCAN")
        self.btn_dbscan_run.on_clicked(self.run_dbscan)

        ax_dbscan_eps_plus = plt.axes([0.13, 0.3, 0.05, 0.04])
        self.btn_eps_plus = Button(ax_dbscan_eps_plus, "ε+")
        self.btn_eps_plus.on_clicked(self.increase_eps)

        ax_dbscan_eps_minus = plt.axes([0.19, 0.3, 0.05, 0.04])
        self.btn_eps_minus = Button(ax_dbscan_eps_minus, "ε-")
        self.btn_eps_minus.on_clicked(self.decrease_eps)

        # Parameter display
        self.eps_text = self.fig.text(
            0.25, 0.32, f"ε = {self.dbscan_eps:.2f}", fontsize=10
        )

    def init_kmeans(self, event):
        """Initialize K-means with random centroids"""
        k = 4
        # Random initialization
        min_x, max_x = (
            self.kmeans_data[:, 0].min(),
            self.kmeans_data[:, 0].max(),
        )
        min_y, max_y = (
            self.kmeans_data[:, 1].min(),
            self.kmeans_data[:, 1].max(),
        )

        self.kmeans_centroids = np.random.uniform(
            low=[min_x, min_y], high=[max_x, max_y], size=(k, 2)
        )

        self.kmeans_iteration = 0
        self.kmeans_history = [self.kmeans_centroids.copy()]

        self.update_kmeans_plot()

    def next_kmeans_iteration(self, event):
        """Perform next K-means iteration"""
        if self.kmeans_centroids is None:
            return

        # Assign points to nearest centroid
        distances = np.sqrt(
            (
                (
                    self.kmeans_data[:, np.newaxis, :]
                    - self.kmeans_centroids[np.newaxis, :, :]
                )
                ** 2
            ).sum(axis=2)
        )
        self.kmeans_labels = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.array(
            [
                self.kmeans_data[self.kmeans_labels == i].mean(axis=0)
                for i in range(len(self.kmeans_centroids))
            ]
        )

        self.kmeans_centroids = new_centroids
        self.kmeans_iteration += 1
        self.kmeans_history.append(self.kmeans_centroids.copy())

        self.update_kmeans_plot()

    def reset_kmeans(self, event):
        """Reset K-means visualization"""
        self.kmeans_iteration = 0
        self.kmeans_centroids = None
        self.kmeans_labels = None
        self.kmeans_history = []

        self.ax_kmeans_iter.clear()
        self.ax_kmeans_iter.set_title("K-means: Iterations")
        self.ax_kmeans_iter.set_xlabel("X")
        self.ax_kmeans_iter.set_ylabel("Y")
        self.ax_kmeans_iter.grid(True, alpha=0.3)

    def update_kmeans_plot(self):
        """Update K-means visualization"""
        self.ax_kmeans_iter.clear()
        title = f"K-means: Iteration {self.kmeans_iteration}"
        self.ax_kmeans_iter.set_title(title)
        self.ax_kmeans_iter.set_xlabel("X")
        self.ax_kmeans_iter.set_ylabel("Y")
        self.ax_kmeans_iter.grid(True, alpha=0.3)

        # Plot data points with colors based on current assignment
        if self.kmeans_labels is not None:
            for i in range(len(self.kmeans_centroids)):
                mask = self.kmeans_labels == i
                if np.any(mask):
                    self.ax_kmeans_iter.scatter(
                        self.kmeans_data[mask, 0],
                        self.kmeans_data[mask, 1],
                        c=self.colors[i % len(self.colors)],
                        alpha=0.6,
                        s=50,
                        label=f"Cluster {i+1}",
                    )
        else:
            self.ax_kmeans_iter.scatter(
                self.kmeans_data[:, 0],
                self.kmeans_data[:, 1],
                c="lightblue",
                alpha=0.6,
                s=50,
            )

        # Plot centroids
        for i, centroid in enumerate(self.kmeans_centroids):
            self.ax_kmeans_iter.scatter(
                centroid[0],
                centroid[1],
                c=self.colors[i % len(self.colors)],
                marker="x",
                s=200,
                linewidths=3,
                label=f"Centroid {i+1}",
            )

        # Draw lines from points to centroids
        if self.kmeans_labels is not None:
            for i, point in enumerate(self.kmeans_data):
                centroid = self.kmeans_centroids[self.kmeans_labels[i]]
                self.ax_kmeans_iter.plot(
                    [point[0], centroid[0]],
                    [point[1], centroid[1]],
                    color=self.colors[self.kmeans_labels[i] % len(self.colors)],
                    alpha=0.3,
                    linewidth=0.5,
                )

        self.ax_kmeans_iter.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        self.fig.canvas.draw()

    def increase_eps(self, event):
        """Increase DBSCAN epsilon parameter"""
        self.dbscan_eps = min(self.dbscan_eps + 0.1, 2.0)
        self.eps_text.set_text(f"ε = {self.dbscan_eps:.2f}")
        self.fig.canvas.draw()

    def decrease_eps(self, event):
        """Decrease DBSCAN epsilon parameter"""
        self.dbscan_eps = max(self.dbscan_eps - 0.1, 0.1)
        self.eps_text.set_text(f"ε = {self.dbscan_eps:.2f}")
        self.fig.canvas.draw()

    def run_dbscan(self, event):
        """Run DBSCAN clustering and visualize"""
        # Clear previous result
        self.ax_dbscan_result.clear()
        self.ax_dbscan_result.set_title("DBSCAN: Clustering Result")
        self.ax_dbscan_result.set_xlabel("X")
        self.ax_dbscan_result.set_ylabel("Y")
        self.ax_dbscan_result.grid(True, alpha=0.3)

        # Run DBSCAN
        dbscan = DBSCAN(
            eps=self.dbscan_eps, min_samples=self.dbscan_min_samples
        )
        labels = dbscan.fit_predict(self.dbscan_data)

        # Visualize epsilon circles around each point
        for i, point in enumerate(self.dbscan_data):
            circle = patches.Circle(
                point,
                self.dbscan_eps,
                fill=False,
                color="gray",
                alpha=0.3,
                linestyle="--",
            )
            self.ax_dbscan_result.add_patch(circle)

        # Color points based on cluster assignment
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)

        # Plot clustered points
        for k in unique_labels:
            if k == -1:  # Noise points
                mask = labels == k
                self.ax_dbscan_result.scatter(
                    self.dbscan_data[mask, 0],
                    self.dbscan_data[mask, 1],
                    c="black",
                    marker="x",
                    s=50,
                    alpha=0.8,
                    label="Noise",
                )
            else:
                mask = labels == k
                self.ax_dbscan_result.scatter(
                    self.dbscan_data[mask, 0],
                    self.dbscan_data[mask, 1],
                    c=self.colors[k % len(self.colors)],
                    s=50,
                    alpha=0.8,
                    label=f"Cluster {k+1}",
                )

        # Add statistics text
        stats_text = f"Clusters: {n_clusters}, Noise: {n_noise}"
        self.ax_dbscan_result.text(
            0.02,
            0.98,
            stats_text,
            transform=self.ax_dbscan_result.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat"),
        )

        self.ax_dbscan_result.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        self.fig.canvas.draw()

    def show(self):
        """Display the visualization"""
        plt.tight_layout()
        plt.show()


def main():
    """Main function to run the clustering visualizer"""
    print("Clustering Visualization Demo")
    print("=" * 40)
    print("Instructions:")
    print("1. K-means:")
    print("   - Click 'Init K-means' to initialize random centroids")
    print("   - Click 'Next Iteration' to see each step")
    print("   - Click 'Reset' to start over")
    print()
    print("2. DBSCAN:")
    print("   - Adjust epsilon (ε) with +/- buttons")
    print("   - Click 'Run DBSCAN' to see clustering result")
    print("   - Gray circles show epsilon neighborhoods")
    print("   - Black X's are noise points")
    print()

    visualizer = ClusteringVisualizer()
    visualizer.show()


if __name__ == "__main__":
    main()
