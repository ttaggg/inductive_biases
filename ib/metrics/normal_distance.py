"""Normal distance metric."""

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

from ib.utils.data import load_pointcloud, write_ply
from ib.utils.labels import INX_TO_LABEL
from ib.utils.pointcloud import filter_incorrect_normals


class NormalCosineSimilarity:
    """Bidirectional Normal Cosine Similarity metric."""

    def __init__(
        self,
        vertices: np.ndarray,
        normals: np.ndarray,
        labels: np.ndarray | None,
    ):
        self.vertices = vertices
        self.normals = normals
        self.labels = labels
        self.tree = KDTree(vertices)

    @classmethod
    def from_pointcloud(
        cls,
        target_vertices: np.ndarray,
        target_normals: np.ndarray,
        target_labels: np.ndarray | None,
    ):
        vertices, normals, labels = filter_incorrect_normals(
            target_vertices, target_normals, target_labels
        )
        return cls(vertices, normals, labels)

    @classmethod
    def from_pointcloud_path(cls, pointcloud_path: Path):
        data = load_pointcloud(pointcloud_path)
        vertices, normals, labels = filter_incorrect_normals(
            data["points"], data["normals"], data["labels"]
        )
        return cls(vertices, normals, labels)

    def _compute_directional_similarity_radius(
        self,
        source_tree: KDTree,
        source_normals: np.ndarray,
        target_tree: KDTree,
        target_normals: np.ndarray,
        radius: float,
    ) -> np.ndarray:
        # NOTE(oleg): numba is 1.5 times faster, and same memory
        # NOTE(oleg): query_ball_point is 2 times slower,
        # but does not require a lot of RAM
        neighbor_indices: list[list] = source_tree.query_ball_tree(
            target_tree, r=radius
        )
        sims = np.nan * np.ones(len(source_normals))
        for i, indices in enumerate(neighbor_indices):
            if len(indices) != 0:
                similarities = np.dot(target_normals[indices], source_normals[i])
                sims[i] = np.max(similarities)
        return sims

    def _compute_directional_similarity_closest(
        self,
        source_vertices: np.ndarray,
        source_normals: np.ndarray,
        target_tree: KDTree,
        target_normals: np.ndarray,
    ) -> np.ndarray:
        _, indices = target_tree.query(source_vertices, k=1, workers=-1)
        sims = np.sum(target_normals[indices] * source_normals, axis=1)
        return sims

    def visualize_and_save_colormaps(
        self,
        vertices: np.ndarray,
        normals: np.ndarray,
        sims: np.ndarray,
        save_path: Path,
        mask: np.ndarray | None = None,
    ) -> None:
        t = (sims + 1.0) / 2.0
        rgba = plt.cm.jet_r(t)
        colors = (rgba[:, :3] * 255).astype(np.uint8)
        if mask is not None:
            vertices = vertices[mask]
            normals = normals[mask]
            colors = colors[mask]
        write_ply(save_path, vertices, normals, colors)

    def visualize_and_save_nanmaps(
        self,
        vertices: np.ndarray,
        sims: np.ndarray,
        save_path: Path,
        mask: np.ndarray | None = None,
    ) -> None:
        overall_mask = np.isnan(sims)
        if mask is not None:
            overall_mask = np.logical_and(overall_mask, mask)
        vertices = vertices[overall_mask]
        write_ply(save_path, vertices)

    def __call__(
        self,
        pred_vertices: np.ndarray,
        pred_normals: np.ndarray,
        radius: float = 0.003,
        save_path: Optional[Path] = None,
    ) -> dict[str, float]:
        pred_tree = KDTree(pred_vertices)
        # Compare with the best normal in some radius.
        radius_sims_t2p = self._compute_directional_similarity_radius(
            self.tree, self.normals, pred_tree, pred_normals, radius
        )
        radius_sims_p2t = self._compute_directional_similarity_radius(
            pred_tree, pred_normals, self.tree, self.normals, radius
        )
        # Compare with the closest normal.
        closest_sims_t2p = self._compute_directional_similarity_closest(
            self.vertices, self.normals, pred_tree, pred_normals
        )
        closest_sims_p2t = self._compute_directional_similarity_closest(
            pred_vertices, pred_normals, self.tree, self.normals
        )
        # Use closest if radius similarity is nan.
        combined_sims_t2p = np.where(
            np.isnan(radius_sims_t2p), closest_sims_t2p, radius_sims_t2p
        )
        combined_sims_p2t = np.where(
            np.isnan(radius_sims_p2t), closest_sims_p2t, radius_sims_p2t
        )
        normal_similarity = (combined_sims_p2t.mean() + combined_sims_t2p.mean()) / 2.0

        if save_path is not None:
            self.visualize_and_save_colormaps(
                pred_vertices,
                pred_normals,
                combined_sims_p2t,
                save_path.with_name(f"{save_path.stem}_p2t.ply"),
            )
            self.visualize_and_save_colormaps(
                self.vertices,
                self.normals,
                combined_sims_t2p,
                save_path.with_name(f"{save_path.stem}_t2p.ply"),
            )
            self.visualize_and_save_nanmaps(
                pred_vertices,
                radius_sims_p2t,
                save_path.with_name(f"{save_path.stem}_nans_p2t.ply"),
            )
            self.visualize_and_save_nanmaps(
                self.vertices,
                radius_sims_t2p,
                save_path.with_name(f"{save_path.stem}_nans_t2p.ply"),
            )

        results = {
            "metrics/normal_similarity": float(normal_similarity),
            "metrics_main/normal_similarity_t2p": float(combined_sims_t2p.mean()),
            "metrics/normal_similarity_p2t": float(combined_sims_p2t.mean()),
            "metrics/ratio_nan_t2p": float(
                np.isnan(radius_sims_t2p).sum() / len(radius_sims_t2p)
            ),
            "metrics/ratio_nan_p2t": float(
                np.isnan(radius_sims_p2t).sum() / len(radius_sims_p2t)
            ),
            "metrics_main/closest_sims_t2p": float(closest_sims_t2p.mean()),
        }

        _, idx_pred = self.tree.query(pred_vertices, workers=-1)

        label_indices = np.unique(self.labels)
        for label_inx in label_indices:
            label_name = INX_TO_LABEL.get(label_inx, "unknown")
            # T2P
            mask_label = self.labels == label_inx
            label_dist_t2p_mean = combined_sims_t2p[mask_label].mean()
            results[f"metrics_labels/normalsclosest_{label_name}_t2p"] = float(
                closest_sims_t2p[mask_label].mean()
            )
            results[f"metrics_labels/normals_{label_name}_t2p"] = float(
                label_dist_t2p_mean
            )
            # P2T
            mask_pred = self.labels[idx_pred] == label_inx
            label_dist_p2t_mean = combined_sims_p2t[mask_pred].mean()
            results[f"metrics_labels/normals_closest_{label_name}_p2t"] = float(
                closest_sims_p2t[mask_pred].mean()
            )
            results[f"metrics_labels/normals_{label_name}_p2t"] = float(
                label_dist_p2t_mean
            )
            # Mean.
            results[f"metrics_labels/normals_{label_name}"] = float(
                (label_dist_t2p_mean + label_dist_p2t_mean) / 2.0
            )

        if self.labels is not None:
            mask_self = self.labels > 0
            mask_pred = self.labels[idx_pred] > 0

            # Low-resolution regions.
            low_freq_t2p = combined_sims_t2p[mask_self].mean()
            low_freq_p2t = combined_sims_p2t[mask_pred].mean()
            results["metrics_main/normals_low_freq_t2p"] = float(low_freq_t2p)
            results["metrics/normals_low_freq_p2t"] = float(low_freq_p2t)
            closest_t2p_low_freq = closest_sims_t2p[mask_self].mean()
            results["metrics_main/normals_closest_low_freq_t2p"] = float(
                closest_t2p_low_freq
            )
            results["metrics_main/normals_low_freq"] = float(
                (low_freq_t2p + low_freq_p2t) / 2.0
            )

            # Other regions.
            mask_self_hf = self.labels < 0
            mask_pred_hf = self.labels[idx_pred] < 0
            high_freq_t2p = combined_sims_t2p[mask_self_hf].mean()

            high_freq_p2t = combined_sims_p2t[mask_pred_hf].mean()
            results["metrics_main/normals_high_freq_t2p"] = float(high_freq_t2p)
            results["metrics/normal_high_freq_p2t"] = float(high_freq_p2t)
            closest_t2p_high_freq = closest_sims_t2p[mask_self_hf].mean()
            results["metrics_main/normals_closest_high_freq_t2p"] = float(
                closest_t2p_high_freq
            )
            results["metrics_main/normals_high_freq"] = float(
                (high_freq_t2p + high_freq_p2t) / 2.0
            )

            # Label-dependent visualizations
            if save_path is not None:
                self.visualize_and_save_colormaps(
                    self.vertices,
                    self.normals,
                    closest_sims_t2p,
                    save_path.with_name(f"{save_path.stem}_closest_target_lf_p2t.ply"),
                    mask_self,
                )
                self.visualize_and_save_colormaps(
                    self.vertices,
                    self.normals,
                    closest_sims_t2p,
                    save_path.with_name(f"{save_path.stem}_closest_target_hf_p2t.ply"),
                    mask_self_hf,
                )
                self.visualize_and_save_colormaps(
                    pred_vertices,
                    pred_normals,
                    combined_sims_p2t,
                    save_path.with_name(f"{save_path.stem}_pred_hf_p2t.ply"),
                    mask_pred_hf,
                )
                self.visualize_and_save_colormaps(
                    self.vertices,
                    self.normals,
                    combined_sims_t2p,
                    save_path.with_name(f"{save_path.stem}_target_hf_t2p.ply"),
                    mask_self_hf,
                )
                self.visualize_and_save_colormaps(
                    pred_vertices,
                    pred_normals,
                    combined_sims_p2t,
                    save_path.with_name(f"{save_path.stem}_pred_lf_p2t.ply"),
                    mask_pred,
                )
                self.visualize_and_save_colormaps(
                    self.vertices,
                    self.normals,
                    combined_sims_t2p,
                    save_path.with_name(f"{save_path.stem}_target_lf_t2p.ply"),
                    mask_self,
                )
                self.visualize_and_save_nanmaps(
                    pred_vertices,
                    radius_sims_p2t,
                    save_path.with_name(f"{save_path.stem}_nans_pred_hf_p2t.ply"),
                    mask_pred_hf,
                )
                self.visualize_and_save_nanmaps(
                    self.vertices,
                    radius_sims_t2p,
                    save_path.with_name(f"{save_path.stem}_nans_target_hf_t2p.ply"),
                    mask_self_hf,
                )
                self.visualize_and_save_nanmaps(
                    pred_vertices,
                    radius_sims_p2t,
                    save_path.with_name(f"{save_path.stem}_nans_pred_lf_p2t.ply"),
                    mask_pred,
                )
                self.visualize_and_save_nanmaps(
                    self.vertices,
                    radius_sims_t2p,
                    save_path.with_name(f"{save_path.stem}_nans_target_lf_t2p.ply"),
                    mask_self,
                )

        return results
