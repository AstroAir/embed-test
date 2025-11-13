"""Embedding quality validation and metrics.

This module provides comprehensive quality assessment for embeddings including:
- MTEB-style evaluation metrics
- Similarity validation and coherence checks
- Embedding distribution analysis
- Quality scoring and ranking
- Performance benchmarking
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from vectorflow.core.embeddings.base import EmbeddingResult
from vectorflow.core.utils.logging import LoggerMixin

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

    SKLEARN_AVAILABLE = True
except ImportError:
    cosine_similarity = None
    euclidean_distances = None
    KMeans = None
    SKLEARN_AVAILABLE = False

import importlib.util

SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None


class QualityMetric(Enum):
    """Quality metrics for embeddings."""

    COSINE_SIMILARITY = "cosine_similarity"
    EUCLIDEAN_DISTANCE = "euclidean_distance"
    INTRA_CLUSTER_COHERENCE = "intra_cluster_coherence"
    INTER_CLUSTER_SEPARATION = "inter_cluster_separation"
    DIMENSIONALITY_UTILIZATION = "dimensionality_utilization"
    EMBEDDING_VARIANCE = "embedding_variance"
    SEMANTIC_CONSISTENCY = "semantic_consistency"
    OUTLIER_DETECTION = "outlier_detection"


@dataclass
class QualityScore:
    """Quality score for embeddings."""

    metric: QualityMetric
    score: float
    max_score: float = 1.0
    description: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_score(self) -> float:
        if self.max_score == 0:
            return 0.0
        return min(1.0, max(0.0, self.score / self.max_score))

    @property
    def percentage(self) -> float:
        return self.normalized_score * 100


@dataclass
class QualityReport:
    """Comprehensive quality report for embeddings."""

    overall_score: float
    individual_scores: list[QualityScore]
    embedding_stats: dict[str, Any]
    recommendations: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_score_by_metric(self, metric: QualityMetric) -> Optional[QualityScore]:
        for score in self.individual_scores:
            if score.metric == metric:
                return score
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "overall_percentage": self.overall_score * 100,
            "individual_scores": [
                {
                    "metric": s.metric.value,
                    "score": s.score,
                    "normalized_score": s.normalized_score,
                    "percentage": s.percentage,
                    "description": s.description,
                    "details": s.details,
                }
                for s in self.individual_scores
            ],
            "embedding_stats": self.embedding_stats,
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }


class EmbeddingQualityValidator(LoggerMixin):
    """Validator for embedding quality assessment."""

    def __init__(self, enable_advanced_metrics: bool = True) -> None:
        self.enable_advanced_metrics = enable_advanced_metrics

        if not SKLEARN_AVAILABLE:
            self.logger.warning(
                "scikit-learn not available. Some quality metrics disabled. "
                "Install with: pip install scikit-learn"
            )
            self.enable_advanced_metrics = False

        if not SCIPY_AVAILABLE:
            self.logger.warning(
                "scipy not available. Statistical tests limited. "
                "Install with: pip install scipy"
            )

    def validate_embeddings(
        self,
        embedding_result: EmbeddingResult,
        texts: Optional[list[str]] = None,
        # kept for API compatibility
        reference_embeddings: Optional[list[list[float]]] = None,
    ) -> QualityReport:
        embeddings = np.array(embedding_result.embeddings)

        quality_scores: list[QualityScore] = []
        embedding_stats = self._calculate_embedding_stats(embeddings)

        dim_score = self._calculate_dimensionality_utilization(embeddings)
        quality_scores.append(dim_score)

        variance_score = self._calculate_embedding_variance(embeddings)
        quality_scores.append(variance_score)

        outlier_score = self._detect_outliers(embeddings)
        quality_scores.append(outlier_score)

        if self.enable_advanced_metrics and SKLEARN_AVAILABLE:
            cluster_scores = self._calculate_clustering_metrics(embeddings)
            quality_scores.extend(cluster_scores)
            if texts:
                semantic_score = self._calculate_semantic_consistency(embeddings, texts)
                quality_scores.append(semantic_score)

        overall_score = self._calculate_overall_score(quality_scores)
        recommendations = self._generate_recommendations(
            quality_scores, embedding_stats
        )

        report = QualityReport(
            overall_score=float(overall_score),
            individual_scores=quality_scores,
            embedding_stats=embedding_stats,
            recommendations=recommendations,
            metadata={
                "model_name": embedding_result.model_name,
                "embedding_count": len(embeddings),
                "embedding_dimension": embedding_result.embedding_dimension,
                "processing_time": embedding_result.processing_time,
                "advanced_metrics_enabled": self.enable_advanced_metrics,
            },
        )

        self.logger.info(
            f"Quality validation completed. Overall score: {overall_score:.3f} "
            f"({overall_score * 100:.1f}%)"
        )
        return report

    def _calculate_embedding_stats(self, embeddings: np.ndarray) -> dict[str, Any]:
        stats_dict: dict[str, Any] = {
            "count": len(embeddings),
            "dimension": embeddings.shape[1],
            "mean_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
            "std_norm": float(np.std(np.linalg.norm(embeddings, axis=1))),
            "min_norm": float(np.min(np.linalg.norm(embeddings, axis=1))),
            "max_norm": float(np.max(np.linalg.norm(embeddings, axis=1))),
            "mean_values": embeddings.mean(axis=0).tolist(),
            "std_values": embeddings.std(axis=0).tolist(),
            "zero_dimensions": int(np.sum(np.all(embeddings == 0, axis=0))),
            "near_zero_dimensions": int(
                np.sum(np.all(np.abs(embeddings) < 1e-6, axis=0))
            ),
        }
        if (
            len(embeddings) <= 1000
            and SKLEARN_AVAILABLE
            and cosine_similarity is not None
        ):
            similarities = cosine_similarity(embeddings)
            mask = ~np.eye(similarities.shape[0], dtype=bool)
            sim_values = similarities[mask]
            stats_dict.update(
                {
                    "mean_pairwise_similarity": float(np.mean(sim_values)),
                    "std_pairwise_similarity": float(np.std(sim_values)),
                    "min_pairwise_similarity": float(np.min(sim_values)),
                    "max_pairwise_similarity": float(np.max(sim_values)),
                }
            )
        return stats_dict

    def _calculate_dimensionality_utilization(
        self, embeddings: np.ndarray
    ) -> QualityScore:
        zero_dims = int(np.sum(np.all(embeddings == 0, axis=0)))
        near_zero_dims = int(np.sum(np.all(np.abs(embeddings) < 1e-6, axis=0)))
        total_dims = int(embeddings.shape[1])
        dim_variances = np.var(embeddings, axis=0)
        low_variance_dims = int(np.sum(dim_variances < 1e-4))
        utilized_dims = total_dims - zero_dims - near_zero_dims - low_variance_dims
        utilization_ratio = float(utilized_dims / total_dims) if total_dims else 0.0
        return QualityScore(
            metric=QualityMetric.DIMENSIONALITY_UTILIZATION,
            score=float(utilization_ratio),
            description=f"Utilization of embedding dimensions: {utilized_dims}/{total_dims}",
            details={
                "total_dimensions": total_dims,
                "utilized_dimensions": utilized_dims,
                "zero_dimensions": zero_dims,
                "near_zero_dimensions": near_zero_dims,
                "low_variance_dimensions": low_variance_dims,
                "utilization_ratio": utilization_ratio,
            },
        )

    def _calculate_embedding_variance(self, embeddings: np.ndarray) -> QualityScore:
        total_variance = float(np.var(embeddings))
        dim_variances = np.var(embeddings, axis=0)
        mean_dim_var = float(np.mean(dim_variances)) if len(dim_variances) else 0.0
        std_dim_var = float(np.std(dim_variances)) if len(dim_variances) else 0.0
        variance_uniformity = (
            float(1.0 - std_dim_var / mean_dim_var) if mean_dim_var > 0 else 0.0
        )
        variance_score = float(min(1.0, total_variance / 0.1))
        combined_score = float((variance_score + variance_uniformity) / 2.0)
        return QualityScore(
            metric=QualityMetric.EMBEDDING_VARIANCE,
            score=combined_score,
            description=f"Embedding variance quality: {combined_score:.3f}",
            details={
                "total_variance": total_variance,
                "mean_dimension_variance": mean_dim_var,
                "variance_uniformity": variance_uniformity,
                "variance_score": variance_score,
            },
        )

    def _detect_outliers(self, embeddings: np.ndarray) -> QualityScore:
        centroid = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        q75, q25 = np.percentile(distances, [75, 25])
        iqr = float(q75 - q25)
        outlier_threshold = float(q75 + 1.5 * iqr)
        outliers = int(np.sum(distances > outlier_threshold))
        outlier_ratio = float(outliers / len(embeddings)) if len(embeddings) else 0.0
        outlier_score = float(max(0.0, 1.0 - outlier_ratio * 2))
        return QualityScore(
            metric=QualityMetric.OUTLIER_DETECTION,
            score=outlier_score,
            description=f"Outlier detection: {outliers}/{len(embeddings)} outliers",
            details={
                "outlier_count": outliers,
                "outlier_ratio": outlier_ratio,
                "outlier_threshold": outlier_threshold,
                "mean_distance": float(np.mean(distances)),
                "std_distance": float(np.std(distances)),
            },
        )

    def _calculate_clustering_metrics(
        self, embeddings: np.ndarray
    ) -> list[QualityScore]:
        if not SKLEARN_AVAILABLE or KMeans is None or len(embeddings) < 10:
            return []
        scores: list[QualityScore] = []
        n_clusters = int(min(10, max(2, len(embeddings) // 20)))
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            intra = self._calculate_intra_cluster_coherence(
                embeddings,
                cluster_labels,
                kmeans.cluster_centers_,
            )
            scores.append(intra)
            inter = self._calculate_inter_cluster_separation(kmeans.cluster_centers_)
            scores.append(inter)
        except Exception as e:
            self.logger.warning(f"Clustering metrics failed: {e}")
        return scores

    def _calculate_intra_cluster_coherence(
        self, embeddings: np.ndarray, labels: np.ndarray, centers: np.ndarray
    ) -> QualityScore:
        coherence_scores: list[float] = []
        for cluster_id in np.unique(labels):
            cluster_embeddings = embeddings[labels == cluster_id]
            if len(cluster_embeddings) > 1:
                center = centers[cluster_id]
                distances = np.linalg.norm(cluster_embeddings - center, axis=1)
                coherence = float(1.0 / (1.0 + np.mean(distances)))
                coherence_scores.append(coherence)
        overall_coherence = (
            float(np.mean(coherence_scores)) if coherence_scores else 0.0
        )
        return QualityScore(
            metric=QualityMetric.INTRA_CLUSTER_COHERENCE,
            score=overall_coherence,
            description=f"Intra-cluster coherence: {overall_coherence:.3f}",
            details={
                "cluster_count": len(np.unique(labels)),
                "coherence_scores": coherence_scores,
                "mean_coherence": overall_coherence,
            },
        )

    def _calculate_inter_cluster_separation(self, centers: np.ndarray) -> QualityScore:
        if len(centers) < 2 or not SKLEARN_AVAILABLE or euclidean_distances is None:
            return QualityScore(
                metric=QualityMetric.INTER_CLUSTER_SEPARATION,
                score=0.0,
                description="Insufficient clusters or sklearn unavailable",
            )
        distances = euclidean_distances(centers)
        mask = ~np.eye(distances.shape[0], dtype=bool)
        inter_distances = distances[mask]
        mean_separation = float(np.mean(inter_distances))
        separation_score = float(min(1.0, mean_separation / 2.0))
        return QualityScore(
            metric=QualityMetric.INTER_CLUSTER_SEPARATION,
            score=separation_score,
            description=f"Inter-cluster separation: {separation_score:.3f}",
            details={
                "mean_separation": mean_separation,
                "min_separation": float(np.min(inter_distances)),
                "max_separation": float(np.max(inter_distances)),
                "std_separation": float(np.std(inter_distances)),
            },
        )

    def _calculate_semantic_consistency(
        self, embeddings: np.ndarray, texts: list[str]
    ) -> QualityScore:
        if not SKLEARN_AVAILABLE or cosine_similarity is None or len(texts) < 2:
            return QualityScore(
                metric=QualityMetric.SEMANTIC_CONSISTENCY,
                score=0.0,
                description="Semantic consistency unavailable (sklearn missing or insufficient texts)",
            )
        sample_size = min(100, max(0, len(texts) // 2))
        if sample_size == 0:
            return QualityScore(
                metric=QualityMetric.SEMANTIC_CONSISTENCY,
                score=0.0,
                description="Not enough texts for semantic consistency sampling",
            )
        indices = np.random.choice(len(texts), size=sample_size * 2, replace=False)
        consistency_scores: list[float] = []
        for i in range(0, len(indices), 2):
            idx1, idx2 = indices[i], indices[i + 1]
            text_sim = self._calculate_text_similarity(texts[idx1], texts[idx2])
            v1 = embeddings[idx1][np.newaxis, :]
            v2 = embeddings[idx2][np.newaxis, :]
            emb_sim_matrix = cosine_similarity(v1, v2)
            emb_sim = float(emb_sim_matrix[0, 0])
            consistency_scores.append(abs(text_sim - emb_sim))
        if not consistency_scores:
            mean_consistency = 0.0
        else:
            mean_consistency = float(1.0 - np.mean(consistency_scores))
        semantic_score = float(max(0.0, mean_consistency))
        return QualityScore(
            metric=QualityMetric.SEMANTIC_CONSISTENCY,
            score=semantic_score,
            description=f"Semantic consistency: {mean_consistency:.3f}",
            details={
                "sample_size": sample_size,
                "consistency_scores": consistency_scores,
                "mean_consistency": mean_consistency,
            },
        )

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return float(intersection / union) if union > 0 else 0.0

    def _calculate_overall_score(self, quality_scores: list[QualityScore]) -> float:
        if not quality_scores:
            return 0.0
        weights = {
            QualityMetric.DIMENSIONALITY_UTILIZATION: 0.2,
            QualityMetric.EMBEDDING_VARIANCE: 0.15,
            QualityMetric.OUTLIER_DETECTION: 0.15,
            QualityMetric.INTRA_CLUSTER_COHERENCE: 0.2,
            QualityMetric.INTER_CLUSTER_SEPARATION: 0.15,
            QualityMetric.SEMANTIC_CONSISTENCY: 0.15,
        }
        weighted_sum = 0.0
        total_weight = 0.0
        for s in quality_scores:
            w = weights.get(s.metric, 0.1)
            weighted_sum += s.normalized_score * w
            total_weight += w
        return float(weighted_sum / total_weight) if total_weight > 0 else 0.0

    def _generate_recommendations(
        self, quality_scores: list[QualityScore], embedding_stats: dict[str, Any]
    ) -> list[str]:
        recommendations: list[str] = []
        for s in quality_scores:
            if s.normalized_score < 0.5:
                if s.metric == QualityMetric.DIMENSIONALITY_UTILIZATION:
                    recommendations.append(
                        "Consider using a model with better dimension utilization or reducing embedding dimension"
                    )
                elif s.metric == QualityMetric.EMBEDDING_VARIANCE:
                    recommendations.append(
                        "Embeddings show low variance; adjust preprocessing or use a more expressive model"
                    )
                elif s.metric == QualityMetric.OUTLIER_DETECTION:
                    recommendations.append(
                        "High outlier ratio; review input text quality and preprocessing"
                    )
                elif s.metric == QualityMetric.SEMANTIC_CONSISTENCY:
                    recommendations.append(
                        "Low semantic consistency; consider fine-tuning or improved preprocessing"
                    )
        if embedding_stats.get("zero_dimensions", 0) > 0:
            recommendations.append(
                f"Found {embedding_stats['zero_dimensions']} unused dimensions; consider dimension reduction"
            )
        if not recommendations:
            recommendations.append(
                "Embedding quality is good. No major issues detected."
            )
        return recommendations
