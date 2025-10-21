"""Tests for embedding quality validation module."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from pdf_vector_system.embeddings.base import EmbeddingResult
from pdf_vector_system.embeddings.quality import (
    EmbeddingQualityValidator,
    QualityMetric,
    QualityReport,
    QualityScore,
)


class TestQualityScore:
    """Test quality score class."""

    def test_creation(self):
        """Test quality score creation."""
        score = QualityScore(
            metric=QualityMetric.COSINE_SIMILARITY,
            score=0.8,
            max_score=1.0,
            description="Test score",
        )

        assert score.metric == QualityMetric.COSINE_SIMILARITY
        assert score.score == 0.8
        assert score.max_score == 1.0
        assert score.description == "Test score"

    def test_normalized_score(self):
        """Test normalized score calculation."""
        score = QualityScore(
            metric=QualityMetric.COSINE_SIMILARITY, score=0.8, max_score=1.0
        )
        assert score.normalized_score == 0.8

        # Test with different max score
        score = QualityScore(
            metric=QualityMetric.COSINE_SIMILARITY, score=4.0, max_score=5.0
        )
        assert score.normalized_score == 0.8

        # Test clamping
        score = QualityScore(
            metric=QualityMetric.COSINE_SIMILARITY, score=1.5, max_score=1.0
        )
        assert score.normalized_score == 1.0

    def test_percentage(self):
        """Test percentage calculation."""
        score = QualityScore(
            metric=QualityMetric.COSINE_SIMILARITY, score=0.75, max_score=1.0
        )
        assert score.percentage == 75.0


class TestQualityReport:
    """Test quality report class."""

    def test_creation(self):
        """Test quality report creation."""
        scores = [
            QualityScore(QualityMetric.COSINE_SIMILARITY, 0.8),
            QualityScore(QualityMetric.EMBEDDING_VARIANCE, 0.7),
        ]

        report = QualityReport(
            overall_score=0.75,
            individual_scores=scores,
            embedding_stats={"count": 100},
            recommendations=["Test recommendation"],
        )

        assert report.overall_score == 0.75
        assert len(report.individual_scores) == 2
        assert report.embedding_stats["count"] == 100
        assert len(report.recommendations) == 1

    def test_get_score_by_metric(self):
        """Test getting score by metric."""
        scores = [
            QualityScore(QualityMetric.COSINE_SIMILARITY, 0.8),
            QualityScore(QualityMetric.EMBEDDING_VARIANCE, 0.7),
        ]

        report = QualityReport(
            overall_score=0.75,
            individual_scores=scores,
            embedding_stats={},
            recommendations=[],
        )

        # Test existing metric
        score = report.get_score_by_metric(QualityMetric.COSINE_SIMILARITY)
        assert score is not None
        assert score.score == 0.8

        # Test non-existing metric
        score = report.get_score_by_metric(QualityMetric.OUTLIER_DETECTION)
        assert score is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        scores = [
            QualityScore(QualityMetric.COSINE_SIMILARITY, 0.8, description="Test")
        ]

        report = QualityReport(
            overall_score=0.75,
            individual_scores=scores,
            embedding_stats={"count": 100},
            recommendations=["Test recommendation"],
        )

        report_dict = report.to_dict()

        assert "overall_score" in report_dict
        assert "overall_percentage" in report_dict
        assert "individual_scores" in report_dict
        assert "embedding_stats" in report_dict
        assert "recommendations" in report_dict

        assert report_dict["overall_score"] == 0.75
        assert report_dict["overall_percentage"] == 75.0
        assert len(report_dict["individual_scores"]) == 1


class TestEmbeddingQualityValidator:
    """Test embedding quality validator."""

    def test_initialization(self):
        """Test validator initialization."""
        validator = EmbeddingQualityValidator()
        assert validator is not None

        # Test with advanced metrics disabled
        validator = EmbeddingQualityValidator(enable_advanced_metrics=False)
        assert validator.enable_advanced_metrics is False

    def test_calculate_embedding_stats(self):
        """Test embedding statistics calculation."""
        validator = EmbeddingQualityValidator()

        # Create test embeddings
        embeddings = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.5, 0.5, 1.0]])

        stats = validator._calculate_embedding_stats(embeddings)

        assert "count" in stats
        assert "dimension" in stats
        assert "mean_norm" in stats
        assert "std_norm" in stats
        assert "min_norm" in stats
        assert "max_norm" in stats
        assert "zero_dimensions" in stats

        assert stats["count"] == 3
        assert stats["dimension"] == 3
        assert stats["zero_dimensions"] == 0

    def test_calculate_dimensionality_utilization(self):
        """Test dimensionality utilization calculation."""
        validator = EmbeddingQualityValidator()

        # Test with well-utilized dimensions
        embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        score = validator._calculate_dimensionality_utilization(embeddings)

        assert score.metric == QualityMetric.DIMENSIONALITY_UTILIZATION
        assert score.score > 0.5  # Should be high utilization
        assert "total_dimensions" in score.details
        assert "utilized_dimensions" in score.details

        # Test with zero dimensions
        embeddings_with_zeros = np.array(
            [[1.0, 0.0, 3.0], [4.0, 0.0, 6.0], [7.0, 0.0, 9.0]]
        )

        score = validator._calculate_dimensionality_utilization(embeddings_with_zeros)
        assert score.details["zero_dimensions"] == 1

    def test_calculate_embedding_variance(self):
        """Test embedding variance calculation."""
        validator = EmbeddingQualityValidator()

        # Test with varied embeddings
        embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        score = validator._calculate_embedding_variance(embeddings)

        assert score.metric == QualityMetric.EMBEDDING_VARIANCE
        assert score.score > 0
        assert "total_variance" in score.details
        assert "variance_uniformity" in score.details

    def test_detect_outliers(self):
        """Test outlier detection."""
        validator = EmbeddingQualityValidator()

        # Create embeddings with one clear outlier
        embeddings = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.1, 1.1, 1.1],
                [0.9, 0.9, 0.9],
                [10.0, 10.0, 10.0],  # Outlier
            ]
        )

        score = validator._detect_outliers(embeddings)

        assert score.metric == QualityMetric.OUTLIER_DETECTION
        assert "outlier_count" in score.details
        assert "outlier_ratio" in score.details
        assert score.details["outlier_count"] >= 1  # Should detect the outlier

    @patch("pdf_vector_system.embeddings.quality.SKLEARN_AVAILABLE", True)
    @patch("pdf_vector_system.embeddings.quality.KMeans")
    def test_calculate_clustering_metrics(self, mock_kmeans):
        """Test clustering metrics calculation."""
        validator = EmbeddingQualityValidator(enable_advanced_metrics=True)

        # Mock KMeans
        mock_kmeans_instance = Mock()
        mock_kmeans_instance.fit_predict.return_value = np.array([0, 0, 1, 1])
        mock_kmeans_instance.cluster_centers_ = np.array(
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
        )
        mock_kmeans.return_value = mock_kmeans_instance

        embeddings = np.array(
            [[1.0, 1.0, 1.0], [1.1, 1.1, 1.1], [2.0, 2.0, 2.0], [2.1, 2.1, 2.1]]
        )

        scores = validator._calculate_clustering_metrics(embeddings)

        assert len(scores) >= 1  # Should return at least one clustering metric
        mock_kmeans.assert_called_once()

    def test_calculate_semantic_consistency(self):
        """Test semantic consistency calculation."""
        validator = EmbeddingQualityValidator()

        embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        texts = ["hello world", "goodbye world", "hello universe"]

        score = validator._calculate_semantic_consistency(embeddings, texts)

        assert score.metric == QualityMetric.SEMANTIC_CONSISTENCY
        assert score.score >= 0
        assert "sample_size" in score.details
        assert "consistency_scores" in score.details

    def test_calculate_overall_score(self):
        """Test overall score calculation."""
        validator = EmbeddingQualityValidator()

        scores = [
            QualityScore(QualityMetric.DIMENSIONALITY_UTILIZATION, 0.8),
            QualityScore(QualityMetric.EMBEDDING_VARIANCE, 0.7),
            QualityScore(QualityMetric.OUTLIER_DETECTION, 0.9),
        ]

        overall_score = validator._calculate_overall_score(scores)

        assert 0 <= overall_score <= 1
        assert overall_score > 0  # Should be positive with good scores

    def test_generate_recommendations(self):
        """Test recommendation generation."""
        validator = EmbeddingQualityValidator()

        # Test with low scores
        low_scores = [
            QualityScore(QualityMetric.DIMENSIONALITY_UTILIZATION, 0.3),
            QualityScore(QualityMetric.EMBEDDING_VARIANCE, 0.2),
            QualityScore(QualityMetric.OUTLIER_DETECTION, 0.1),
        ]

        recommendations = validator._generate_recommendations(
            low_scores, {"zero_dimensions": 5}
        )

        assert len(recommendations) > 0
        assert any("dimension" in rec.lower() for rec in recommendations)

        # Test with good scores
        good_scores = [
            QualityScore(QualityMetric.DIMENSIONALITY_UTILIZATION, 0.9),
            QualityScore(QualityMetric.EMBEDDING_VARIANCE, 0.8),
            QualityScore(QualityMetric.OUTLIER_DETECTION, 0.9),
        ]

        recommendations = validator._generate_recommendations(
            good_scores, {"zero_dimensions": 0}
        )

        assert len(recommendations) > 0
        assert any("good" in rec.lower() for rec in recommendations)

    def test_validate_embeddings_basic(self):
        """Test basic embedding validation."""
        validator = EmbeddingQualityValidator(enable_advanced_metrics=False)

        # Create test embedding result
        embeddings = [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.5, 0.5, 1.0]]

        embedding_result = EmbeddingResult(
            embeddings=embeddings,
            model_name="test-model",
            embedding_dimension=3,
            processing_time=1.0,
        )

        report = validator.validate_embeddings(embedding_result)

        assert isinstance(report, QualityReport)
        assert 0 <= report.overall_score <= 1
        assert len(report.individual_scores) > 0
        assert len(report.recommendations) > 0
        assert "model_name" in report.metadata
        assert report.metadata["model_name"] == "test-model"

    @patch("pdf_vector_system.embeddings.quality.SKLEARN_AVAILABLE", True)
    def test_validate_embeddings_advanced(self):
        """Test advanced embedding validation."""
        validator = EmbeddingQualityValidator(enable_advanced_metrics=True)

        # Create test embedding result
        embeddings = [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
            [0.5, 0.5, 1.0],
            [0.2, 0.8, 0.3],
        ]

        embedding_result = EmbeddingResult(
            embeddings=embeddings,
            model_name="test-model",
            embedding_dimension=3,
            processing_time=1.0,
        )

        texts = ["hello", "world", "test", "example"]

        with patch.object(validator, "_calculate_clustering_metrics", return_value=[]):
            report = validator.validate_embeddings(embedding_result, texts=texts)

        assert isinstance(report, QualityReport)
        assert 0 <= report.overall_score <= 1
        assert len(report.individual_scores) > 0
        assert report.metadata["advanced_metrics_enabled"] is True

    def test_validate_embeddings_empty(self):
        """Test validation with empty embeddings."""
        validator = EmbeddingQualityValidator()

        embedding_result = EmbeddingResult(
            embeddings=[],
            model_name="test-model",
            embedding_dimension=0,
            processing_time=1.0,
        )

        # Should handle empty embeddings gracefully
        with pytest.raises((ValueError, IndexError)):
            validator.validate_embeddings(embedding_result)


# Integration tests
class TestQualityValidationIntegration:
    """Integration tests for quality validation."""

    def test_end_to_end_validation(self):
        """Test end-to-end quality validation."""
        # Create realistic embeddings
        np.random.seed(42)
        embeddings = np.random.normal(0, 1, (50, 128)).tolist()

        embedding_result = EmbeddingResult(
            embeddings=embeddings,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dimension=128,
            processing_time=2.5,
            metadata={"batch_size": 50},
        )

        validator = EmbeddingQualityValidator(enable_advanced_metrics=False)
        report = validator.validate_embeddings(embedding_result)

        # Verify report structure
        assert isinstance(report, QualityReport)
        assert 0 <= report.overall_score <= 1
        assert len(report.individual_scores) >= 3  # At least basic metrics
        assert len(report.recommendations) > 0

        # Verify report can be converted to dict
        report_dict = report.to_dict()
        assert "overall_score" in report_dict
        assert "individual_scores" in report_dict
        assert "recommendations" in report_dict

        # Verify individual scores
        for score in report.individual_scores:
            assert isinstance(score, QualityScore)
            assert 0 <= score.normalized_score <= 1
            assert score.description != ""

    def test_quality_metrics_consistency(self):
        """Test consistency of quality metrics."""
        validator = EmbeddingQualityValidator()

        # Create two sets of embeddings - one good, one poor
        good_embeddings = np.random.normal(0, 1, (20, 64))
        poor_embeddings = np.zeros((20, 64))  # All zeros - poor quality

        good_result = EmbeddingResult(
            embeddings=good_embeddings.tolist(),
            model_name="test",
            embedding_dimension=64,
            processing_time=1.0,
        )

        poor_result = EmbeddingResult(
            embeddings=poor_embeddings.tolist(),
            model_name="test",
            embedding_dimension=64,
            processing_time=1.0,
        )

        good_report = validator.validate_embeddings(good_result)
        poor_report = validator.validate_embeddings(poor_result)

        # Good embeddings should have higher overall score
        assert good_report.overall_score > poor_report.overall_score

        # Poor embeddings should have more recommendations
        assert len(poor_report.recommendations) >= len(good_report.recommendations)
