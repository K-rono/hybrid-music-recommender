from __future__ import annotations

import time
import psutil
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from collections import defaultdict

from .explainer import RecommendationExplainer, RecommendationExplanation
from .recommender import HybridConfig


@dataclass
class StabilityMetrics:
    """Metrics for explanation stability"""
    feature_consistency: float  # % of runs with same top 3 features
    user_consistency: float     # % of runs with same top 3 similar users
    score_consistency: float    # % of runs with same score breakdown


@dataclass
class CoverageMetrics:
    """Metrics for explanation coverage"""
    total_recommendations: int
    successful_explanations: int
    coverage_percentage: float
    failed_explanations: int
    failure_reasons: Dict[str, int]


@dataclass
class EfficiencyMetrics:
    """Metrics for explanation efficiency"""
    avg_generation_time: float  # seconds
    min_generation_time: float
    max_generation_time: float


@dataclass
class UserEvaluationResult:
    """Evaluation results for a single user"""
    user_id: str
    total_recommendations: int
    stability: StabilityMetrics
    coverage: CoverageMetrics
    efficiency: EfficiencyMetrics
    individual_explanations: List[Dict[str, Any]]


@dataclass
class BatchEvaluationResult:
    """Aggregated evaluation results for all users"""
    total_users: int
    total_recommendations: int
    avg_stability: StabilityMetrics
    avg_coverage: CoverageMetrics
    avg_efficiency: EfficiencyMetrics
    user_results: List[UserEvaluationResult]
    evaluation_timestamp: str


class ExplainabilityEvaluator:
    """Evaluator for explainability metrics: Stability, Coverage, Efficiency"""
    
    def __init__(self, explainer: RecommendationExplainer):
        self.explainer = explainer
        self.process = psutil.Process()
    
    def evaluate_stability(
        self, 
        user_id: str, 
        track_id: str, 
        score: float, 
        n_runs: int = 3
    ) -> StabilityMetrics:
        """Evaluate explanation stability by running multiple times"""
        
        explanations = []
        
        # Generate explanations multiple times
        for _ in range(n_runs):
            try:
                explanation = self.explainer.explain_recommendation(user_id, track_id, score)
                explanations.append(explanation)
            except Exception as e:
                print(f"Error generating explanation for run {_}: {e}")
                continue
        
        if len(explanations) < 2:
            return StabilityMetrics(
                feature_consistency=0.0,
                user_consistency=0.0,
                score_consistency=0.0
            )
        
        # Analyze feature consistency
        feature_consistency = self._analyze_feature_consistency(explanations)
        
        # Analyze user consistency
        user_consistency = self._analyze_user_consistency(explanations)
        
        # Analyze score consistency
        score_consistency = self._analyze_score_consistency(explanations)
        
        return StabilityMetrics(
            feature_consistency=feature_consistency,
            user_consistency=user_consistency,
            score_consistency=score_consistency
        )
    
    def _analyze_feature_consistency(self, explanations: List[RecommendationExplanation]) -> float:
        """Analyze consistency of top 3 features across runs"""
        if len(explanations) < 2:
            return 0.0
        
        # Get top 3 features for each run
        feature_sets = []
        for exp in explanations:
            top_features = [c.feature_name for c in exp.content_contributions[:3]]
            feature_sets.append(set(top_features))
        
        # Calculate consistency
        consistent_runs = 0
        for i in range(len(feature_sets)):
            for j in range(i + 1, len(feature_sets)):
                if feature_sets[i] == feature_sets[j]:
                    consistent_runs += 1
        
        total_comparisons = len(feature_sets) * (len(feature_sets) - 1) // 2
        return consistent_runs / total_comparisons if total_comparisons > 0 else 0.0
    
    def _analyze_user_consistency(self, explanations: List[RecommendationExplanation]) -> float:
        """Analyze consistency of top 3 similar users across runs"""
        if len(explanations) < 2:
            return 0.0
        
        # Get top 3 users for each run
        user_sets = []
        for exp in explanations:
            top_users = [uid for uid, _ in exp.collaborative_insight.similar_users[:3]]
            user_sets.append(set(top_users))
        
        # Calculate consistency
        consistent_runs = 0
        for i in range(len(user_sets)):
            for j in range(i + 1, len(user_sets)):
                if user_sets[i] == user_sets[j]:
                    consistent_runs += 1
        
        total_comparisons = len(user_sets) * (len(user_sets) - 1) // 2
        return consistent_runs / total_comparisons if total_comparisons > 0 else 0.0
    
    def _analyze_score_consistency(self, explanations: List[RecommendationExplanation]) -> float:
        """Analyze consistency of score breakdown across runs"""
        if len(explanations) < 2:
            return 0.0
        
        # Get score breakdowns
        scores = []
        for exp in explanations:
            scores.append(exp.score_breakdown)
        
        # Calculate variance in scores
        cf_scores = [s['collaborative_score'] for s in scores]
        content_scores = [s['content_score'] for s in scores]
        total_scores = [s['total_score'] for s in scores]
        
        # Consider consistent if variance is low (within 1% of mean)
        cf_consistent = np.std(cf_scores) < 0.01 * np.mean(cf_scores) if np.mean(cf_scores) > 0 else True
        content_consistent = np.std(content_scores) < 0.01 * np.mean(content_scores) if np.mean(content_scores) > 0 else True
        total_consistent = np.std(total_scores) < 0.01 * np.mean(total_scores) if np.mean(total_scores) > 0 else True
        
        return sum([cf_consistent, content_consistent, total_consistent]) / 3.0
    
    
    def evaluate_coverage(
        self, 
        recommendations: List[Tuple[str, float]], 
        user_id: str
    ) -> CoverageMetrics:
        """Evaluate explanation coverage for a user's recommendations"""
        
        total_recommendations = len(recommendations)
        successful_explanations = 0
        failed_explanations = 0
        failure_reasons = defaultdict(int)
        
        for track_id, score in recommendations:
            try:
                explanation = self.explainer.explain_recommendation(user_id, track_id, score)
                if explanation and explanation.content_contributions:
                    successful_explanations += 1
                else:
                    failed_explanations += 1
                    failure_reasons["empty_explanation"] += 1
            except Exception as e:
                failed_explanations += 1
                failure_reasons[str(type(e).__name__)] += 1
        
        coverage_percentage = (successful_explanations / total_recommendations * 100) if total_recommendations > 0 else 0.0
        
        return CoverageMetrics(
            total_recommendations=total_recommendations,
            successful_explanations=successful_explanations,
            coverage_percentage=coverage_percentage,
            failed_explanations=failed_explanations,
            failure_reasons=dict(failure_reasons)
        )
    
    def evaluate_efficiency(
        self, 
        user_id: str, 
        track_id: str, 
        score: float,
        n_runs: int = 5
    ) -> EfficiencyMetrics:
        """Evaluate explanation generation efficiency"""
        
        generation_times = []
        
        for _ in range(n_runs):
            # Clear cache to test cold start
            if _ == 0:
                from .explainer import clear_explanation_cache
                clear_explanation_cache()
            
            # Time the explanation generation
            start_time = time.time()
            try:
                explanation = self.explainer.explain_recommendation(user_id, track_id, score)
                end_time = time.time()
                
                generation_time = end_time - start_time
                generation_times.append(generation_time)
                    
            except Exception as e:
                print(f"Error in efficiency test run {_}: {e}")
                continue
        
        if not generation_times:
            return EfficiencyMetrics(
                avg_generation_time=0.0,
                min_generation_time=0.0,
                max_generation_time=0.0
            )
        
        return EfficiencyMetrics(
            avg_generation_time=np.mean(generation_times),
            min_generation_time=np.min(generation_times),
            max_generation_time=np.max(generation_times)
        )
    
    def evaluate_user(
        self, 
        user_id: str, 
        recommendations: List[Tuple[str, float]],
        stability_runs: int = 3,
        efficiency_runs: int = 5
    ) -> UserEvaluationResult:
        """Evaluate explainability for a single user"""
        
        individual_explanations = []
        
        # Evaluate coverage
        coverage = self.evaluate_coverage(recommendations, user_id)
        
        # Evaluate stability and efficiency for first few recommendations
        stability_results = []
        efficiency_results = []
        
        for i, (track_id, score) in enumerate(recommendations[:5]):  # Test first 5 recommendations
            # Stability evaluation
            stability = self.evaluate_stability(user_id, track_id, score, stability_runs)
            stability_results.append(stability)
            
            # Efficiency evaluation
            efficiency = self.evaluate_efficiency(user_id, track_id, score, efficiency_runs)
            efficiency_results.append(efficiency)
            
            # Store individual explanation data
            try:
                explanation = self.explainer.explain_recommendation(user_id, track_id, score)
                individual_explanations.append({
                    "track_id": track_id,
                    "score": score,
                    "top_features": [c.feature_name for c in explanation.content_contributions[:3]],
                    "top_users": [uid for uid, _ in explanation.collaborative_insight.similar_users[:3]],
                    "cf_score": explanation.score_breakdown["collaborative_score"],
                    "content_score": explanation.score_breakdown["content_score"]
                })
            except Exception as e:
                individual_explanations.append({
                    "track_id": track_id,
                    "score": score,
                    "error": str(e)
                })
        
        # Aggregate stability metrics
        avg_stability = StabilityMetrics(
            feature_consistency=np.mean([s.feature_consistency for s in stability_results]),
            user_consistency=np.mean([s.user_consistency for s in stability_results]),
            score_consistency=np.mean([s.score_consistency for s in stability_results])
        )
        
        # Aggregate efficiency metrics
        avg_efficiency = EfficiencyMetrics(
            avg_generation_time=np.mean([e.avg_generation_time for e in efficiency_results]),
            min_generation_time=np.min([e.min_generation_time for e in efficiency_results]),
            max_generation_time=np.max([e.max_generation_time for e in efficiency_results])
        )
        
        return UserEvaluationResult(
            user_id=user_id,
            total_recommendations=len(recommendations),
            stability=avg_stability,
            coverage=coverage,
            efficiency=avg_efficiency,
            individual_explanations=individual_explanations
        )
    
    def evaluate_batch(
        self, 
        user_recommendations: Dict[str, List[Tuple[str, float]]],
        stability_runs: int = 3,
        efficiency_runs: int = 5
    ) -> BatchEvaluationResult:
        """Evaluate explainability for multiple users"""
        
        user_results = []
        
        for user_id, recommendations in user_recommendations.items():
            print(f"Evaluating user {user_id}...")
            user_result = self.evaluate_user(
                user_id, recommendations, stability_runs, efficiency_runs
            )
            user_results.append(user_result)
        
        # Aggregate results
        total_users = len(user_results)
        total_recommendations = sum(ur.total_recommendations for ur in user_results)
        
        # Average stability
        avg_stability = StabilityMetrics(
            feature_consistency=np.mean([ur.stability.feature_consistency for ur in user_results]),
            user_consistency=np.mean([ur.stability.user_consistency for ur in user_results]),
            score_consistency=np.mean([ur.stability.score_consistency for ur in user_results])
        )
        
        # Average coverage
        avg_coverage = CoverageMetrics(
            total_recommendations=total_recommendations,
            successful_explanations=sum(ur.coverage.successful_explanations for ur in user_results),
            coverage_percentage=np.mean([ur.coverage.coverage_percentage for ur in user_results]),
            failed_explanations=sum(ur.coverage.failed_explanations for ur in user_results),
            failure_reasons={}  # Aggregate failure reasons if needed
        )
        
        # Average efficiency
        avg_efficiency = EfficiencyMetrics(
            avg_generation_time=np.mean([ur.efficiency.avg_generation_time for ur in user_results]),
            min_generation_time=np.min([ur.efficiency.min_generation_time for ur in user_results]),
            max_generation_time=np.max([ur.efficiency.max_generation_time for ur in user_results])
        )
        
        return BatchEvaluationResult(
            total_users=total_users,
            total_recommendations=total_recommendations,
            avg_stability=avg_stability,
            avg_coverage=avg_coverage,
            avg_efficiency=avg_efficiency,
            user_results=user_results,
            evaluation_timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def save_results(self, results: BatchEvaluationResult, filepath: str) -> None:
        """Save evaluation results to JSON file"""
        results_dict = asdict(results)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(item) for item in obj]
            else:
                return convert_numpy(obj)
        
        results_dict = recursive_convert(results_dict)
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to {filepath}")
