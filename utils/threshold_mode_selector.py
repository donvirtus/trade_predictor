#!/usr/bin/env python3
"""
Smart Threshold Mode Auto-Selector

Automatically analyzes data and recommends optimal threshold_mode
based on data characteristics, trading goals, and performance metrics.

Author: AI Assistant  
Created: 2025-10-01
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

# Import existing utilities
from utils.threshold_balancer import ThresholdAutoBalancer, load_data_for_balancing
from utils.adaptive_thresholds import AdaptiveThresholdCalculator

logger = logging.getLogger(__name__)

class ThresholdModeSelector:
    """
    Intelligent threshold mode selection based on data analysis
    """
    
    def __init__(self, config=None):
        self.config = config
        self.analysis_results = {}
        
    def analyze_data_characteristics(self, db_path: str, timeframe: str) -> Dict:
        """
        Analyze data characteristics untuk inform threshold mode selection
        
        Returns:
            Dict dengan data characteristics analysis
        """
        try:
            df = load_data_for_balancing(db_path, limit_rows=3000)
            if df.empty:
                return {}
            
            # Calculate price movements untuk different horizons
            horizons = [1, 5, 10, 20]
            characteristics = {
                'timeframe': timeframe,
                'total_samples': len(df),
                'volatility_metrics': {},
                'movement_analysis': {},
                'trend_characteristics': {}
            }
            
            for horizon in horizons:
                returns = df['close'].pct_change(horizon).shift(-horizon) * 100
                returns_clean = returns.dropna()
                
                if len(returns_clean) > 50:
                    # Movement distribution analysis
                    characteristics['movement_analysis'][f'h{horizon}'] = {
                        'mean_abs_movement': float(returns_clean.abs().mean()),
                        'median_abs_movement': float(returns_clean.abs().median()),
                        'std_movement': float(returns_clean.std()),
                        'percentile_80': float(returns_clean.abs().quantile(0.8)),
                        'percentile_90': float(returns_clean.abs().quantile(0.9)),
                        'max_movement': float(returns_clean.abs().max()),
                        'skewness': float(returns_clean.skew()) if len(returns_clean) > 3 else 0
                    }
            
            # Overall volatility assessment
            daily_volatility = df['close'].pct_change().std() * 100
            characteristics['volatility_metrics'] = {
                'daily_volatility_pct': float(daily_volatility),
                'volatility_regime': self._classify_volatility_regime(daily_volatility),
                'price_range_pct': float((df['close'].max() - df['close'].min()) / df['close'].mean() * 100)
            }
            
            # Trend characteristics
            price_trend = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
            characteristics['trend_characteristics'] = {
                'overall_trend_pct': float(price_trend),
                'trend_direction': 'bullish' if price_trend > 5 else 'bearish' if price_trend < -5 else 'sideways',
                'trend_consistency': float(self._calculate_trend_consistency(df))
            }
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Error analyzing data characteristics: {e}")
            return {}
    
    def _classify_volatility_regime(self, daily_vol: float) -> str:
        """Classify volatility regime"""
        if daily_vol > 3.0:
            return "high"
        elif daily_vol > 1.5:
            return "medium"
        else:
            return "low"
    
    def _calculate_trend_consistency(self, df: pd.DataFrame) -> float:
        """Calculate trend consistency (0-1 scale)"""
        try:
            # Simple trend consistency: correlation between time and price
            time_index = np.arange(len(df))
            correlation = np.corrcoef(time_index, df['close'])[0, 1]
            return abs(correlation)  # Absolute correlation indicates consistency
        except:
            return 0.0
    
    def test_threshold_modes(self, db_path: str, timeframe: str) -> Dict:
        """
        Test all threshold modes and compare performance
        
        Returns:
            Dict dengan performance comparison
        """
        try:
            results = {}
            df = load_data_for_balancing(db_path, limit_rows=2000)
            
            if df.empty:
                return {}
            
            # Test 1: Manual threshold (current adaptive manual)
            manual_results = self._test_manual_threshold(df, timeframe)
            if manual_results:
                results['manual'] = manual_results
            
            # Test 2: Auto-balanced threshold
            auto_balanced_results = self._test_auto_balanced_threshold(df)
            if auto_balanced_results:
                results['auto_balanced'] = auto_balanced_results
            
            # Test 3: Adaptive volatility-based  
            adaptive_results = self._test_adaptive_threshold(df, timeframe)
            if adaptive_results:
                results['adaptive'] = adaptive_results
            
            return results
            
        except Exception as e:
            logger.error(f"Error testing threshold modes: {e}")
            return {}
    
    def _test_manual_threshold(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Test current manual threshold approach"""
        try:
            # Use current config values
            base_threshold = 1.0  # From config
            manual_multipliers = {
                "5m": 0.15, "15m": 0.30, "30m": 0.40, "1h": 0.50,
                "2h": 0.65, "4h": 0.80, "6h": 1.00
            }
            
            threshold = base_threshold * manual_multipliers.get(timeframe, 0.5)
            
            # Test distribution
            returns = df['close'].pct_change(20).shift(-20) * 100
            returns_clean = returns.dropna()
            
            if len(returns_clean) < 50:
                return {}
            
            up = (returns_clean > threshold).sum()
            down = (returns_clean < -threshold).sum() 
            sideways = len(returns_clean) - up - down
            
            distribution = {
                'UP': (up / len(returns_clean)) * 100,
                'SIDEWAYS': (sideways / len(returns_clean)) * 100,
                'DOWN': (down / len(returns_clean)) * 100
            }
            
            # Calculate balance score
            balance_score = 100 - abs(33.3 - distribution['UP']) - abs(33.3 - distribution['DOWN']) - abs(33.4 - distribution['SIDEWAYS'])
            
            return {
                'threshold_pct': threshold,
                'distribution': distribution,
                'balance_score': max(0, balance_score),
                'trading_signals_pct': distribution['UP'] + distribution['DOWN'],
                'method': 'manual_timeframe_based'
            }
            
        except Exception as e:
            logger.error(f"Error testing manual threshold: {e}")
            return {}
    
    def _test_auto_balanced_threshold(self, df: pd.DataFrame) -> Dict:
        """Test auto-balanced threshold approach"""
        try:
            balancer = ThresholdAutoBalancer()
            result = balancer.auto_balance_timeframe(df, horizon=20, method='binary_search')
            
            if result and 'optimal_threshold' in result:
                return {
                    'threshold_pct': result['optimal_threshold'],
                    'distribution': result['distribution'],
                    'balance_score': result['balance_score'],
                    'trading_signals_pct': result['trading_percentage'],
                    'method': 'auto_balanced_optimization'
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error testing auto-balanced threshold: {e}")
            return {}
    
    def _test_adaptive_threshold(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Test adaptive volatility-based threshold"""
        try:
            # Simple adaptive approach: threshold based on recent volatility
            returns = df['close'].pct_change().dropna()
            recent_volatility = returns.tail(50).std() * 100  # Last 50 periods
            
            # Adaptive threshold: scale with volatility
            base_vol = 1.5  # Base volatility assumption
            vol_multiplier = recent_volatility / base_vol
            adaptive_threshold = 0.5 * vol_multiplier  # Adaptive calculation
            
            # Constrain reasonable range
            adaptive_threshold = max(0.1, min(2.0, adaptive_threshold))
            
            # Test distribution
            test_returns = df['close'].pct_change(20).shift(-20) * 100
            test_returns_clean = test_returns.dropna()
            
            if len(test_returns_clean) < 50:
                return {}
            
            up = (test_returns_clean > adaptive_threshold).sum()
            down = (test_returns_clean < -adaptive_threshold).sum()
            sideways = len(test_returns_clean) - up - down
            
            distribution = {
                'UP': (up / len(test_returns_clean)) * 100,
                'SIDEWAYS': (sideways / len(test_returns_clean)) * 100,
                'DOWN': (down / len(test_returns_clean)) * 100
            }
            
            balance_score = 100 - abs(33.3 - distribution['UP']) - abs(33.3 - distribution['DOWN']) - abs(33.4 - distribution['SIDEWAYS'])
            
            return {
                'threshold_pct': adaptive_threshold,
                'distribution': distribution,
                'balance_score': max(0, balance_score),
                'trading_signals_pct': distribution['UP'] + distribution['DOWN'],
                'method': 'adaptive_volatility_based',
                'volatility_input': recent_volatility
            }
            
        except Exception as e:
            logger.error(f"Error testing adaptive threshold: {e}")
            return {}
    
    def recommend_threshold_mode(self, db_path: str, timeframe: str, 
                                trading_style: str = 'balanced') -> Dict:
        """
        Main recommendation function
        
        Args:
            db_path: Path to database
            timeframe: Trading timeframe
            trading_style: 'conservative', 'balanced', 'aggressive'
            
        Returns:
            Dict dengan recommendation dan reasoning
        """
        try:
            print(f"\n🤖 ANALYZING OPTIMAL THRESHOLD MODE for {timeframe}")
            print("=" * 60)
            
            # Step 1: Analyze data characteristics
            print("📊 Step 1: Analyzing data characteristics...")
            characteristics = self.analyze_data_characteristics(db_path, timeframe)
            
            if not characteristics:
                return {'error': 'Failed to analyze data characteristics'}
            
            # Step 2: Test all threshold modes
            print("🧪 Step 2: Testing threshold modes...")
            mode_results = self.test_threshold_modes(db_path, timeframe)
            
            if not mode_results:
                return {'error': 'Failed to test threshold modes'}
            
            # Step 3: Score and rank modes
            print("📈 Step 3: Scoring and ranking...")
            scored_modes = self._score_threshold_modes(mode_results, characteristics, trading_style)
            
            # Step 4: Generate recommendation
            recommendation = self._generate_recommendation(scored_modes, characteristics, trading_style)
            
            # Step 5: Display results
            self._display_analysis_results(characteristics, mode_results, recommendation)
            
            return {
                'recommendation': recommendation,
                'characteristics': characteristics,
                'mode_results': mode_results,
                'scored_modes': scored_modes
            }
            
        except Exception as e:
            logger.error(f"Error in recommend_threshold_mode: {e}")
            return {'error': str(e)}
    
    def _score_threshold_modes(self, mode_results: Dict, characteristics: Dict, 
                              trading_style: str) -> List[Tuple]:
        """Score threshold modes based on multiple criteria"""
        
        scores = []
        
        for mode, result in mode_results.items():
            score = 0
            reasons = []
            
            # Criteria 1: Balance Score (30% weight)
            balance_weight = 0.3
            balance_score = result.get('balance_score', 0)
            score += balance_score * balance_weight
            
            if balance_score > 80:
                reasons.append("Excellent class balance")
            elif balance_score > 60:
                reasons.append("Good class balance")
            else:
                reasons.append("Poor class balance")
            
            # Criteria 2: Trading Signals Percentage (25% weight)
            signals_weight = 0.25
            trading_pct = result.get('trading_signals_pct', 50)
            
            # Optimal trading percentage depends on style
            if trading_style == 'conservative':
                optimal_trading = 50  # 50% trading signals
            elif trading_style == 'aggressive':  
                optimal_trading = 70  # 70% trading signals
            else:  # balanced
                optimal_trading = 60  # 60% trading signals
            
            signals_score = 100 - abs(trading_pct - optimal_trading)
            score += max(0, signals_score) * signals_weight
            
            # Criteria 3: Volatility Appropriateness (20% weight)
            vol_weight = 0.2
            vol_regime = characteristics.get('volatility_metrics', {}).get('volatility_regime', 'medium')
            threshold = result.get('threshold_pct', 0.5)
            
            vol_appropriateness = self._assess_volatility_appropriateness(vol_regime, threshold)
            score += vol_appropriateness * vol_weight
            
            # Criteria 4: Method Complexity/Reliability (15% weight)
            complexity_weight = 0.15
            method_score = self._assess_method_reliability(mode, result)
            score += method_score * complexity_weight
            
            # Criteria 5: Timeframe Suitability (10% weight)
            timeframe_weight = 0.1
            timeframe_score = self._assess_timeframe_suitability(mode, characteristics['timeframe'])
            score += timeframe_score * timeframe_weight
            
            scores.append((mode, score, reasons, result))
        
        # Sort by score (descending)
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    def _assess_volatility_appropriateness(self, vol_regime: str, threshold: float) -> float:
        """Assess if threshold is appropriate for volatility regime"""
        if vol_regime == 'high':
            # High volatility needs higher threshold
            if threshold > 0.8:
                return 100
            elif threshold > 0.5:
                return 70
            else:
                return 30
        elif vol_regime == 'low':
            # Low volatility needs lower threshold
            if threshold < 0.3:
                return 100
            elif threshold < 0.6:
                return 70
            else:
                return 30
        else:  # medium
            # Medium volatility: moderate threshold
            if 0.3 <= threshold <= 0.8:
                return 100
            else:
                return 50
    
    def _assess_method_reliability(self, mode: str, result: Dict) -> float:
        """Assess method reliability and complexity"""
        reliability_scores = {
            'manual': 85,  # High reliability, human expertise
            'auto_balanced': 95,  # Highest reliability, data-driven
            'adaptive': 75  # Good but more complex
        }
        return reliability_scores.get(mode, 50)
    
    def _assess_timeframe_suitability(self, mode: str, timeframe: str) -> float:
        """Assess mode suitability for specific timeframe"""
        
        # Timeframe categorization
        short_tf = ['5m', '15m', '30m']
        medium_tf = ['1h', '2h', '4h']
        long_tf = ['6h', '12h', '1d']
        
        if timeframe in short_tf:
            # Short timeframes: adaptive works best
            scores = {'adaptive': 100, 'manual': 85, 'auto_balanced': 70}
        elif timeframe in medium_tf:
            # Medium timeframes: all work well
            scores = {'auto_balanced': 100, 'adaptive': 90, 'manual': 80}
        else:  # long timeframes
            # Long timeframes: auto_balanced and manual work best
            scores = {'auto_balanced': 100, 'manual': 90, 'adaptive': 70}
        
        return scores.get(mode, 50)
    
    def _generate_recommendation(self, scored_modes: List[Tuple], 
                               characteristics: Dict, trading_style: str) -> Dict:
        """Generate final recommendation with reasoning"""
        
        if not scored_modes:
            return {'mode': 'manual', 'confidence': 'low', 'reasoning': ['No analysis data available']}
        
        best_mode, best_score, best_reasons, best_result = scored_modes[0]
        
        # Confidence based on score difference and absolute score
        if best_score > 85:
            confidence = 'high'
        elif best_score > 70:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        # Generate reasoning
        reasoning = [f"Chosen: {best_mode} (score: {best_score:.1f}/100)"]
        reasoning.extend(best_reasons)
        
        # Add specific insights
        vol_regime = characteristics.get('volatility_metrics', {}).get('volatility_regime', 'unknown')
        reasoning.append(f"Volatility regime: {vol_regime}")
        
        trend = characteristics.get('trend_characteristics', {}).get('trend_direction', 'unknown')
        reasoning.append(f"Market trend: {trend}")
        
        # Configuration suggestion
        config_suggestion = self._generate_config_suggestion(best_mode, best_result)
        
        return {
            'recommended_mode': best_mode,
            'confidence': confidence,
            'score': round(best_score, 1),
            'reasoning': reasoning,
            'config_suggestion': config_suggestion,
            'alternative_modes': [(mode, round(score, 1)) for mode, score, _, _ in scored_modes[1:3]]
        }
    
    def _generate_config_suggestion(self, mode: str, result: Dict) -> Dict:
        """Generate specific config.yaml suggestions"""
        
        threshold = result.get('threshold_pct', 0.5)
        
        if mode == 'manual':
            return {
                'threshold_mode': 'adaptive',
                'sideways_threshold_pct': 1.0,
                'adaptive_thresholds': {
                    'method': 'manual',
                    'manual_multipliers': {
                        '15m': round(threshold, 3)
                    }
                }
            }
        elif mode == 'auto_balanced':
            return {
                'threshold_mode': 'auto_balanced',
                'sideways_threshold_pct': round(threshold, 3),
                'auto_balance': {
                    'target_distribution': {'UP': 33.0, 'SIDEWAYS': 34.0, 'DOWN': 33.0}
                }
            }
        else:  # adaptive
            return {
                'threshold_mode': 'adaptive',
                'adaptive_thresholds': {
                    'method': 'auto_volatility',
                    'auto_update': True
                }
            }
    
    def _display_analysis_results(self, characteristics: Dict, mode_results: Dict, 
                                recommendation: Dict):
        """Display comprehensive analysis results"""
        
        print("\n📋 DATA CHARACTERISTICS:")
        print("-" * 30)
        vol_metrics = characteristics.get('volatility_metrics', {})
        print(f"Volatility Regime: {vol_metrics.get('volatility_regime', 'unknown')}")
        print(f"Daily Volatility: {vol_metrics.get('daily_volatility_pct', 0):.2f}%")
        
        trend_metrics = characteristics.get('trend_characteristics', {})
        print(f"Trend Direction: {trend_metrics.get('trend_direction', 'unknown')}")
        print(f"Overall Trend: {trend_metrics.get('overall_trend_pct', 0):+.1f}%")
        
        print("\n🧪 THRESHOLD MODE COMPARISON:")
        print("-" * 40)
        for mode, result in mode_results.items():
            dist = result.get('distribution', {})
            print(f"\n{mode.upper()}:")
            print(f"  Threshold: {result.get('threshold_pct', 0):.3f}%")
            print(f"  Distribution: UP={dist.get('UP', 0):.1f}% | SIDE={dist.get('SIDEWAYS', 0):.1f}% | DOWN={dist.get('DOWN', 0):.1f}%")
            print(f"  Balance Score: {result.get('balance_score', 0):.1f}/100")
            print(f"  Trading Signals: {result.get('trading_signals_pct', 0):.1f}%")
        
        print(f"\n🏆 RECOMMENDATION:")
        print("-" * 20)
        print(f"Best Mode: {recommendation['recommended_mode'].upper()}")
        print(f"Confidence: {recommendation['confidence'].upper()}")
        print(f"Score: {recommendation['score']}/100")
        
        print(f"\nReasoning:")
        for reason in recommendation['reasoning']:
            print(f"  • {reason}")
        
        print(f"\n⚙️ CONFIG SUGGESTION:")
        config = recommendation['config_suggestion']
        print(f"threshold_mode: \"{config['threshold_mode']}\"")
        for key, value in config.items():
            if key != 'threshold_mode':
                print(f"{key}: {value}")


def main():
    """Main function for CLI usage"""
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description='Smart Threshold Mode Selector')
    parser.add_argument('--db-path', help='Specific database path')
    parser.add_argument('--timeframe', default='15m', help='Timeframe to analyze')
    parser.add_argument('--trading-style', default='balanced', 
                       choices=['conservative', 'balanced', 'aggressive'],
                       help='Trading style preference')
    
    args = parser.parse_args()
    
    selector = ThresholdModeSelector()
    
    if args.db_path:
        # Analyze specific database
        result = selector.recommend_threshold_mode(args.db_path, args.timeframe, args.trading_style)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"\n✅ Analysis complete for {args.db_path}")
    else:
        # Auto-discover and analyze all databases
        db_pattern = 'data/db/*_15m.sqlite'  # Focus on 15m for demo
        db_files = glob.glob(db_pattern)
        
        if not db_files:
            print("❌ No databases found matching pattern")
            return
        
        print(f"🔍 Found {len(db_files)} databases to analyze")
        
        for db_path in db_files:
            print(f"\n{'='*80}")
            pair = os.path.basename(db_path).replace('_15m.sqlite', '').upper()
            print(f"📊 ANALYZING {pair} 15m")
            
            result = selector.recommend_threshold_mode(db_path, '15m', args.trading_style)
            
            if 'error' not in result:
                rec = result['recommendation']
                print(f"\n🎯 QUICK RESULT: {rec['recommended_mode'].upper()} mode recommended "
                      f"(confidence: {rec['confidence']}, score: {rec['score']}/100)")


if __name__ == '__main__':
    main()