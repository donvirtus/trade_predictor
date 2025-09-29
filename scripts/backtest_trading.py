#!/usr/bin/env python3
"""
Proper Trading Backtesting Script

Implementasi backtest yang benar dengan:
1. Precision/Recall metrics (bukan accuracy)
2. Sharpe ratio dan max drawdown
3. Profit factor dan win rate
4. Risk-adjusted returns
5. Sequential crossing strategy evaluation
"""

import os
import sys
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import argparse
import json
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.logging import get_logger

logger = get_logger('backtest_trading')


class TradingBacktester:
    def __init__(self, timeframe: str = "15m"):
        self.timeframe = timeframe
        self.db_path = f"data/db/btc_{timeframe}.sqlite"
        
    def load_predictions(self, predictions_file: str) -> pd.DataFrame:
        """Load model predictions"""
        if predictions_file.endswith('.csv'):
            df = pd.read_csv(predictions_file)
        else:
            # Assume JSON format from predict_enhanced.py
            with open(predictions_file, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            
        # Ensure timestamp column
        if 'timestamp' not in df.columns:
            raise ValueError("Predictions must have timestamp column")
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')
    
    def load_price_data(self) -> pd.DataFrame:
        """Load actual price data for backtesting"""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found: {self.db_path}")
            
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            "SELECT timestamp, open, high, low, close, volume FROM features ORDER BY timestamp ASC",
            conn
        )
        conn.close()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def merge_predictions_with_prices(self, predictions: pd.DataFrame, 
                                    prices: pd.DataFrame) -> pd.DataFrame:
        """Merge predictions with actual price data"""
        merged = pd.merge(predictions, prices, on='timestamp', how='inner')
        return merged.sort_values('timestamp').reset_index(drop=True)
    
    def simulate_trades(self, df: pd.DataFrame, 
                       take_profit_pct: float = 2.0,
                       stop_loss_pct: float = 1.5,
                       confidence_threshold: float = 0.7,
                       max_hold_periods: int = 20) -> pd.DataFrame:
        """
        Simulate trading based on model predictions
        
        Parameters:
        - take_profit_pct: TP level (%)
        - stop_loss_pct: SL level (%)  
        - confidence_threshold: Minimum prediction confidence
        - max_hold_periods: Maximum holding periods before timeout
        """
        trades = []
        position = None
        entry_price = None
        entry_time = None
        entry_index = None
        
        for i, row in df.iterrows():
            current_price = row['close']
            current_time = row['timestamp']
            
            # Check exit conditions if in position
            if position is not None:
                pnl_pct = 0
                outcome = 'TIMEOUT'
                
                if position == 'LONG':
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                    if pnl_pct >= take_profit_pct:
                        outcome = 'TP'
                    elif pnl_pct <= -stop_loss_pct:
                        outcome = 'SL'
                    elif i - entry_index >= max_hold_periods:
                        outcome = 'TIMEOUT'
                    else:
                        continue  # Hold position
                        
                elif position == 'SHORT':
                    pnl_pct = (entry_price - current_price) / entry_price * 100
                    if pnl_pct >= take_profit_pct:
                        outcome = 'TP'
                    elif pnl_pct <= -stop_loss_pct:
                        outcome = 'SL'
                    elif i - entry_index >= max_hold_periods:
                        outcome = 'TIMEOUT'
                    else:
                        continue  # Hold position
                
                # Close position
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'direction': position,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct,
                    'outcome': outcome,
                    'hold_periods': i - entry_index,
                    'prediction_confidence': row.get('confidence', 0.5)
                })
                
                position = None
                entry_price = None
                entry_time = None
                entry_index = None
                
            # Check entry conditions if no position
            else:
                prediction = row.get('prediction', 1)  # Default: SIDEWAYS
                confidence = row.get('confidence', 0.5)
                
                if confidence >= confidence_threshold:
                    if prediction == 2:  # UP prediction
                        position = 'LONG'
                        entry_price = current_price
                        entry_time = current_time
                        entry_index = i
                    elif prediction == 0:  # DOWN prediction  
                        position = 'SHORT'
                        entry_price = current_price
                        entry_time = current_time
                        entry_index = i
        
        return pd.DataFrame(trades)
    
    def compute_backtest_metrics(self, trades: pd.DataFrame) -> Dict:
        """
        ✅ PROPER TRADING METRICS (bukan accuracy!)
        """
        if trades.empty:
            return {'error': 'No trades generated'}
            
        metrics = {}
        
        # Basic trade statistics
        total_trades = len(trades)
        winning_trades = (trades['pnl_pct'] > 0).sum()
        losing_trades = (trades['pnl_pct'] < 0).sum()
        
        metrics['total_trades'] = int(total_trades)
        metrics['winning_trades'] = int(winning_trades)
        metrics['losing_trades'] = int(losing_trades)
        metrics['win_rate'] = float(winning_trades / total_trades) if total_trades > 0 else 0.0
        
        # PnL statistics
        total_pnl = trades['pnl_pct'].sum()
        avg_pnl = trades['pnl_pct'].mean()
        avg_winner = trades[trades['pnl_pct'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0.0
        avg_loser = trades[trades['pnl_pct'] < 0]['pnl_pct'].mean() if losing_trades > 0 else 0.0
        
        metrics['total_pnl_pct'] = float(total_pnl)
        metrics['avg_pnl_pct'] = float(avg_pnl)
        metrics['avg_winner_pct'] = float(avg_winner)
        metrics['avg_loser_pct'] = float(avg_loser)
        
        # ✅ PROFIT FACTOR
        gross_profit = trades[trades['pnl_pct'] > 0]['pnl_pct'].sum()
        gross_loss = abs(trades[trades['pnl_pct'] < 0]['pnl_pct'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        metrics['profit_factor'] = float(profit_factor)
        
        # ✅ SHARPE RATIO (risk-adjusted returns)
        pnl_std = trades['pnl_pct'].std()
        sharpe_ratio = avg_pnl / pnl_std if pnl_std > 0 else 0.0
        metrics['sharpe_ratio'] = float(sharpe_ratio)
        
        # ✅ MAX DRAWDOWN
        cumulative_pnl = trades['pnl_pct'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = running_max - cumulative_pnl
        max_drawdown = drawdown.max()
        metrics['max_drawdown_pct'] = float(max_drawdown)
        
        # Outcome analysis
        outcome_counts = trades['outcome'].value_counts().to_dict()
        for outcome, count in outcome_counts.items():
            metrics[f'outcome_{outcome.lower()}'] = int(count)
            metrics[f'outcome_{outcome.lower()}_pct'] = float(count / total_trades * 100)
            
        # Direction analysis
        for direction in ['LONG', 'SHORT']:
            dir_trades = trades[trades['direction'] == direction]
            if not dir_trades.empty:
                dir_wins = (dir_trades['pnl_pct'] > 0).sum()
                dir_total = len(dir_trades)
                metrics[f'{direction.lower()}_trades'] = int(dir_total)
                metrics[f'{direction.lower()}_win_rate'] = float(dir_wins / dir_total)
                metrics[f'{direction.lower()}_avg_pnl'] = float(dir_trades['pnl_pct'].mean())
        
        # Holding period statistics
        avg_hold_periods = trades['hold_periods'].mean()
        max_hold_periods = trades['hold_periods'].max()
        metrics['avg_hold_periods'] = float(avg_hold_periods)
        metrics['max_hold_periods'] = int(max_hold_periods)
        
        # ✅ TRADING-SPECIFIC PRECISION/RECALL
        # Precision: Of all our BUY signals, how many were profitable?
        long_trades = trades[trades['direction'] == 'LONG']
        short_trades = trades[trades['direction'] == 'SHORT']
        
        if not long_trades.empty:
            long_precision = (long_trades['pnl_pct'] > 0).sum() / len(long_trades)
            metrics['long_precision'] = float(long_precision)
            
        if not short_trades.empty:
            short_precision = (short_trades['pnl_pct'] > 0).sum() / len(short_trades)
            metrics['short_precision'] = float(short_precision)
        
        # Overall precision (profitable trades / total trades)
        metrics['overall_precision'] = metrics['win_rate']
        
        # Risk-reward ratio
        if avg_loser != 0:
            risk_reward_ratio = abs(avg_winner / avg_loser)
            metrics['risk_reward_ratio'] = float(risk_reward_ratio)
        
        # Kelly Criterion (optimal position sizing)
        if metrics['win_rate'] > 0 and avg_loser != 0:
            kelly_pct = (metrics['win_rate'] * abs(avg_winner) - (1 - metrics['win_rate']) * abs(avg_loser)) / abs(avg_loser)
            metrics['kelly_pct'] = float(kelly_pct * 100)
        
        return metrics
    
    def run_backtest(self, predictions_file: str, 
                    take_profit_pct: float = 2.0,
                    stop_loss_pct: float = 1.5,
                    confidence_threshold: float = 0.7,
                    max_hold_periods: int = 20) -> Dict:
        """Run complete backtest"""
        logger.info(f"Running backtest for {self.timeframe} timeframe")
        
        # Load data
        predictions = self.load_predictions(predictions_file)
        prices = self.load_price_data()
        
        # Merge predictions with prices
        df = self.merge_predictions_with_prices(predictions, prices)
        
        logger.info(f"Merged data: {len(df)} rows")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Simulate trades
        trades = self.simulate_trades(
            df, 
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            confidence_threshold=confidence_threshold,
            max_hold_periods=max_hold_periods
        )
        
        logger.info(f"Generated {len(trades)} trades")
        
        # Compute metrics
        metrics = self.compute_backtest_metrics(trades)
        
        # Add backtest parameters
        metrics['backtest_params'] = {
            'timeframe': self.timeframe,
            'take_profit_pct': take_profit_pct,
            'stop_loss_pct': stop_loss_pct,
            'confidence_threshold': confidence_threshold,
            'max_hold_periods': max_hold_periods,
            'data_points': len(df),
            'date_range': [str(df['timestamp'].min()), str(df['timestamp'].max())]
        }
        
        return {
            'metrics': metrics,
            'trades': trades,
            'merged_data': df
        }
    
    def save_results(self, results: Dict, output_dir: str = "data/backtest_results"):
        """Save backtest results"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_file = os.path.join(output_dir, f"backtest_metrics_{self.timeframe}_{timestamp}.json")
        with open(metrics_file, 'w') as f:
            json.dump(results['metrics'], f, indent=2, default=str)
        
        # Save trades
        trades_file = os.path.join(output_dir, f"backtest_trades_{self.timeframe}_{timestamp}.csv")
        results['trades'].to_csv(trades_file, index=False)
        
        logger.info(f"Saved backtest results:")
        logger.info(f"  Metrics: {metrics_file}")
        logger.info(f"  Trades: {trades_file}")
        
        return metrics_file, trades_file
    
    def print_summary(self, metrics: Dict):
        """Print backtest summary"""
        if 'error' in metrics:
            print(f"❌ Backtest Error: {metrics['error']}")
            return
            
        print("🎯 TRADING BACKTEST RESULTS")
        print("=" * 50)
        
        print(f"📊 Basic Statistics:")
        print(f"  Total Trades: {metrics.get('total_trades', 0)}")
        print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
        print(f"  Total PnL: {metrics.get('total_pnl_pct', 0):+.2f}%")
        print(f"  Average PnL: {metrics.get('avg_pnl_pct', 0):+.3f}%")
        
        print(f"\n💰 Performance Metrics:")
        print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"  Risk/Reward: {metrics.get('risk_reward_ratio', 0):.2f}")
        
        print(f"\n📈 Trading Precision:")
        print(f"  Overall Precision: {metrics.get('overall_precision', 0):.2%}")
        if 'long_precision' in metrics:
            print(f"  Long Precision: {metrics.get('long_precision', 0):.2%}")
        if 'short_precision' in metrics:
            print(f"  Short Precision: {metrics.get('short_precision', 0):.2%}")
        
        print(f"\n⏱️  Holding Analysis:")
        print(f"  Average Hold: {metrics.get('avg_hold_periods', 0):.1f} periods")
        print(f"  Max Hold: {metrics.get('max_hold_periods', 0)} periods")
        
        print(f"\n🎲 Outcomes:")
        for outcome in ['TP', 'SL', 'TIMEOUT']:
            count = metrics.get(f'outcome_{outcome.lower()}', 0)
            pct = metrics.get(f'outcome_{outcome.lower()}_pct', 0)
            print(f"  {outcome}: {count} ({pct:.1f}%)")
        
        print(f"\n💡 Kelly Criterion: {metrics.get('kelly_pct', 0):+.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Trading Strategy Backtester')
    parser.add_argument('predictions', help='Path to predictions file (CSV or JSON)')
    parser.add_argument('--timeframe', default='15m', 
                       choices=['5m', '15m', '30m', '1h', '2h', '4h', '6h'],
                       help='Timeframe for backtesting')
    parser.add_argument('--take-profit', type=float, default=2.0,
                       help='Take profit percentage (default: 2.0)')
    parser.add_argument('--stop-loss', type=float, default=1.5, 
                       help='Stop loss percentage (default: 1.5)')
    parser.add_argument('--confidence', type=float, default=0.7,
                       help='Minimum prediction confidence (default: 0.7)')
    parser.add_argument('--max-hold', type=int, default=20,
                       help='Maximum holding periods (default: 20)')
    parser.add_argument('--output-dir', default='data/backtest_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    try:
        backtester = TradingBacktester(timeframe=args.timeframe)
        
        results = backtester.run_backtest(
            predictions_file=args.predictions,
            take_profit_pct=args.take_profit,
            stop_loss_pct=args.stop_loss,
            confidence_threshold=args.confidence,
            max_hold_periods=args.max_hold
        )
        
        # Print summary
        backtester.print_summary(results['metrics'])
        
        # Save results
        metrics_file, trades_file = backtester.save_results(results, args.output_dir)
        
        print(f"\n✅ Backtest completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        print(f"❌ Backtest failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()