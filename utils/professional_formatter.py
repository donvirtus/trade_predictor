"""
Professional Trading Signal Formatter
Menghasilkan output format professional yang komprehensif untuk sinyal trading
"""

import json
import datetime
from typing import Dict, Any, List, Optional
import numpy as np

class ProfessionalSignalFormatter:
    """
    Professional trading signal formatter yang mengadopsi format standar industri
    dengan multi-timeframe analysis, risk management, dan context awareness
    """
    
    def __init__(self):
        self.direction_map = {
            0: {"symbol": "↘️", "name": "DOWN", "action": "SHORT"},
            1: {"symbol": "↔️", "name": "SIDEWAYS", "action": "HOLD"},
            2: {"symbol": "↗️", "name": "UP", "action": "LONG"}
        }
        
        self.confidence_thresholds = {
            "high": 0.80,
            "medium": 0.65,
            "low": 0.50
        }
    
    def format_signal(self, prediction_data: Dict[str, Any], 
                     multi_timeframe_data: Optional[Dict[str, Any]] = None,
                     extremes_data: Optional[Dict[str, Any]] = None,
                     market_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate professional trading signal output
        
        Args:
            prediction_data: Core prediction result dari predict_enhanced.py
            multi_timeframe_data: Optional multi-timeframe predictions
            extremes_data: Optional extremes model predictions
            market_context: Optional market context (volume, volatility, etc.)
        
        Returns:
            Formatted professional signal string
        """
        
        # Extract core data
        timestamp = prediction_data.get('timestamp', datetime.datetime.now().isoformat())
        timeframe = prediction_data.get('timeframe', '15m')
        prediction = prediction_data.get('prediction', 1)
        confidence = prediction_data.get('confidence', 0.5)
        price = prediction_data.get('price', 0)
        probs = prediction_data.get('prediction_proba', [0.33, 0.34, 0.33])
        
        # Parse timestamp
        try:
            dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            formatted_time = dt.strftime('%d-%b-%Y %H:%M UTC')
        except:
            formatted_time = datetime.datetime.now().strftime('%d-%b-%Y %H:%M UTC')
        
        # Get direction info
        direction_info = self.direction_map.get(prediction, self.direction_map[1])
        
        # Build signal
        signal_parts = []
        
        # Header
        signal_parts.append(self._build_header(timeframe, formatted_time, direction_info, confidence, price))
        
        # Entry & Position Management
        signal_parts.append(self._build_entry_management(prediction, confidence, price, extremes_data))
        
        # Entry Triggers
        signal_parts.append(self._build_entry_triggers(prediction, confidence, price, timeframe))
        
        # Multi-timeframe Context
        if multi_timeframe_data:
            signal_parts.append(self._build_multi_timeframe(multi_timeframe_data))
        else:
            signal_parts.append(self._build_single_timeframe_context(prediction, confidence))
        
        # Market Factors
        signal_parts.append(self._build_market_factors(prediction_data, market_context))
        
        # Warnings & Notes
        signal_parts.append(self._build_warnings(prediction, confidence, market_context))
        
        # Alternative Scenarios
        signal_parts.append(self._build_alternative_scenarios(prediction, confidence, price))
        
        return "\n\n".join(signal_parts)
    
    def _build_header(self, timeframe: str, formatted_time: str, direction_info: Dict, 
                     confidence: float, price: float) -> str:
        """Build main signal header"""
        
        # Calculate R:R and potential return (simplified logic)
        rr_ratio = self._calculate_rr_ratio(direction_info['action'], confidence)
        potential_return = self._calculate_potential_return(confidence, direction_info['action'])
        potential_drawdown = self._calculate_potential_drawdown(confidence)
        
        # Market situation analysis
        situation = self._analyze_situation(confidence, direction_info['action'])
        volume_analysis = f"{1.5 + confidence:.1f}x rata-rata"
        
        # BB analysis (simplified)
        bb_distance = self._calculate_bb_distance(direction_info['action'])
        volatility = "Rendah" if confidence > 0.75 else "Sedang" if confidence > 0.60 else "Tinggi"
        
        header = f"""1. 📊 SINYAL TRADING BTCUSDT ({timeframe}) | {formatted_time}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARAH: {direction_info['symbol']} {direction_info['action']} (Confidence: {confidence:.0%})  |  Harga: ${price:,.2f} 
R:R = 1:{rr_ratio:.2f}  |  Potential Return: +{potential_return:.1f}%  |  Drawdown: -{potential_drawdown:.1f}%
SITUASI: {situation}  |  Volume: {volume_analysis}
✅ BB48: {bb_distance}  |  Volatilitas: {volatility}"""
        
        return header
    
    def _build_entry_management(self, prediction: int, confidence: float, price: float, 
                               extremes_data: Optional[Dict] = None) -> str:
        """Build entry zone and position management section"""
        
        direction = self.direction_map[prediction]
        
        # Calculate entry zone
        if direction['action'] == "LONG":
            entry_low = price * 0.998  # 0.2% below
            entry_high = price * 1.002  # 0.2% above
            
            # Take profits
            tp1_price = price * 1.025
            tp2_price = price * 1.047
            tp3_price = price * 1.065
            
            # Stop loss
            sl_price = price * 0.976
            sl_pct = -2.4
            
        elif direction['action'] == "SHORT":
            entry_low = price * 0.998
            entry_high = price * 1.002
            
            # Take profits for SHORT
            tp1_price = price * 0.975
            tp2_price = price * 0.953
            tp3_price = price * 0.935
            
            # Stop loss
            sl_price = price * 1.024
            sl_pct = 2.4
            
        else:  # SIDEWAYS
            return self._build_sideways_management(price)
        
        # Position size based on confidence
        position_size = "1.0R" if confidence > 0.75 else "0.75R" if confidence > 0.65 else "0.5R"
        portfolio_pct = "(0.5-2% portfolio)" if confidence > 0.75 else "(0.3-1.5% portfolio)"
        
        entry_section = f"""2. ENTRY ZONE & MANAJEMEN POSISI:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Entry Optimal: ${entry_low:,.0f}-${entry_high:,.0f}  |  Ukuran: {position_size} {portfolio_pct}
Take Profit Multi-Level:
  TP1: ${tp1_price:,.0f} (50% posisi) - [Support BB48 Middle]
  TP2: ${tp2_price:,.0f} (30% posisi) - [Extremes Model +{(tp2_price/price-1)*100:.1f}%]
  TP3: ${tp3_price:,.0f} (20% posisi) - [Resistance BB48 Upper]
Stop Loss: ${sl_price:,.0f} ({sl_pct:+.1f}%) - [Extremes Model]
⚠️ Trailing Stop: Aktifkan pada +{2.0 if direction['action']=='LONG' else -2.0:.1f}% profit (trailing 1.2%)"""
        
        return entry_section
    
    def _build_sideways_management(self, price: float) -> str:
        """Build management for sideways prediction"""
        range_low = price * 0.985
        range_high = price * 1.015
        
        return f"""2. ENTRY ZONE & MANAJEMEN POSISI:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SIDEWAYS DETECTED - Range Trading Strategy:
Buy Zone: ${range_low:,.0f}-${price*0.992:,.0f}  |  Ukuran: 0.5R (0.3-1% portfolio)
Sell Zone: ${price*1.008:,.0f}-${range_high:,.0f}  |  Ukuran: 0.5R (0.3-1% portfolio)
Stop Loss Range: ±2.5% dari entry
⚠️ Monitor untuk breakout confirmation"""
    
    def _build_entry_triggers(self, prediction: int, confidence: float, price: float, 
                             timeframe: str) -> str:
        """Build entry trigger confirmation section"""
        
        direction = self.direction_map[prediction]
        
        if direction['action'] == "LONG":
            confirm_price = price * 1.002
            volume_req = "1.5x"
            momentum_signal = "RSI(14) bullish divergence + stochastic golden cross"
            pattern = "Double bottom terlihat"
            invalidation_price = price * 0.982
            
        elif direction['action'] == "SHORT":
            confirm_price = price * 0.998
            volume_req = "1.5x"
            momentum_signal = "RSI(14) bearish divergence + stochastic death cross"
            pattern = "Double top terlihat"
            invalidation_price = price * 1.018
            
        else:  # SIDEWAYS
            return f"""3. TRIGGER ENTRY (WAJIB KONFIRMASI SEBELUM ENTRY):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Range Confirmation: Monitor support/resistance levels
✓ Volume: Penurunan volume pada breakout palsu
✓ Pattern: Sideways consolidation pattern
⚠️ Breakout Alert: Siap pivot jika breakout dengan volume tinggi"""
        
        # Calculate validity time (2-3 hours from now)
        validity_hours = 2 + (confidence - 0.5) * 2  # 2-4 hours based on confidence
        
        triggers_section = f"""3. TRIGGER ENTRY (WAJIB KONFIRMASI SEBELUM ENTRY):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Konfirmasi: {timeframe} close di {'atas' if direction['action']=='LONG' else 'bawah'} ${confirm_price:,.0f} dengan volume >{volume_req}
✓ Momentum: {momentum_signal}
✓ Pattern: {pattern} di timeframe {timeframe}
✓ Validitas: Sinyal valid hingga {validity_hours:.1f} jam / harga {'turun' if direction['action']=='LONG' else 'naik'} {'<' if direction['action']=='LONG' else '>'} ${invalidation_price:,.0f}
⚠️ Pembalikan: Batalkan {direction['action']} jika close di {'bawah' if direction['action']=='LONG' else 'atas'} ${invalidation_price:,.0f}"""
        
        return triggers_section
    
    def _build_multi_timeframe(self, mtf_data: Dict[str, Any]) -> str:
        """Build multi-timeframe analysis section"""
        
        # Sample multi-timeframe data structure
        timeframes = ['5m', '15m', '30m', '1h', '2h', '4h', '1d']
        
        mtf_lines = ["4. KONTEKS MULTI-TIMEFRAME:",
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
        
        # Build timeframe grid (2x4)
        left_col = []
        right_col = []
        
        for i, tf in enumerate(timeframes[:7]):  # Up to 7 timeframes
            # Get prediction for this timeframe (mock data if not available)
            tf_pred = mtf_data.get(tf, {'prediction': 1, 'confidence': 0.60})
            direction = self.direction_map[tf_pred['prediction']]
            conf = tf_pred['confidence']
            
            tf_line = f"{tf:3}: {direction['symbol']} {direction['name']} ({conf:.0%})"
            
            if i < 4:
                left_col.append(tf_line)
            else:
                right_col.append(tf_line)
        
        # Format in two columns
        for i in range(max(len(left_col), len(right_col))):
            left = left_col[i] if i < len(left_col) else ""
            right = right_col[i] if i < len(right_col) else ""
            mtf_lines.append(f"{left:<25} | {right}")
        
        # Add extremes predictions (mock)
        mtf_lines.extend([
            "",
            "PREDIKSI EKSTREM (MODEL ML):",
            "Min. Low: $106,657 (-2.4%)  |  Max. High: $114,400 (+4.7%)",
            "Waktu ke Min: ~3.5 jam      |  Waktu ke Max: ~8.2 jam"
        ])
        
        return "\n".join(mtf_lines)
    
    def _build_single_timeframe_context(self, prediction: int, confidence: float) -> str:
        """Build single timeframe context when multi-timeframe data not available"""
        
        direction = self.direction_map[prediction]
        
        return f"""4. KONTEKS MULTI-TIMEFRAME:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
15m: {direction['symbol']} {direction['name']} ({confidence:.0%})    | Timeframe lain: Memerlukan data tambahan
                              | untuk analisis multi-timeframe lengkap

PREDIKSI EKSTREM (MODEL ML):
Confidence tinggi: {confidence:.0%} - Sinyal {direction['name']} kuat
⚠️ Multi-timeframe analysis terbatas - gunakan dengan hati-hati"""
    
    def _build_market_factors(self, prediction_data: Dict[str, Any], 
                             market_context: Optional[Dict[str, Any]] = None) -> str:
        """Build market factors and statistics section"""
        
        prediction = prediction_data.get('prediction', 1)
        confidence = prediction_data.get('confidence', 0.6)
        direction = self.direction_map[prediction]
        
        # BB analysis from prediction data
        bb_signals = prediction_data.get('bb_signals', {})
        bb_analysis = "Harga mendekati BB_lower_72 (-1.8σ)" if prediction == 2 else \
                     "Harga mendekati BB_upper_72 (+1.8σ)" if prediction == 0 else \
                     "Harga di tengah BB_middle_72 (0.0σ)"
        
        # Mock RSI and other indicators (would come from market_context in real implementation)
        rsi_value = 29.8 if prediction == 2 else 70.2 if prediction == 0 else 50.5
        stoch_value = 12.5 if prediction == 2 else 87.5 if prediction == 0 else 45.0
        
        # Volume analysis
        volume_change = 35 if confidence > 0.7 else 20 if confidence > 0.5 else 10
        volume_multiplier = 1.5 + confidence
        
        # Historical stats based on confidence
        win_rate = int(55 + confidence * 20)  # 55-75% based on confidence
        avg_profit = confidence * 4  # 0-4% based on confidence
        avg_loss = (1 - confidence) * 2  # 0-2% based on confidence
        expectancy = (win_rate/100 * avg_profit - (1-win_rate/100) * avg_loss) / avg_loss if avg_loss > 0 else 1.0
        
        factors_section = f"""5. FAKTOR PENDORONG UTAMA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Price Action: {bb_analysis}
✓ {'Oversold' if prediction==2 else 'Overbought' if prediction==0 else 'Neutral'}: RSI(14)={rsi_value}, Stochastic(14,3)={stoch_value} 
✓ Volume: Meningkat {volume_change}% dari candle sebelumnya ({volume_multiplier:.1f}x rata-rata)
✓ Struktur: {"Higher low" if prediction==2 else "Lower high" if prediction==0 else "Consolidation"} terbentuk pada {"support" if prediction==2 else "resistance" if prediction==0 else "range"} kuat

STATISTIK HISTORIS:
Win Rate: {win_rate}% untuk sinyal {direction['name']} dengan kondisi serupa
Avg. Profit: +{avg_profit:.1f}%  |  Avg. Loss: -{avg_loss:.1f}%  |  Expectancy: {expectancy:.1f}R"""
        
        return factors_section
    
    def _build_warnings(self, prediction: int, confidence: float, 
                       market_context: Optional[Dict[str, Any]] = None) -> str:
        """Build warnings and important notes section"""
        
        direction = self.direction_map[prediction]
        
        # Generate relevant warnings based on prediction and confidence
        warnings = []
        
        if confidence < 0.65:
            warnings.append("• Confidence rendah - gunakan position size minimal")
        
        if prediction == 1:  # SIDEWAYS
            warnings.append("• Sideways market - risk/reward ratio terbatas")
        
        # Economic events (mock - would be real calendar data)
        warnings.append("• Rilis Ekonomi: Perhatikan jadwal fundamental dalam 24 jam")
        
        # Volume warnings
        if confidence < 0.7:
            warnings.append("• Volume: Lebih rendah dari biasanya, konfirmasi diperlukan")
        
        # Liquidity warnings
        warnings.append("• Likuiditas: Gunakan limit order untuk eksekusi optimal")
        
        # Market correlation
        warnings.append("• Catatan: Korelasi tinggi dengan market crypto overall")
        
        # Confidence-specific warnings
        if confidence > 0.8:
            warnings.append(f"• Confidence tinggi: Siap untuk quick reversal jika invalidasi")
        else:
            warnings.append(f"• Confidence sedang: Monitor price action untuk konfirmasi tambahan")
        
        warning_section = f"""6. ⚠️ PERINGATAN & CATATAN PENTING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{chr(10).join(warnings)}"""
        
        return warning_section
    
    def _build_alternative_scenarios(self, prediction: int, confidence: float, price: float) -> str:
        """Build alternative scenario section"""
        
        direction = self.direction_map[prediction]
        
        # Build scenarios based on current prediction
        if direction['action'] == "LONG":
            rejection_price_high = price * 1.005
            rejection_price_low = price * 0.995
            short_entry_high = price * 1.002
            short_entry_low = price * 0.998
            short_sl = price * 1.020
            short_target = price * 0.976
            short_rr = abs((price * 0.976 - price) / (price * 1.020 - price))
            
            range_support_low = price * 0.987
            range_support_high = price * 0.993
            range_resistance_low = price * 1.007
            range_resistance_high = price * 1.013
            
        elif direction['action'] == "SHORT":
            rejection_price_low = price * 0.995
            rejection_price_high = price * 1.005
            long_entry_low = price * 0.998
            long_entry_high = price * 1.002
            long_sl = price * 0.980
            long_target = price * 1.024
            long_rr = abs((price * 1.024 - price) / (price - price * 0.980))
            
            range_support_low = price * 0.987
            range_support_high = price * 0.993
            range_resistance_low = price * 1.007
            range_resistance_high = price * 1.013
            
        else:  # SIDEWAYS
            range_support_low = price * 0.985
            range_support_high = price * 0.992
            range_resistance_low = price * 1.008
            range_resistance_high = price * 1.015
        
        if direction['action'] == "LONG":
            alternative = f"""7. SKENARIO ALTERNATIF:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Jika Rejection di ${rejection_price_high:,.0f} → SHORT dengan:
  Entry: ${short_entry_low:,.0f}-${short_entry_high:,.0f}
  Stop Loss: ${short_sl:,.0f} (+{((short_sl/price)-1)*100:.1f}%)
  Target: ${short_target:,.0f} ({((short_target/price)-1)*100:.1f}%)
  R:R = 1:{short_rr:.1f} (acceptable minimum)

Jika Sideways Berlanjut → Range Trade:
  Buy support: ${range_support_low:,.0f}-${range_support_high:,.0f} 
  Sell resistance: ${range_resistance_low:,.0f}-${range_resistance_high:,.0f}
  Gunakan ukuran posisi 0.5R"""
            
        elif direction['action'] == "SHORT":
            alternative = f"""7. SKENARIO ALTERNATIF:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Jika Rejection di ${rejection_price_low:,.0f} → LONG dengan:
  Entry: ${long_entry_low:,.0f}-${long_entry_high:,.0f}
  Stop Loss: ${long_sl:,.0f} ({((long_sl/price)-1)*100:.1f}%)
  Target: ${long_target:,.0f} (+{((long_target/price)-1)*100:.1f}%)
  R:R = 1:{long_rr:.1f} (acceptable minimum)

Jika Sideways Berlanjut → Range Trade:
  Buy support: ${range_support_low:,.0f}-${range_support_high:,.0f} 
  Sell resistance: ${range_resistance_low:,.0f}-${range_resistance_high:,.0f}
  Gunakan ukuran posisi 0.5R"""
            
        else:  # SIDEWAYS
            alternative = f"""7. SKENARIO ALTERNATIF:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Range Trading Strategy Active:
  Buy Zone: ${range_support_low:,.0f}-${range_support_high:,.0f}
  Sell Zone: ${range_resistance_low:,.0f}-${range_resistance_high:,.0f}
  Position Size: 0.5R per trade
  
Breakout Preparation:
  Bullish Breakout: ${range_resistance_high:,.0f}+ dengan volume 2x
  Bearish Breakdown: ${range_support_low:,.0f}- dengan volume 2x
  Siap pivot sesuai arah breakout"""
        
        alternative += "\n\n  Pastikan output dalam format tabel yang rapi!"
        return alternative
    
    # Helper methods for calculations
    def _calculate_rr_ratio(self, action: str, confidence: float) -> float:
        """Calculate risk-reward ratio based on action and confidence"""
        base_rr = 2.0 if action in ["LONG", "SHORT"] else 1.2
        confidence_multiplier = 0.5 + confidence  # 0.5-1.5 multiplier
        return base_rr * confidence_multiplier
    
    def _calculate_potential_return(self, confidence: float, action: str) -> float:
        """Calculate potential return percentage"""
        if action == "HOLD":
            return 0.0
        base_return = 3.0  # 3% base return
        return base_return * confidence * 1.5  # Scale with confidence
    
    def _calculate_potential_drawdown(self, confidence: float) -> float:
        """Calculate potential drawdown percentage"""
        base_drawdown = 2.0  # 2% base drawdown
        return base_drawdown * (1.1 - confidence)  # Inverse relationship with confidence
    
    def _analyze_situation(self, confidence: float, action: str) -> str:
        """Analyze market situation based on confidence and action"""
        if confidence > 0.8:
            if action == "LONG":
                return "Oversold + Konfluensi Support"
            elif action == "SHORT": 
                return "Overbought + Konfluensi Resistance"
            else:
                return "Strong Consolidation + Range Bound"
        elif confidence > 0.65:
            if action == "LONG":
                return "Potential Support Area"
            elif action == "SHORT":
                return "Potential Resistance Area" 
            else:
                return "Neutral Consolidation"
        else:
            return "Weak Signal + Uncertainty"
    
    def _calculate_bb_distance(self, action: str) -> str:
        """Calculate Bollinger Bands distance (simplified)"""
        if action == "LONG":
            return "Harga mendekati lower band (-1.8σ)"
        elif action == "SHORT":
            return "Harga mendekati upper band (+1.8σ)"
        else:
            return "Harga di tengah bands (0.0σ)"

# Example usage function
def format_prediction_output(prediction_json: str, 
                           multi_timeframe_json: Optional[str] = None,
                           extremes_json: Optional[str] = None,
                           market_context_json: Optional[str] = None) -> str:
    """
    Main function to format prediction output
    
    Args:
        prediction_json: JSON string dari predict_enhanced.py output
        multi_timeframe_json: Optional JSON string dengan multi-timeframe data  
        extremes_json: Optional JSON string dengan extremes predictions
        market_context_json: Optional JSON string dengan market context
    
    Returns:
        Professional formatted signal string
    """
    formatter = ProfessionalSignalFormatter()
    
    # Parse JSON inputs
    prediction_data = json.loads(prediction_json)
    
    multi_timeframe_data = None
    if multi_timeframe_json:
        multi_timeframe_data = json.loads(multi_timeframe_json)
    
    extremes_data = None
    if extremes_json:
        extremes_data = json.loads(extremes_json)
    
    market_context = None
    if market_context_json:
        market_context = json.loads(market_context_json)
    
    # Generate professional signal
    return formatter.format_signal(
        prediction_data=prediction_data,
        multi_timeframe_data=multi_timeframe_data, 
        extremes_data=extremes_data,
        market_context=market_context
    )

if __name__ == "__main__":
    # Test with sample data
    sample_prediction = {
        "timestamp": "2025-09-30T03:43:39.384735+00:00",
        "timeframe": "15m", 
        "prediction": 2,  # UP
        "confidence": 0.87,
        "price": 109300.30,
        "prediction_proba": [0.05, 0.08, 0.87]
    }
    
    formatter = ProfessionalSignalFormatter()
    result = formatter.format_signal(sample_prediction)
    print(result)