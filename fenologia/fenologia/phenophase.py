"""
Módulo local v2 para extração de estágios fenológicos de séries temporais de NDVI.
Versão melhorada com melhor detecção de ciclos (safra e safrinha) usando:
1. Suavização adaptativa da série temporal
2. Detecção robuста de mínimos locais (solo exposto)
3. Segmentação de ciclos independentes
4. Ajuste de logística dupla em cada ciclo
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks, medfilt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def adaptive_smoothing(ndvi_values: np.ndarray, dates: np.ndarray, 
                      method: str = 'savgol', target_noise_std: float = None) -> np.ndarray:
    """
    Aplica suavização adaptativa à série temporal.
    
    Args:
        ndvi_values: Array de valores NDVI
        dates: Array de datas (usado para calcular densidade de dados)
        method: 'savgol', 'median', ou 'both'
        target_noise_std: Se None, detecta automaticamente
    
    Returns:
        Array suavizado
    """
    if len(ndvi_values) < 5:
        return ndvi_values
    
    # Calcula densidade de dados (pontos por dia)
    total_days = (dates[-1] - dates[0]) / np.timedelta64(1, 'D')
    points_per_day = len(ndvi_values) / max(total_days, 1)
    
    # Define tamanho de janela adaptativo
    if points_per_day > 0.5:  # Dados densos (diários ou próximos)
        window = max(5, min(15, int(7 * points_per_day)))  # 5-15 dias
    elif points_per_day > 0.1:  # Dados moderados (semanais)
        window = max(3, min(7, int(3 * points_per_day)))
    else:  # Dados esparsos (mensais)
        window = max(3, min(5, int(points_per_day)))
    
    # Garante que é ímpar
    window = window if window % 2 == 1 else window + 1
    
    smoothed = ndvi_values.copy()
    
    if method in ['median', 'both']:
        # Primeira passagem: filtro mediano para remover outliers
        smoothed = medfilt(smoothed, kernel_size=window)
    
    if method in ['savgol', 'both']:
        # Segunda passagem: Savitzky-Golay para suavizar mantendo pontos de inflexão
        polyorder = min(2, window - 2)
        if len(smoothed) > window:
            smoothed = savgol_filter(smoothed, window_length=window, polyorder=polyorder)
    
    return smoothed


def detect_trough_peaks(ndvi_values: np.ndarray, dates: np.ndarray,
                       method: str = 'adaptive', min_distance_days: int = 30,
                       quantile_threshold: float = None) -> np.ndarray:
    """
    Detecta mínimos locais (vales) na série temporal NDVI.
    Esses vales representam períodos de solo exposto entre ciclos.
    Versão melhorada com maior sensibilidade e filtragem de falsos positivos.
    AGORA COM MELHOR SUPORTE PARA SAFRA/SAFRINHA (dupla safra).
    
    Args:
        ndvi_values: Array de valores NDVI
        dates: Array de datas
        method: 'adaptive', 'quantile', ou 'derivative'
        min_distance_days: Distância mínima entre vales em dias
        quantile_threshold: Percentil para threshold automático (None = auto)
    
    Returns:
        Array de índices dos mínimos locais ordenados
    """
    if len(ndvi_values) < 5:
        return np.array([])
    
    # Converte distância em dias para índices
    total_days = (dates[-1] - dates[0]) / np.timedelta64(1, 'D')
    min_distance_idx = max(1, int(min_distance_days * len(ndvi_values) / total_days))

    # Inverte a série para usar find_peaks
    ndvi_inverted = -ndvi_values

    # Calcula estatísticas
    ndvi_std = np.std(ndvi_values)
    ndvi_min = np.min(ndvi_values)
    ndvi_max = np.max(ndvi_values)
    ndvi_mean = np.mean(ndvi_values)
    ndvi_range = ndvi_max - ndvi_min

    # Detecção de vales com prominência mínima realista (evita pegar ruído de alta frequência)
    prominence_very_low = max(0.015, ndvi_std * 0.20)
    distance_very_loose = max(1, int(min_distance_idx * 0.6))  # Até 40% mais curto que o mínimo
    
    all_troughs, all_props = find_peaks(ndvi_inverted, distance=distance_very_loose, 
                                        prominence=prominence_very_low)
    
    if len(all_troughs) == 0:
        # Fallback: pega os N mínimos mais profundos
        n_expected = max(2, int(total_days / 150))  # Espera mais ciclos (1 a cada ~150 dias para safra+safrinha)
        all_troughs = np.argsort(ndvi_values)[:n_expected]
        all_troughs = np.sort(all_troughs)
        return all_troughs
    
    # FILTRAGEM INTELIGENTE: Separa vales reais de ruído
    # Vales reais devem cumprir UM desses critérios:
    # 1. Estar abaixo de um threshold de solo exposto
    # 2. Ser um mínimo LOCAL significativo (comparado aos vizinhos imediatos)
    # 3. Marcar transição clara entre ciclos (mudança abrupta na derivada)
    
    threshold_solo_exposto = ndvi_mean - (ndvi_std * 0.6)
    threshold_minimo_local = ndvi_mean - (ndvi_std * 0.3)
    
    vales_classificados = []
    
    for trough_idx in all_troughs:
        trough_value = ndvi_values[trough_idx]
        score = 0
        
        # Critério 1: Solo exposto (muito baixo)
        if trough_value <= threshold_solo_exposto:
            score += 3
        elif trough_value <= threshold_minimo_local:
            score += 2
        
        # Critério 2: Mínimo local significativo (comparado aos vizinhos)
        window_size = max(2, int(min_distance_idx * 0.3))  # Janela adaptativa
        left_idx = max(0, trough_idx - window_size)
        right_idx = min(len(ndvi_values), trough_idx + window_size)
        
        neighbors = ndvi_values[left_idx:right_idx]
        neighbors_min = np.min(neighbors)
        neighbors_mean = np.mean(neighbors)
        
        # Se for o mínimo local ou bem perto dele, é um bom candidato
        if abs(trough_value - neighbors_min) < 0.01 and trough_value < (neighbors_mean - ndvi_std * 0.15):
            score += 2
        
        # Critério 3: Detecta "knees" - mudanças abruptas em NDVI (entre ciclos)
        if trough_idx > 2 and trough_idx < len(ndvi_values) - 2:
            derivative_before = ndvi_values[trough_idx] - np.mean(ndvi_values[max(0, trough_idx-3):trough_idx])
            derivative_after = np.mean(ndvi_values[trough_idx+1:min(len(ndvi_values), trough_idx+4)]) - ndvi_values[trough_idx]
            
            # Se há mudança abrupta (típico de transição entre ciclos)
            if abs(derivative_before) > ndvi_std * 0.15 or abs(derivative_after) > ndvi_std * 0.15:
                score += 1
        
        # Score >= 3: exige que o vale seja profundo OU profundo + mínimo local.
        # Rejeita dips intra-ciclo que só são mínimos locais rasos (score=2 por
        # critério 2 apenas) — esses fragmentam peaks gaussianos reais.
        if score >= 3:
            vales_classificados.append((trough_idx, score, trough_value))
    
    # Ordena por score para priorizar vales reais
    vales_classificados.sort(key=lambda x: (-x[1], x[2]))  # Maior score, menor NDVI
    
    # Vincula ao tipo de cultura: max ~2 troughs por comprimento mínimo de ciclo
    max_vales = max(2, int(total_days / (min_distance_days * 2.0)))
    vales_filtrados = [v[0] for v in vales_classificados[:max_vales]]
    
    if len(vales_filtrados) > 0:
        # Ordena temporalmente
        vales_filtrados = np.array(sorted(vales_filtrados))
        
        # Remove vales duplicados/muito próximos (< 75 dias)
        min_trough_dist_strong = max(1, int(75 * len(ndvi_values) / total_days))
        final_vales = []
        for v in vales_filtrados:
            if len(final_vales) == 0 or v - final_vales[-1] >= min_trough_dist_strong:
                final_vales.append(v)
        
        return np.array(final_vales)
    else:
        # Fallback: os vales mais profundos
        return np.array(sorted([v[0] for v in vales_classificados[:max(2, int(total_days / 200))]]))


def segment_cycles(ndvi_values: np.ndarray, dates: np.ndarray, troughs: np.ndarray,
                  min_cycle_length_days: int = 45, extend_edges: bool = True) -> List[Dict[str, Any]]:
    """
    Segmenta a série temporal em ciclos independentes baseado nos vales.
    Versão refinada para criar ciclos entre vales consecutivos.
    MELHORADA para detectar safra/safrinha com ciclos curtos e adjacentes.
    
    Args:
        ndvi_values: Array de valores NDVI
        dates: Array de datas
        troughs: Índices dos vales (mínimos locais)
        min_cycle_length_days: Comprimento mínimo do ciclo em dias
        extend_edges: Se True, estende ciclos até as bordas da série
    
    Returns:
        Lista de dicionários com informações de cada ciclo
    """
    cycles = []
    
    if len(troughs) == 0:
        # Se não há vales, considera toda a série como um ciclo
        cycles.append({
            'cycle_num': 1,
            'start_idx': 0,
            'end_idx': len(ndvi_values) - 1,
            'start_date': pd.Timestamp(dates[0]),
            'end_date': pd.Timestamp(dates[-1]),
            'length_days': (dates[-1] - dates[0]) / np.timedelta64(1, 'D'),
            'min_ndvi': np.min(ndvi_values),
            'max_ndvi': np.max(ndvi_values),
        })
        return cycles
    
    # Garante que troughs está ordenado e remove duplicatas
    troughs = np.unique(troughs)
    
    total_days = (dates[-1] - dates[0]) / np.timedelta64(1, 'D')
    min_trough_distance = max(1, int(65 * len(ndvi_values) / total_days))
    
    filtered_troughs = []
    for trough in troughs:
        if len(filtered_troughs) == 0 or (trough - filtered_troughs[-1] >= min_trough_distance):
            filtered_troughs.append(trough)
    
    troughs = np.array(filtered_troughs)
    
    # Estratégia melhorada de segmentação para safra/safrinha
    # Agora cria ciclos de forma mais inteligente:
    # - Ciclo inicial: do início até primeiro vale
    # - Ciclos intermediários: cada um começa onde o anterior termina
    # - Ciclo final: do último vale até o fim
    
    # Ciclo inicial
    if len(troughs) > 0:
        start_idx = 0
        end_idx = troughs[0]
        
        cycle_length_days = (dates[end_idx] - dates[start_idx]) / np.timedelta64(1, 'D')
        
        if cycle_length_days >= min_cycle_length_days:
            cycles.append({
                'cycle_num': len(cycles) + 1,
                'start_idx': int(start_idx),
                'end_idx': int(end_idx),
                'start_date': pd.Timestamp(dates[start_idx]),
                'end_date': pd.Timestamp(dates[end_idx]),
                'length_days': cycle_length_days,
                'min_ndvi': np.min(ndvi_values[start_idx:end_idx + 1]),
                'max_ndvi': np.max(ndvi_values[start_idx:end_idx + 1]),
                'trough_idx': -1,
                'trough_ndvi': np.nan,
            })
    
    # Ciclos intermediários: cada um vai de um vale ao próximo
    for i in range(len(troughs) - 1):
        start_idx = troughs[i]
        end_idx = troughs[i + 1]
        
        cycle_length_days = (dates[end_idx] - dates[start_idx]) / np.timedelta64(1, 'D')
        
        # MUDANÇA IMPORTANTE: Reduzido threshold mínimo de ciclo
        # Antes: respeitava min_cycle_length_days
        # Agora: aceita ciclos com até 25 dias (típico de safrinha)
        min_cycle_short = min_cycle_length_days * 0.80
        
        if cycle_length_days >= min_cycle_short:
            cycles.append({
                'cycle_num': len(cycles) + 1,
                'start_idx': int(start_idx),
                'end_idx': int(end_idx),
                'start_date': pd.Timestamp(dates[start_idx]),
                'end_date': pd.Timestamp(dates[end_idx]),
                'length_days': cycle_length_days,
                'min_ndvi': np.min(ndvi_values[start_idx:end_idx + 1]),
                'max_ndvi': np.max(ndvi_values[start_idx:end_idx + 1]),
                'trough_idx': int(troughs[i + 1]),
                'trough_ndvi': ndvi_values[troughs[i + 1]],
            })
    
    # Ciclo final: do último vale até o fim
    if len(troughs) > 0:
        start_idx = troughs[-1]
        end_idx = len(ndvi_values) - 1
        
        cycle_length_days = (dates[end_idx] - dates[start_idx]) / np.timedelta64(1, 'D')
        min_cycle_short = min_cycle_length_days * 0.55
        
        if cycle_length_days >= min_cycle_short:
            cycles.append({
                'cycle_num': len(cycles) + 1,
                'start_idx': int(start_idx),
                'end_idx': int(end_idx),
                'start_date': pd.Timestamp(dates[start_idx]),
                'end_date': pd.Timestamp(dates[end_idx]),
                'length_days': cycle_length_days,
                'min_ndvi': np.min(ndvi_values[start_idx:end_idx + 1]),
                'max_ndvi': np.max(ndvi_values[start_idx:end_idx + 1]),
                'trough_idx': int(troughs[-1]),
                'trough_ndvi': ndvi_values[troughs[-1]],
            })
    
    return cycles


def detect_vegetation_peaks(
    ndvi_values: np.ndarray,
    dates: np.ndarray,
    min_distance_days: int = 90,
) -> np.ndarray:
    """
    Detecta picos vegetativos (máximos locais) diretamente na série EVI.

    Não depende de vales nem de sazonalidade: procura máximos locais acima
    da média da série, com proeminência mínima adaptativa e distância mínima
    entre picos configurável por cultura.

    Args:
        ndvi_values: Array de valores EVI (suavizado recomendado).
        dates: Array de datas (numpy datetime64).
        min_distance_days: Distância mínima entre picos em dias.

    Returns:
        Array de índices dos picos, ordenados temporalmente.
    """
    if len(ndvi_values) < 5:
        return np.array([int(np.argmax(ndvi_values))])

    total_days = float((dates[-1] - dates[0]) / np.timedelta64(1, 'D'))
    min_distance_idx = max(1, int(min_distance_days * len(ndvi_values) / total_days))

    ndvi_std = np.std(ndvi_values)
    ndvi_mean = np.mean(ndvi_values)

    # Proeminência mínima adaptativa: pico deve se destacar pelo menos 25%
    # do desvio padrão da série.  Piso baixo (0.03) para não sufocar culturas
    # perenes (café, citrus, dendê) que têm EVI de baixa variabilidade.
    prominence_min = max(0.03, ndvi_std * 0.25)

    peaks, _ = find_peaks(
        ndvi_values,
        distance=min_distance_idx,
        prominence=prominence_min,
    )

    if len(peaks) == 0:
        return np.array([int(np.argmax(ndvi_values))])

    return peaks


def segment_around_peaks(
    ndvi_values: np.ndarray,
    dates: np.ndarray,
    peaks: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Define janelas de ciclo ao redor de cada pico vegetativo.

    A janela de cada ciclo vai do ponto médio entre picos consecutivos
    (ou borda da série) até o próximo ponto médio, garantindo que cada
    pico fique aproximadamente no centro da sua janela.

    Args:
        ndvi_values: Array completo de valores EVI.
        dates: Array completo de datas.
        peaks: Índices dos picos (saída de detect_vegetation_peaks).

    Returns:
        Lista de dicionários de ciclo compatíveis com fit_curve_to_cycle.
    """
    cycles = []
    n = len(ndvi_values)

    for i, peak_idx in enumerate(peaks):
        if i == 0:
            # Para o primeiro ciclo, usa o vale antes do pico como borda esquerda
            # em vez do índice 0. Isso evita incluir a cauda descendente de um
            # ciclo anterior que torna o ajuste da logística ambíguo.
            if peak_idx > 1:
                trough_idx = int(np.argmin(ndvi_values[:peak_idx]))
                left = trough_idx if trough_idx > 0 else 0
            else:
                left = 0
        else:
            left = int((int(peaks[i - 1]) + peak_idx) // 2)

        if i == len(peaks) - 1:
            # Para o último ciclo, usa o vale depois do pico como borda direita.
            if peak_idx < n - 2:
                tail = ndvi_values[peak_idx:]
                trough_rel = int(np.argmin(tail))
                right = peak_idx + trough_rel if trough_rel > 0 else n - 1
            else:
                right = n - 1
        else:
            right = int((peak_idx + int(peaks[i + 1])) // 2)

        length_days = float((dates[right] - dates[left]) / np.timedelta64(1, 'D'))

        cycles.append({
            'cycle_num': i + 1,
            'start_idx': int(left),
            'end_idx': int(right),
            'peak_idx': int(peak_idx),
            'start_date': pd.Timestamp(dates[left]),
            'end_date': pd.Timestamp(dates[right]),
            'length_days': length_days,
            'min_ndvi': float(np.min(ndvi_values[left:right + 1])),
            'max_ndvi': float(ndvi_values[peak_idx]),
            # Ciclos que tocam a borda da série são truncados: o flanco ausente
            # reduz o R² do fit mesmo com um pico real e bem demarcado.
            'at_series_start': (left == 0),
            'at_series_end':   (right == n - 1),
        })

    return cycles


def gaussian(x: np.ndarray, amplitude: float, mean: float, std: float, offset: float) -> np.ndarray:
    """Gaussiana simétrica — mantida para compatibilidade."""
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * std ** 2)) + offset


def double_logistic(x: np.ndarray, amplitude: float,
                    m1: float, k1: float,
                    m2: float, k2: float,
                    offset: float) -> np.ndarray:
    """
    Função logística dupla para modelagem de picos vegetativos.

    f(t) = offset + amplitude · L_rise(t) · L_fall(t)

    onde:
      L_rise(t) = 1 / (1 + exp(-k1 · (t − m1)))   sigmoide de ascensão
      L_fall(t) = 1 / (1 + exp( k2 · (t − m2)))   sigmoide de declínio

    Parâmetros:
      amplitude : altura do pico acima da linha de base
      m1, k1   : ponto de inflexão e taxa de ascensão (k1 > 0)
                 SOS analítico ≈ m1 − ln(3)/k1  (onde L_rise = 0.25)
      m2, k2   : ponto de inflexão e taxa de declínio (k2 > 0, m2 > m1)
                 EOS analítico ≈ m2 + ln(3)/k2  (onde L_fall = 0.25)
      offset   : linha de base (EVI mínimo)

    Vantagem sobre a Gaussiana: m1/m2 podem estar fora da janela da série,
    permitindo capturar picos truncados no início e fim da série temporal.
    """
    x = np.asarray(x, dtype=float)
    L_rise = 1.0 / (1.0 + np.exp(-k1 * (x - m1)))
    L_fall = 1.0 / (1.0 + np.exp( k2 * (x - m2)))
    return offset + amplitude * L_rise * L_fall


# Mantém asymmetric_gaussian como alias de compatibilidade (não usar em código novo)
def asymmetric_gaussian(x: np.ndarray, amplitude: float, mean: float,
                        std_left: float, std_right: float, offset: float) -> np.ndarray:
    """Legado — use double_logistic para código novo."""
    x = np.asarray(x, dtype=float)
    std = np.where(x < mean, std_left, std_right)
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * std ** 2)) + offset


def classify_season_type(pos_date: pd.Timestamp) -> str:
    """
    Classifica um ciclo como 'safra' ou 'safrinha' pelo DOY do pico vegetativo (POS).

    Safra (principal): POS próximo do início das chuvas no Brasil (out–mar),
        DOY ≤ 90 (jan–mar) ou DOY ≥ 274 (out–dez).
    Safrinha (segunda safra): POS no meio do ano seco (abr–set), DOY 91–273.

    Args:
        pos_date: Data do pico vegetativo (POS).

    Returns:
        'safra' ou 'safrinha'
    """
    doy = pos_date.day_of_year
    if doy <= 90 or doy >= 274:
        return 'safra'
    return 'safrinha'


def fit_curve_to_cycle(ndvi_values: np.ndarray, dates: np.ndarray, cycle: Dict[str, Any],
                       quality_threshold: float = 0.6) -> Dict[str, Any]:
    """
    Ajusta uma logística dupla a um ciclo específico e extrai parâmetros fenológicos.

    Args:
        ndvi_values: Array completo de valores NDVI
        dates: Array completo de datas
        cycle: Dicionário do ciclo
        quality_threshold: Threshold mínimo de R² para considerar fit bem-sucedido

    Returns:
        Dicionário com parâmetros da logística dupla e estágios fenológicos
    """
    start_idx = cycle['start_idx']
    end_idx = cycle['end_idx']

    # Extrai dados do ciclo
    ndvi_cycle = ndvi_values[start_idx:end_idx + 1]
    dates_cycle = dates[start_idx:end_idx + 1]

    # Converte datas para dias desde o início do ciclo
    days_since_start = np.array([(d - dates_cycle[0]) / np.timedelta64(1, 'D')
                                  for d in dates_cycle], dtype=float)

    amplitude_init = float(np.max(ndvi_cycle) - np.min(ndvi_cycle))
    offset_init    = float(np.min(ndvi_cycle))
    window_len     = float(days_since_start[-1] - days_since_start[0])

    peak_pos_idx  = int(np.argmax(ndvi_cycle))
    peak_pos_days = float(days_since_start[peak_pos_idx])

    # Chutes iniciais para a logística dupla
    # m1 ≈ ponto de inflexão da subida (~30% antes do pico)
    # m2 ≈ ponto de inflexão da descida (~30% depois do pico)
    rise_window = max(peak_pos_days, 1.0)
    fall_window = max(window_len - peak_pos_days, 1.0)

    m1_init = peak_pos_days - rise_window * 0.3
    m2_init = peak_pos_days + fall_window * 0.3
    # Steepness: ln(3) / (metade da janela de subida/descida)
    k1_init = max(0.02, float(np.log(3)) / max(rise_window * 0.4, 1.0))
    k2_init = max(0.02, float(np.log(3)) / max(fall_window * 0.4, 1.0))

    initial_guess = [amplitude_init, m1_init, k1_init, m2_init, k2_init, offset_init]

    # m1 e m2 podem estar fora da janela visível (séries truncadas nas bordas):
    # permite extrapolação de até 50% do comprimento da janela para cada lado.
    slack = window_len * 0.5
    lower_bounds = [0.01, -slack,           0.005, peak_pos_days, 0.005, -0.5]
    upper_bounds = [1.5,  peak_pos_days,    2.0,   window_len + slack, 2.0, float(np.max(ndvi_cycle))]

    for i in range(len(initial_guess)):
        initial_guess[i] = float(np.clip(initial_guess[i], lower_bounds[i], upper_bounds[i]))

    try:
        popt, _ = curve_fit(
            double_logistic,
            days_since_start,
            ndvi_cycle,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=15000,
            method='trf',
        )

        amplitude, m1, k1, m2, k2, offset = popt

        # Valida parâmetros degenerados
        if amplitude < 0.01 or k1 < 1e-4 or k2 < 1e-4 or m2 <= m1:
            return {'fit_success': False, 'reason': 'Parâmetros degenerados', 'cycle': cycle}

        # POS: máximo numérico da curva ajustada
        t_dense   = np.linspace(days_since_start[0], days_since_start[-1], 2000)
        y_dense   = double_logistic(t_dense, *popt)
        pos_days  = float(t_dense[int(np.argmax(y_dense))])

        # SOS/EOS analíticos:
        #   L_rise = 0.25  →  t = m1 - ln(3)/k1
        #   L_fall = 0.25  →  t = m2 + ln(3)/k2
        _ln3    = float(np.log(3))
        sos_days = m1 - _ln3 / k1
        eos_days = m2 + _ln3 / k2

        # Calcula R² nos dados observados
        residuals = ndvi_cycle - double_logistic(days_since_start, *popt)
        ss_res    = float(np.sum(residuals ** 2))
        ss_tot    = float(np.sum((ndvi_cycle - np.mean(ndvi_cycle)) ** 2))
        r_squared = 1.0 - ss_res / (ss_tot + 1e-8)

        if r_squared < quality_threshold:
            return {
                'fit_success': False,
                'reason': f'R² baixo: {r_squared:.3f}',
                'r_squared': r_squared,
                'cycle': cycle,
            }

        # Converte dias → datas absolutas
        t0       = pd.Timestamp(dates_cycle[0])
        sos_date = t0 + timedelta(days=sos_days)
        pos_date = t0 + timedelta(days=pos_days)
        eos_date = t0 + timedelta(days=eos_days)

        sos_ndvi = float(double_logistic(np.array([sos_days]), *popt)[0])
        pos_ndvi = float(double_logistic(np.array([pos_days]), *popt)[0])
        eos_ndvi = float(double_logistic(np.array([eos_days]), *popt)[0])

        return {
            'fit_success': True,
            'cycle_num': cycle['cycle_num'],
            'cycle_start': cycle['start_date'],
            'cycle_end': cycle['end_date'],
            'cycle_length_days': cycle['length_days'],
            'at_series_start': cycle.get('at_series_start', False),
            'at_series_end':   cycle.get('at_series_end',   False),
            'season_type': classify_season_type(pos_date),
            'r_squared': r_squared,
            'rmse': float(np.sqrt(np.mean(residuals ** 2))),
            'curve_params': {
                'amplitude':      float(amplitude),
                'offset':         float(offset),
                'mean_days':      pos_days,       # POS em dias
                'm1':             float(m1),      # inflexão de subida
                'k1':             float(k1),      # taxa de subida (day⁻¹)
                'm2':             float(m2),      # inflexão de descida
                'k2':             float(k2),      # taxa de descida (day⁻¹)
                'std_left_days':  abs(pos_days - m1),
                'std_right_days': abs(m2 - pos_days),
                'std_dev_days':   abs(m2 - m1) / 4.0,
            },
            'phenophase_dates': {'sos': sos_date, 'pos': pos_date, 'eos': eos_date},
            'phenophase_values': {
                'sos_ndvi': sos_ndvi,
                'pos_ndvi': pos_ndvi,
                'eos_ndvi': eos_ndvi,
            },
            'phenophase_days': {
                'sos_days': sos_days,
                'pos_days': pos_days,
                'eos_days': eos_days,
            },
        }

    except Exception as e:
        return {'fit_success': False, 'reason': f'Erro na otimização: {str(e)}', 'cycle': cycle}


fit_gaussian_to_cycle = fit_curve_to_cycle  # alias de compatibilidade


def extract_phenometrics(df_ts: pd.DataFrame, ndvi_column: str = 'NDVI_mean',
                           min_cycle_length_days: int = 45,
                           smoothing_method: str = 'savgol',
                           quality_threshold: float = 0.6,
                           quantile_trough: float = 20) -> Dict[str, Any]:
    """
    Extrai métricas fenológicas completas usando nova metodologia v2.
    Otimizada para detectar múltiplas safras (safra e safrinha).
    AGORA COM MELHOR SUPORTE A SAFRA/SAFRINHA E CICLOS CURTOS.
    
    Args:
        df_ts: DataFrame com série temporal (deve ter 'datetime' e coluna NDVI)
        ndvi_column: Nome da coluna NDVI
        min_cycle_length_days: Comprimento mínimo do ciclo em dias
        smoothing_method: 'savgol', 'median', ou 'both'
        quality_threshold: Threshold mínimo de R² para fit bem-sucedido
        quantile_trough: Percentil para detectar vales (mais baixo = mais sensível)
    
    Returns:
        Dicionário com métricas fenológicas e ciclos ajustados
    """
    # Preparação dos dados
    df_ts = df_ts.copy()
    df_ts['datetime'] = pd.to_datetime(df_ts['datetime'])
    df_ts = df_ts.sort_values('datetime')
    
    ndvi_values = df_ts[ndvi_column].values.astype(float)
    dates = df_ts['datetime'].values
    
    # Valida dados
    if len(ndvi_values) < 10:
        return {
            'success': False,
            'error': 'Série temporal muito curta (< 10 pontos)',
            'cycles': []
        }
    
    # Remove NaNs
    valid_mask = ~np.isnan(ndvi_values)
    ndvi_values = ndvi_values[valid_mask]
    dates = dates[valid_mask]
    
    if len(ndvi_values) < 10:
        return {
            'success': False,
            'error': 'Série temporal muito curta após remover NaNs',
            'cycles': []
        }
    
    # Etapa 1: Suavização adaptativa
    ndvi_smooth = adaptive_smoothing(ndvi_values, dates, method=smoothing_method)

    # Etapa 2: Detecção direta de picos vegetativos (não depende de vales).
    # A distância mínima de DETECÇÃO é mais curta que o ciclo mínimo agronômico
    # para capturar safra+safrinha mesmo quando o vale entre-safra fica elevado
    # (hexágonos mistos onde parte dos pixels ainda tem cultura crescendo enquanto
    # outra parte está em solo exposto).  O filtro por MIN_CYCLE_DAYS é aplicado
    # no pós-fit via extract_phenology.py, não aqui.
    detection_distance_days = max(60, int(min_cycle_length_days * 0.65))
    peaks = detect_vegetation_peaks(
        ndvi_smooth, dates,
        min_distance_days=detection_distance_days,
    )

    # Etapa 3: Define janelas ao redor de cada pico
    cycles = segment_around_peaks(ndvi_values, dates, peaks)

    # Etapa 4: Ajuste de logística dupla em cada janela
    fitted_cycles = []
    for cycle in cycles:
        result = fit_curve_to_cycle(ndvi_values, dates, cycle,
                                    quality_threshold=quality_threshold)
        fitted_cycles.append(result)

    # Extrai estatísticas
    successful_cycles = [c for c in fitted_cycles if c.get('fit_success', False)]
    failed_cycles = [c for c in fitted_cycles if not c.get('fit_success', False)]

    if successful_cycles:
        mean_r_squared = np.mean([c['r_squared'] for c in successful_cycles])
        mean_cycle_length = np.mean([c['cycle_length_days'] for c in successful_cycles])
        mean_rmse = np.mean([c['rmse'] for c in successful_cycles])
    else:
        mean_r_squared = 0.0
        mean_cycle_length = 0.0
        mean_rmse = 0.0

    return {
        'success': True,
        'num_cycles_detected': len(cycles),
        'num_successful_fits': len(successful_cycles),
        'num_failed_fits': len(failed_cycles),
        'mean_r_squared': float(mean_r_squared),
        'mean_rmse': float(mean_rmse),
        'mean_cycle_length_days': float(mean_cycle_length),
        'data_points': len(ndvi_values),
        'total_days': float((dates[-1] - dates[0]) / np.timedelta64(1, 'D')),
        'cycles': fitted_cycles,
        'diagnostics': {
            'ndvi_min': float(np.min(ndvi_values)),
            'ndvi_max': float(np.max(ndvi_values)),
            'ndvi_mean': float(np.mean(ndvi_values)),
            'ndvi_std': float(np.std(ndvi_values)),
            'num_peaks': len(peaks),
            'peak_indices': peaks.tolist(),
            'detection_distance_days': detection_distance_days,
        }
    }


def print_phenometrics_summary(phenometrics: Dict) -> None:
    """
    Imprime um resumo dos resultados das métricas fenológicas v2.
    """
    print("\n" + "="*80)
    print("RESUMO DE MÉTRICAS FENOLÓGICAS (MÉTODO V2)")
    print("="*80)
    
    if not phenometrics.get('success', False):
        print(f"❌ Erro: {phenometrics.get('error', 'Desconhecido')}")
        return
    
    print(f"\n📊 ESTATÍSTICAS GERAIS:")
    print(f"   Ciclos detectados: {phenometrics['num_cycles_detected']}")
    print(f"   Ajustes bem-sucedidos: {phenometrics['num_successful_fits']}")
    print(f"   Ajustes falhados: {phenometrics['num_failed_fits']}")
    print(f"   Pontos de dados: {phenometrics['data_points']}")
    print(f"   Duração total: {phenometrics['total_days']:.1f} dias")
    
    print(f"\n📈 QUALIDADE DO AJUSTE:")
    print(f"   R² médio: {phenometrics['mean_r_squared']:.4f}")
    print(f"   RMSE médio: {phenometrics['mean_rmse']:.4f}")
    print(f"   Comprimento médio do ciclo: {phenometrics['mean_cycle_length_days']:.1f} dias")
    
    print(f"\n🌾 DADOS NDVI:")
    diag = phenometrics['diagnostics']
    print(f"   Mínimo: {diag['ndvi_min']:.4f}")
    print(f"   Máximo: {diag['ndvi_max']:.4f}")
    print(f"   Média: {diag['ndvi_mean']:.4f}")
    print(f"   Desvio padrão: {diag['ndvi_std']:.4f}")
    print(f"   Picos detectados: {diag.get('num_peaks', diag.get('num_troughs', 0))}")
    
    print("\n" + "-"*80)
    print("DETALHES DE CADA CICLO:")
    print("-"*80)
    
    for cycle in phenometrics['cycles']:
        if cycle['fit_success']:
            print(f"\n✅ Ciclo {cycle['cycle_num']}:")
            print(f"   Período: {cycle['cycle_start'].strftime('%Y-%m-%d')} a {cycle['cycle_end'].strftime('%Y-%m-%d')}")
            print(f"   Duração: {cycle['cycle_length_days']:.1f} dias")
            print(f"   R²: {cycle['r_squared']:.4f}")
            print(f"   RMSE: {cycle['rmse']:.4f}")
            print(f"   SOS: {cycle['phenophase_dates']['sos'].strftime('%Y-%m-%d')} (NDVI: {cycle['phenophase_values']['sos_ndvi']:.4f})")
            print(f"   POS: {cycle['phenophase_dates']['pos'].strftime('%Y-%m-%d')} (NDVI: {cycle['phenophase_values']['pos_ndvi']:.4f})")
            print(f"   EOS: {cycle['phenophase_dates']['eos'].strftime('%Y-%m-%d')} (NDVI: {cycle['phenophase_values']['eos_ndvi']:.4f})")
        else:
            print(f"\n❌ Ciclo {cycle['cycle_num']}: Falha no ajuste")
            print(f"   Motivo: {cycle['reason']}")
    
    print("\n" + "="*80 + "\n")


def plot_diagnostic(df_ts: pd.DataFrame, phenometrics: Dict, ndvi_column: str = 'NDVI_mean',
                       title: str = 'Diagnóstico de Detecção de Safras (V2)') -> plt.Figure:
    """
    Plota diagnóstico completo da detecção de ciclos.
    """
    df_ts = df_ts.copy()
    df_ts['datetime'] = pd.to_datetime(df_ts['datetime'])
    df_ts = df_ts.sort_values('datetime')
    
    ndvi_values = df_ts[ndvi_column].values
    dates = df_ts['datetime'].values
    
    # Suavização para visualização
    if len(ndvi_values) > 11:
        ndvi_smooth = savgol_filter(ndvi_values, window_length=11, polyorder=2)
    else:
        ndvi_smooth = ndvi_values
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Gráfico 1: NDVI bruto vs suavizado
    ax = axes[0]
    ax.plot(dates, ndvi_values, 'k-', linewidth=1, alpha=0.5, label='NDVI Original')
    ax.plot(dates, ndvi_smooth, 'b-', linewidth=2, label='NDVI Suavizado')
    
    # Marca vales detectados
    peaks_idx = phenometrics['diagnostics'].get('peak_indices', phenometrics['diagnostics'].get('troughs_indices', []))
    if len(peaks_idx) > 0:
        ax.scatter(dates[peaks_idx], ndvi_values[peaks_idx], color='green', s=100,
                  marker='^', label='Picos Detectados', zorder=5)
    
    ax.set_ylabel('NDVI', fontsize=11)
    ax.set_title(f'{title} - NDVI Original vs Suavizado', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Gráfico 2: Ciclos com cores
    ax = axes[1]
    ax.plot(dates, ndvi_values, 'k-', linewidth=0.8, alpha=0.3, label='NDVI Original')
    
    colors = plt.cm.tab10(np.linspace(0, 1, max(5, len(phenometrics['cycles']))))
    
    for i, cycle in enumerate(phenometrics['cycles']):
        if cycle['fit_success']:
            start_idx = phenometrics['cycles'].index(cycle)  # Posição aproximada
            color = colors[i % len(colors)]
            
            # Extrai dados do ciclo
            cycle_start = cycle['cycle_start']
            cycle_end = cycle['cycle_end']
            
            # Marca período do ciclo
            ax.axvspan(cycle_start, cycle_end, alpha=0.1, color=color)
            
            # Marca pontos fenológicos
            ax.scatter([cycle['phenophase_dates']['sos']], 
                      [cycle['phenophase_values']['sos_ndvi']], 
                      marker='o', s=80, color=color, edgecolors='black', linewidth=1.5)
            ax.scatter([cycle['phenophase_dates']['pos']], 
                      [cycle['phenophase_values']['pos_ndvi']], 
                      marker='*', s=300, color=color, edgecolors='black', linewidth=1.5)
            ax.scatter([cycle['phenophase_dates']['eos']], 
                      [cycle['phenophase_values']['eos_ndvi']], 
                      marker='s', s=80, color=color, edgecolors='black', linewidth=1.5)
    
    ax.set_ylabel('NDVI', fontsize=11)
    ax.set_title('Ciclos Detectados com Pontos Fenológicos', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(['NDVI Original'] + [f'Ciclo {c["cycle_num"]}' for c in phenometrics['cycles'] 
                                   if c['fit_success']], loc='upper right')
    
    # Gráfico 3: Ajuste de logística dupla
    ax = axes[2]
    ax.plot(dates, ndvi_values, 'k-', linewidth=1.5, label='NDVI Original', zorder=1)

    for i, cycle in enumerate(phenometrics['cycles']):
        if cycle['fit_success']:
            color = colors[i % len(colors)]

            cycle_start_date = cycle['cycle_start']
            cycle_end_date = cycle['cycle_end']

            cycle_dates_mask = (dates >= cycle_start_date) & (dates <= cycle_end_date)
            cycle_dates = dates[cycle_dates_mask]

            if len(cycle_dates) > 0:
                days_since_start = np.array([(d - cycle_dates[0]) / np.timedelta64(1, 'D')
                                            for d in cycle_dates], dtype=float)

                params = cycle['curve_params']
                curve_vals = double_logistic(
                    days_since_start,
                    params['amplitude'],
                    params['m1'],
                    params['k1'],
                    params['m2'],
                    params['k2'],
                    params['offset'])

                ax.plot(cycle_dates, curve_vals, '--', linewidth=2.5,
                       color=color, label=f'Ciclo {cycle["cycle_num"]} (R²={cycle["r_squared"]:.3f})')

    ax.set_xlabel('Data', fontsize=11)
    ax.set_ylabel('NDVI', fontsize=11)
    ax.set_title('Ajustes Logísticos por Ciclo', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    return fig
