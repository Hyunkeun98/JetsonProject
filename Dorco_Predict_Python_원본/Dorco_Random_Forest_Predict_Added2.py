# Random Forest > XGBoost 로 변경
#
import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
import joblib
import time 

# 설정 상수
ANGLE_MIN = 107.0
ANGLE_MAX = 113.0
ANGLE_DIFF_MAX = 2.5

X_MIN = 0.82
X_MAX = 0.92
X_DIFF_MAX = 0.1

ERROR_THRESHOLD = 4    # 에러 판단 기준 (불량 4번 이상)
PREDICTION_WINDOW = 60 # 예측 단위 시간 (향후 1시간 = 60분)

# ==============================================================================
# [보조 함수] 기울기(속도) 계산 함수
# ==============================================================================
def calculate_trend_slope(buffer):
    if len(buffer) < 2:
        return 0.0
    x = np.arange(len(buffer))
    y = np.array(buffer)
    # y값이 모두 동일할 경우 경사도는 0
    if np.all(y == y[0]):
        return 0.0
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)

# ==============================================================================
# [1단계] 데이터 로드 및 전수 양불 판정 (0/1)
# ==============================================================================
def load_and_label_dataset(file_pattern):
    file_list = []
    
    if isinstance(file_pattern, list):
        for pattern in file_pattern:
            file_list.extend(glob.glob(pattern))
    else:
        file_list = glob.glob(file_pattern)
        
    file_list = [f for f in set(file_list) if os.path.isfile(f)]
    file_list = sorted(file_list)

    if not file_list:
        raise FileNotFoundError(f"학습에 사용할 CSV 파일이 없습니다. 입력 경로: {file_pattern}")
        
    df_list = []
    for f in file_list:
        try:
            temp_df = pd.read_csv(f, engine='c', low_memory=False)
            df_list.append(temp_df)
        except PermissionError:
            print(f">> [경고] {f} 파일이 다른 프로그램에 의해 열려 있어 건너뜁니다.")
            continue
            
    if not df_list:
        raise ValueError("로드된 데이터프레임이 비어 있습니다.")
        
    df = pd.concat(df_list, ignore_index=True)
    
    df['DATETIME'] = pd.to_datetime(
        df['INSP_DATE'].astype(str) + df['INSP_TIME'].astype(str).str.zfill(6), 
        format='%Y%m%d%H%M%S', errors='coerce'
    )
    df = df.dropna(subset=['DATETIME']).sort_values(by='DATETIME').reset_index(drop=True)
    
    target_cols = [c for c in df.columns if c.startswith(('A_L_', 'A_R_', 'X_L_', 'X_R_'))]
    df[target_cols] = df[target_cols] / 1000.0

    is_ng = np.zeros(len(df), dtype=int)
    for pos in range(1, 13):
        suffix = f"{pos:02d}"
        cond_ang = (df[f'A_L_{suffix}'] >= ANGLE_MIN) & (df[f'A_L_{suffix}'] <= ANGLE_MAX) & \
                   (df[f'A_R_{suffix}'] >= ANGLE_MIN) & (df[f'A_R_{suffix}'] <= ANGLE_MAX) & \
                   (np.abs(df[f'A_L_{suffix}'] - df[f'A_R_{suffix}']) <= ANGLE_DIFF_MAX)
                   
        cond_x = (df[f'X_L_{suffix}'] >= X_MIN) & (df[f'X_L_{suffix}'] <= X_MAX) & \
                 (df[f'X_R_{suffix}'] >= X_MIN) & (df[f'X_R_{suffix}'] <= X_MAX) & \
                 (np.abs(df[f'X_L_{suffix}'] - df[f'X_R_{suffix}']) <= X_DIFF_MAX)
                
        is_ng |= ~(cond_ang & cond_x)
        
    df['IS_NG'] = is_ng.astype(int)
    return df

# ==============================================================================
# [2단계] AI 모델 데이터셋 구성 (피처 정밀화 버전)
# ==============================================================================
def create_ml_features_and_targets(df):
    df_time = df.set_index('DATETIME')
    
    # 1. 1분 단위로 불량 개수 1차 집계
    df_min = df_time['IS_NG'].resample('1min').sum().to_frame()
    
    # 2. 피처 생성
    # [시간 주기성 피처 추가] (설비 작업 시간 조 영향 파악)
    df_min['Hour'] = df_min.index.hour
    df_min['X_hour_sin'] = np.sin(2 * np.pi * df_min['Hour'] / 24.0)
    df_min['X_hour_cos'] = np.cos(2 * np.pi * df_min['Hour'] / 24.0)
    
    # [직전 시점 시차(Lag) 피처 추가] -> 최신 트렌드를 직접 입력
    for lag in [1, 2, 3, 5]:
        df_min[f'X_ng_lag_{lag}'] = df_min['IS_NG'].shift(lag).fillna(0)
        
    # [연속 불량 지속 시간 추가]
    # 연속으로 불량이 발생하고 있는 상태(분 단위) 누적 계산
    is_active = (df_min['IS_NG'] > 0).astype(int)
    df_min['X_consecutive_ng_mins'] = is_active.groupby((is_active != is_active.shift()).cumsum()).cumsum()
    
    # [다중 윈도우 통계량] (1m std 제거 - 상수값 무의미)
    df_min['X_ng_sum_1m'] = df_min['IS_NG'].copy()
    df_min['X_ng_mean_1m'] = df_min['IS_NG'].copy()
    df_min['X_ng_max_1m'] = df_min['IS_NG'].copy()
    
    for w in [5, 15, 30]:
        df_min[f'X_ng_sum_{w}m'] = df_min['IS_NG'].rolling(window=w, min_periods=1).sum()
        df_min[f'X_ng_mean_{w}m'] = df_min['IS_NG'].rolling(window=w, min_periods=1).mean()
        df_min[f'X_ng_std_{w}m'] = df_min['IS_NG'].rolling(window=w, min_periods=1).std().fillna(0)
        df_min[f'X_ng_max_{w}m'] = df_min['IS_NG'].rolling(window=w, min_periods=1).max()
        # 변동계수(CV): 불량 변동 안정성 추적
        df_min[f'X_ng_cv_{w}m'] = df_min[f'X_ng_std_{w}m'] / (df_min[f'X_ng_mean_{w}m'] + 1e-5)
    
    # [상세 다각적 기울기]
    df_min['X_velocity_30m'] = df_min['IS_NG'].rolling(window=30, min_periods=2).apply(calculate_trend_slope, raw=True).fillna(0)
    df_min['X_velocity_15m'] = df_min['IS_NG'].rolling(window=15, min_periods=2).apply(calculate_trend_slope, raw=True).fillna(0)
    df_min['X_velocity_5m'] = df_min['IS_NG'].rolling(window=5, min_periods=2).apply(calculate_trend_slope, raw=True).fillna(0)
    df_min['X_velocity_1m'] = df_min['IS_NG'].rolling(window=2, min_periods=2).apply(calculate_trend_slope, raw=True).fillna(0)
    
    # [불량 가속도]
    df_min['X_acceleration_1_5'] = df_min['X_velocity_1m'] - df_min['X_velocity_5m']
    df_min['X_acceleration_5_15'] = df_min['X_velocity_5m'] - df_min['X_velocity_15m']
    df_min['X_acceleration_15_30'] = df_min['X_velocity_15m'] - df_min['X_velocity_30m']
    
    # [상대적 밀도 비율 피처]
    df_min['X_ng_ratio_1_30'] = (df_min['X_ng_sum_1m'] / (df_min['X_ng_sum_30m'] + 1e-5))
    df_min['X_ng_ratio_5_15'] = (df_min['X_ng_sum_5m'] / (df_min['X_ng_sum_15m'] + 1e-5))
    
    # [EWMA 필터 고도화]
    ewma_1 = df_min['IS_NG'].ewm(span=2, adjust=False).mean()
    ewma_5 = df_min['IS_NG'].ewm(span=5, adjust=False).mean()
    ewma_15 = df_min['IS_NG'].ewm(span=15, adjust=False).mean()
    ewma_30 = df_min['IS_NG'].ewm(span=30, adjust=False).mean()
    
    df_min['X_ewma_diff_1_5'] = ewma_1 - ewma_5
    df_min['X_ewma_diff_5_15'] = ewma_5 - ewma_15
    df_min['X_ewma_diff_15_30'] = ewma_15 - ewma_30
    
    # 3. 정답 라벨(Y) 생성
    future_ng_sum = df_min['IS_NG'].iloc[::-1].rolling(window=PREDICTION_WINDOW, min_periods=1).sum().iloc[::-1]
    df_min['Y_TARGET'] = (future_ng_sum >= ERROR_THRESHOLD).astype(int)
    
    # 고도화된 최종 학습 피처 목록 정리
    feature_cols = [
        'X_hour_sin', 'X_hour_cos',
        'X_ng_lag_1', 'X_ng_lag_2', 'X_ng_lag_3', 'X_ng_lag_5',
        'X_consecutive_ng_mins',
        'X_ng_sum_1m', 'X_ng_mean_1m', 'X_ng_max_1m',
        'X_ng_sum_5m', 'X_ng_mean_5m', 'X_ng_std_5m', 'X_ng_max_5m', 'X_ng_cv_5m',
        'X_ng_sum_15m', 'X_ng_mean_15m', 'X_ng_std_15m', 'X_ng_max_15m', 'X_ng_cv_15m',
        'X_ng_sum_30m', 'X_ng_mean_30m', 'X_ng_std_30m', 'X_ng_max_30m', 'X_ng_cv_30m',
        'X_velocity_30m', 'X_velocity_15m', 'X_velocity_5m', 'X_velocity_1m',
        'X_acceleration_1_5', 'X_acceleration_5_15', 'X_acceleration_15_30',
        'X_ng_ratio_1_30', 'X_ng_ratio_5_15',
        'X_ewma_diff_1_5', 'X_ewma_diff_5_15', 'X_ewma_diff_15_30'
    ]
    
    df_ml = df_min.dropna()
    X = df_ml[feature_cols]
    y = df_ml['Y_TARGET']
    
    return X, y, feature_cols

# ==============================================================================
# [가상 폴백] 고밀도 전조 특징 대응 모형 생성기 (피처 일치화)
# ==============================================================================
def make_mock_ml_dataset():
    np.random.seed(42)
    n_samples = 2000
    
    X_list = []
    y_list = []
    
    for idx in range(n_samples):
        hour = (idx % 1440) // 60
        hr_sin = np.sin(2 * np.pi * hour / 24.0)
        hr_cos = np.cos(2 * np.pi * hour / 24.0)
        
        # 1. 정상 상태 (가끔 간헐적 불량)
        if np.random.rand() > 0.15:
            ng_1 = float(np.random.randint(0, 1))
            ng_5 = float(ng_1 + np.random.randint(0, 2))
            ng_15 = float(ng_5 + np.random.randint(0, 2))
            ng_30 = float(ng_15 + np.random.randint(0, 2))
            
            vel_30 = float(np.random.uniform(-0.01, 0.01))
            vel_15 = float(np.random.uniform(-0.02, 0.02))
            vel_5 = float(np.random.uniform(-0.03, 0.03))
            vel_1 = float(np.random.uniform(-0.05, 0.05))
            
            consec = float(np.random.randint(0, 2))
            target = 0
        # 2. 전조 증상 고밀도 폭발 상태
        else:
            ng_1 = float(np.random.randint(1, 4))
            ng_5 = float(ng_1 + np.random.randint(2, 6))
            ng_15 = float(ng_5 + np.random.randint(3, 8))
            ng_30 = float(ng_15 + np.random.randint(4, 10))
            
            vel_30 = float(np.random.uniform(0.05, 0.15))
            vel_15 = float(np.random.uniform(0.12, 0.35))
            vel_5 = float(np.random.uniform(0.30, 0.80))
            vel_1 = float(np.random.uniform(0.50, 1.50))
            
            consec = float(np.random.randint(3, 10))
            target = 1
            
        acc_1_5 = vel_1 - vel_5
        acc_5_15 = vel_5 - vel_15
        acc_15_30 = vel_15 - vel_30
        
        ratio_1_30 = ng_1 / (ng_30 + 1e-5)
        ratio_5_15 = ng_5 / (ng_15 + 1e-5)
        
        ewma_1_5 = vel_1 * 0.9
        ewma_5_15 = vel_5 * 0.7
        ewma_15_30 = vel_15 * 0.5
        
        # Lag Mocking
        l1, l2, l3, l5 = ng_1, max(0.0, ng_1-1), max(0.0, ng_1-2), max(0.0, ng_1-3)
        
        X_list.append([
            hr_sin, hr_cos,
            l1, l2, l3, l5,
            consec,
            ng_1, ng_1, ng_1,
            ng_5, ng_5/5.0, np.sqrt(ng_5)*0.2, ng_5, 0.2,
            ng_15, ng_15/15.0, np.sqrt(ng_15)*0.3, ng_15, 0.3,
            ng_30, ng_30/30.0, np.sqrt(ng_30)*0.4, ng_30, 0.4,
            vel_30, vel_15, vel_5, vel_1,
            acc_1_5, acc_5_15, acc_15_30,
            ratio_1_30, ratio_5_15,
            ewma_1_5, ewma_5_15, ewma_15_30
        ])
        y_list.append(target)
        
    feature_cols = [
        'X_hour_sin', 'X_hour_cos',
        'X_ng_lag_1', 'X_ng_lag_2', 'X_ng_lag_3', 'X_ng_lag_5',
        'X_consecutive_ng_mins',
        'X_ng_sum_1m', 'X_ng_mean_1m', 'X_ng_max_1m',
        'X_ng_sum_5m', 'X_ng_mean_5m', 'X_ng_std_5m', 'X_ng_max_5m', 'X_ng_cv_5m',
        'X_ng_sum_15m', 'X_ng_mean_15m', 'X_ng_std_15m', 'X_ng_max_15m', 'X_ng_cv_15m',
        'X_ng_sum_30m', 'X_ng_mean_30m', 'X_ng_std_30m', 'X_ng_max_30m', 'X_ng_cv_30m',
        'X_velocity_30m', 'X_velocity_15m', 'X_velocity_5m', 'X_velocity_1m',
        'X_acceleration_1_5', 'X_acceleration_5_15', 'X_acceleration_15_30',
        'X_ng_ratio_1_30', 'X_ng_ratio_5_15',
        'X_ewma_diff_1_5', 'X_ewma_diff_5_15', 'X_ewma_diff_15_30'
    ]
    X = pd.DataFrame(X_list, columns=feature_cols)
    y = pd.Series(y_list)
    return X, y, feature_cols

# ==============================================================================
# [3단계] AI 모델 학습 (XGBoost Pipeline)
# ==============================================================================
def train_predictive_model():
    print(">> 1. 데이터셋 로드 및 전처리 시작...")
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    target_pattern = os.path.join(current_dir, "DATA", "44호기_*월.csv")
    file_path = sorted(glob.glob(target_pattern))
    
    if len(file_path) > 0:
        df_raw = load_and_label_dataset(file_path) 
        print(">> 2. 시계열 기반 머신러닝 피처/타겟 스펙 생성 중...")
        X, y, feature_cols = create_ml_features_and_targets(df_raw)
    else:
        print(">> [안내] 실 데이터 파일이 없어 1/5/15/30분 고밀도 전조 특징 학습용 세트를 자동 생성합니다...")
        X, y, feature_cols = make_mock_ml_dataset()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print(f">> 3. XGBoost AI 모델 학습 시작 (데이터 크기: {X_train.shape[0]}행)...")
    
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    scale_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    model = XGBClassifier(
        n_estimators=600, # 트리 수 소폭 확장
        max_depth=5,      # 과적합 방지를 위해 깊이 5로 하향
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    if len(np.unique(y_test)) < 2:
        auc_score = 0.5
    else:
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
    print(f">> 4. 모델 검증 완료. 예측 신뢰도(XGBoost AUC Score): {auc_score:.4f}")
    
    custom_threshold = 0.35 
    y_pred_class = (y_pred_proba >= custom_threshold).astype(int)
    
    print(f"[상세 검증 지표 - 예지보전 임계값: {custom_threshold}]")
    print(classification_report(y_test, y_pred_class, target_names=['정상', '위험(에러4회이상)'], zero_division=0))
    
    model_dir = os.path.join(current_dir, 'Model')
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'pm_rf_model.pkl') 
    features_path = os.path.join(model_dir, 'rf_model_features.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(feature_cols, features_path)
    print(f">> 5. 고성능 XGBoost 예지예측 모델 저장 완료 -> 저장경로: {model_dir}")

# ==============================================================================
# [4단계] 실시간 데이터 기반 확률 추론 (Inference Engine)
# ==============================================================================
class RealTimeInferenceEngine:
    def __init__(self, model_path, features_path):
        self.model = None
        self.feature_cols = None
        self.recent_minutes_buffer = []
        
        # 연속 상태 추적기
        self.consecutive_active_minutes = 0
        
        # EWMA 상태 관리 변수
        self.ewma_1 = 0.0
        self.ewma_5 = 0.0
        self.ewma_15 = 0.0
        self.ewma_30 = 0.0
        self.is_first_ewma = True
        
        self.load_model_artifacts(model_path, features_path)

    def load_model_artifacts(self, model_path, features_path):
        if os.path.exists(model_path) and os.path.exists(features_path):
            try:
                self.model = joblib.load(model_path)
                self.feature_cols = joblib.load(features_path)
                print(">> [엔진] 고밀도 XGBoost 예지 보전 모델 적재 완료.")
            except Exception as e:
                print(f">> [엔진] 파일 로드 중 오류 발생: {e}")
                self.model = None
                self.feature_cols = None
        else:
            print(">> [엔진] 모델 아티팩트 미발견. 재생성이 필요합니다.")

    def inject_minute_data_and_predict(self, raw_index_stream, current_time_obj=None):
        if self.model is None or self.feature_cols is None:
            return 0.0
            
        current_minute_ng_count = sum(raw_index_stream)
        
        # 시간 변수 추출 (주기성 Cos/Sin)
        if current_time_obj is None:
            # 기본값으로 현재 시각 정보 사용
            current_time_obj = pd.Timestamp.now()
        hour = current_time_obj.hour
        hr_sin = np.sin(2 * np.pi * hour / 24.0)
        hr_cos = np.cos(2 * np.pi * hour / 24.0)
        
        # 연속 불량 분 카운팅
        if current_minute_ng_count > 0:
            self.consecutive_active_minutes += 1
        else:
            self.consecutive_active_minutes = 0
            
        self.recent_minutes_buffer.append(current_minute_ng_count)
        if len(self.recent_minutes_buffer) > 30:
            self.recent_minutes_buffer.pop(0)
            
        buf_arr = np.array(self.recent_minutes_buffer)
        
        # Lag 피처 구현 (인스턴스에서 직접 추적)
        l1 = self.recent_minutes_buffer[-2] if len(self.recent_minutes_buffer) >= 2 else 0.0
        l2 = self.recent_minutes_buffer[-3] if len(self.recent_minutes_buffer) >= 3 else 0.0
        l3 = self.recent_minutes_buffer[-4] if len(self.recent_minutes_buffer) >= 4 else 0.0
        l5 = self.recent_minutes_buffer[-6] if len(self.recent_minutes_buffer) >= 6 else 0.0
        
        # 다중 윈도우 통계값 생성 함수
        def get_window_stats(window_size):
            sub_buf = buf_arr[-window_size:] if len(buf_arr) >= window_size else buf_arr
            s = float(sub_buf.sum())
            m = float(sub_buf.mean())
            sd = float(sub_buf.std()) if len(sub_buf) > 1 else 0.0
            mx = float(sub_buf.max())
            cv = sd / (m + 1e-5)
            return s, m, sd, mx, cv

        s1, m1, _, mx1, _ = get_window_stats(1)
        s5, m5, sd5, mx5, cv5 = get_window_stats(5)
        s15, m15, sd15, mx15, cv15 = get_window_stats(15)
        s30, m30, sd30, mx30, cv30 = get_window_stats(30)
        
        # 전조증상 고도화 수치: 기울기
        x_velocity_1m = float(calculate_trend_slope(self.recent_minutes_buffer[-2:]))
        x_velocity_5m = float(calculate_trend_slope(self.recent_minutes_buffer[-5:]))
        x_velocity_15m = float(calculate_trend_slope(self.recent_minutes_buffer[-15:]))
        x_velocity_30m = float(calculate_trend_slope(self.recent_minutes_buffer))
        
        # 불량 가속도
        x_acceleration_1_5 = x_velocity_1m - x_velocity_5m
        x_acceleration_5_15 = x_velocity_5m - x_velocity_15m
        x_acceleration_15_30 = x_velocity_15m - x_velocity_30m
        
        # 상대적 비율
        x_ng_ratio_1_30 = s1 / (s30 + 1e-5)
        x_ng_ratio_5_15 = s5 / (s15 + 1e-5)
        
        # 다중 EWMA 필터 차이 계산
        if self.is_first_ewma:
            self.ewma_1 = float(current_minute_ng_count)
            self.ewma_5 = float(current_minute_ng_count)
            self.ewma_15 = float(current_minute_ng_count)
            self.ewma_30 = float(current_minute_ng_count)
            self.is_first_ewma = False
        else:
            alpha_1 = 2.0 / (2.0 + 1.0)
            alpha_5 = 2.0 / (5.0 + 1.0)
            alpha_15 = 2.0 / (15.0 + 1.0)
            alpha_30 = 2.0 / (30.0 + 1.0)
            
            self.ewma_1 = alpha_1 * current_minute_ng_count + (1.0 - alpha_1) * self.ewma_1
            self.ewma_5 = alpha_5 * current_minute_ng_count + (1.0 - alpha_5) * self.ewma_5
            self.ewma_15 = alpha_15 * current_minute_ng_count + (1.0 - alpha_15) * self.ewma_15
            self.ewma_30 = alpha_30 * current_minute_ng_count + (1.0 - alpha_30) * self.ewma_30
            
        x_ewma_diff_1_5 = self.ewma_1 - self.ewma_5
        x_ewma_diff_5_15 = self.ewma_5 - self.ewma_15
        x_ewma_diff_15_30 = self.ewma_15 - self.ewma_30
        
        # 학습된 모델과 동일한 정렬 구성
        input_data = pd.DataFrame([[
            hr_sin, hr_cos,
            l1, l2, l3, l5,
            float(self.consecutive_active_minutes),
            s1, m1, mx1,
            s5, m5, sd5, mx5, cv5,
            s15, m15, sd15, mx15, cv15,
            s30, m30, sd30, mx30, cv30,
            x_velocity_30m, x_velocity_15m, x_velocity_5m, x_velocity_1m,
            x_acceleration_1_5, x_acceleration_5_15, x_acceleration_15_30,
            x_ng_ratio_1_30, x_ng_ratio_5_15,
            x_ewma_diff_1_5, x_ewma_diff_5_15, x_ewma_diff_15_30
        ]], columns=self.feature_cols)
        
        probabilities = self.model.predict_proba(input_data)
        risk_probability = float(probabilities[0][1])
        
        return risk_probability

# ==============================================================================
# [5단계] 시뮬레이션 테스트 가동
# ==============================================================================
if __name__ == "__main__":
    main_start_time = time.perf_counter()    
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
        abs_model_path = os.path.join(current_dir, 'Model', 'pm_rf_model.pkl')
        abs_features_path = os.path.join(current_dir, 'Model', 'rf_model_features.pkl')

        print(">> [시스템] 기존 구버전 잔재 파일을 정밀 삭제합니다...")
        for file_path in [abs_model_path, abs_features_path]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"파일 삭제 오류: {e}")
        
        print(">> [시스템] 실시간 고밀도 추론 엔진 초기화...")
        inference_engine = RealTimeInferenceEngine(
            model_path=abs_model_path,
            features_path=abs_features_path
        )
     
        if inference_engine.model is None or inference_engine.feature_cols is None:
            print(">> [시스템 안내] 신규 고도화 피처가 적용된 학습 세트 가동 중...")
            train_predictive_model()
            inference_engine.load_model_artifacts(abs_model_path, abs_features_path)

        # 테스트용 임의 시간 정의 (주기성 확인 목적)
        test_time = pd.Timestamp('2026-07-16 14:00:00')

        # 테스트 실시간 스트림 데이터 (1분당 불량 개수 추론)
        time_series_stream = [
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 1분 (1개)
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 2분 (3개)          
            [0, 0, 0, 3, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 3분 (7개)          
            [0, 2, 0, 3, 0, 0, 1, 0, 0, 2, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 4분 (12개)
        ]
        
        print(">> 6. 실시간 현장 데이터 유입에 따른 예지 확률 추론 결과:")
        
        inference_start_time = time.perf_counter()
        
        for t_min, raw_index_stream in enumerate(time_series_stream, start=1):
            sim_time = test_time + pd.Timedelta(minutes=t_min)
            current_minute_ng_count = sum(raw_index_stream) 
            prob = inference_engine.inject_minute_data_and_predict(raw_index_stream, current_time_obj=sim_time)
            
            temp_buffer = inference_engine.recent_minutes_buffer
            current_slope_30m = calculate_trend_slope(temp_buffer)
            current_slope_1m = calculate_trend_slope(temp_buffer[-2:]) # 1분 전조
            
            print(f"[{t_min}분 경과 | 시각: {sim_time.strftime('%H:%M')}] 분당 불량 발생 수: {current_minute_ng_count}ea")
            print(f"[전조 분석] 연속 불량 지속: {inference_engine.consecutive_active_minutes}분 | 30분 기울기: {current_slope_30m:+.3f} | 1분 기울기: {current_slope_1m:+.3f}")
            print(f"[AI 예지분석 결과] 향후 1시간 내 에러 발생 위험도: {prob*100:.2f}%")
            
            if prob >= 0.35:        # Custom_Threshold
                print(f"[현장 제어 알람] 초단기/장기 전조 임계 돌파 ({prob*100:.1f}%) -> 조기 보전 권고")
            
        inference_end_time = time.perf_counter()       

        print("=" * 50)
        print(f">> [성능 리포트] 고도화 피처가 반영된 실시간 연산 소요 시간: {inference_end_time - inference_start_time:.6f}초") 
        
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        main_end_time = time.perf_counter()
        print(f">> [시스템] 전체 실행 시간: {main_end_time - main_start_time:.4f}초")
