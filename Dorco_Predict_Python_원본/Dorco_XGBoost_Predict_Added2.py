import pandas as pd
import numpy as np
import glob
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib

# XGBoost 기반 예측 엔진 탑재
try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError("XGBoost 라이브러리가 필요합니다. 'pip install xgboost'를 먼저 실행해 주세요.")

# ==============================================================================
# [설정 및 판단 기준 상수]
# ==============================================================================
ANGLE_MIN, ANGLE_MAX, ANGLE_DIFF_MAX = 107.0, 113.0, 2.5
X_MIN, X_MAX, X_DIFF_MAX = 0.82, 0.92, 0.1

ERROR_THRESHOLD = 4    # 에러 판단 기준 (향후 1시간 내 불량 4번 이상 발생 시 위험)
PREDICTION_WINDOW = 60 # 예측 단위 시간 (향후 1시간 = 60분)

# ==============================================================================
# [수학적 보조 함수] 선형 추세 기울기(속도) 연산
# ==============================================================================
def calculate_trend_slope(buffer_list):
    """최근 유입 데이터 스트림의 선형 추세 기울기(Velocity)를 계산합니다."""
    n = len(buffer_list)
    if n < 2:
        return 0.0
    x = np.arange(n)
    y = np.array(buffer_list)
    
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    return numerator / denominator if denominator != 0 else 0.0

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
        
    file_list = sorted(list(set(file_list)))

    if not file_list:
        raise FileNotFoundError(f"학습에 사용할 CSV 파일이 없습니다. 입력 경로: {file_pattern}")
        
    df_list = []
    for f in file_list:
        try:
            temp_df = pd.read_csv(f, engine='c', low_memory=False)
            df_list.append(temp_df)
        except PermissionError:
            print(f">> [경고] 파일이 열려 있어 로드할 수 없습니다: {os.path.basename(f)}")
            continue
            
    df = pd.concat(df_list, ignore_index=True)
    
    # DATETIME 생성 및 정렬
    df['DATETIME'] = pd.to_datetime(
        df['INSP_DATE'].astype(str) + df['INSP_TIME'].astype(str).str.zfill(6), 
        format='%Y%m%d%H%M%S', errors='coerce'
    )
    df = df.dropna(subset=['DATETIME']).sort_values(by='DATETIME').reset_index(drop=True)
    
    # mm 단위를 m 단위로 스케일 변환
    target_cols = [c for c in df.columns if c.startswith(('A_L_', 'A_R_', 'X_L_', 'X_R_'))]
    df[target_cols] = df[target_cols] / 1000.0

    # 12포인트 종합 양불 판정 실행
    is_ng = np.zeros(len(df), dtype=int)
    for pos in range(1, 13):
        suffix = f"{pos:02d}"
        cond_ang = (df[f'A_L_{suffix}'] >= ANGLE_MIN) & (df[f'A_L_{suffix}'] <= ANGLE_MAX) & \
                   (df[f'A_R_{suffix}'] >= ANGLE_MIN) & (df[f'A_R_{suffix}'] <= ANGLE_MAX) & \
                   (np.abs(df[f'A_L_{suffix}'] - df[f'A_R_{suffix}']) <= ANGLE_DIFF_MAX)
                   
        cond_x = (df[f'X_L_{suffix}'] >= X_MIN) & (df[f'X_L_{suffix}'] <= X_MAX) & \
                 (df[f'X_R_{suffix}'] >= X_MIN) & (df[f'X_R_{suffix}'] <= X_MAX) & \
                 (np.abs(df[f'X_L_{suffix}'] - df[f'X_R_{suffix}']) <= X_DIFF_MAX)
        
        is_ng = np.where(~(cond_ang & cond_x), 1, is_ng)
        
    df['IS_NG'] = is_ng
    return df

# ==============================================================================
# [2단계] AI 모델 데이터셋 구성 (15개 고정밀 전조 증상 피처화)
# ==============================================================================
def create_ml_features_and_targets(df):
    df_time = df.set_index('DATETIME')
    df_min = df_time['IS_NG'].resample('1min').sum().to_frame()
    
    # --- [통계량 피처 고도화] ---
    df_min['X_ng_sum_30m'] = df_min['IS_NG'].rolling(window=30, min_periods=1).sum()
    df_min['X_ng_sum_10m'] = df_min['IS_NG'].rolling(window=10, min_periods=1).sum()
    df_min['X_ng_mean_30m'] = df_min['IS_NG'].rolling(window=30, min_periods=1).mean()
    df_min['X_ng_std_30m'] = df_min['IS_NG'].rolling(window=30, min_periods=1).std().fillna(0)
    df_min['X_ng_max_30m'] = df_min['IS_NG'].rolling(window=30, min_periods=1).max()
    
    # --- [다중 윈도우 추세 기울기 (속도)] ---
    df_min['X_vel_30m'] = df_min['IS_NG'].rolling(window=30, min_periods=2).apply(calculate_trend_slope, raw=True).fillna(0)
    df_min['X_vel_10m'] = df_min['IS_NG'].rolling(window=10, min_periods=2).apply(calculate_trend_slope, raw=True).fillna(0)
    df_min['X_vel_5m'] = df_min['IS_NG'].rolling(window=5, min_periods=2).apply(calculate_trend_slope, raw=True).fillna(0)
    
    # --- [불량 가속도 (Acceleration)] ---
    df_min['X_accel_5_30'] = df_min['X_vel_5m'] - df_min['X_vel_30m']
    
    # --- [단기 변동성 상한선 채널 돌파 강도 (Bollinger Band 응용)] ---
    df_min['X_vol_upper_band'] = df_min['X_ng_mean_30m'] + (2.0 * df_min['X_ng_std_30m'])
    df_min['X_band_break_intensity'] = (df_min['IS_NG'] - df_min['X_vol_upper_band']).clip(lower=0)

    # --- [순간 모멘텀 지수] ---
    df_min['X_momentum_3m'] = df_min['IS_NG'].diff(3).fillna(0)
    
    # --- [MACD 관점의 누적 불량 에너지 수렴/확산 분석] ---
    ewma_5 = df_min['IS_NG'].ewm(span=5, adjust=False).mean()
    ewma_30 = df_min['IS_NG'].ewm(span=30, adjust=False).mean()
    df_min['X_macd'] = ewma_5 - ewma_30
    df_min['X_macd_signal'] = df_min['X_macd'].ewm(span=5, adjust=False).mean()
    df_min['X_macd_hist'] = df_min['X_macd'] - df_min['X_macd_signal']
    
    # --- [정답 데이터셋 라벨링 (향후 1시간 내 누적 불량 4회 이상 발생 여부)] ---
    future_ng_sum = df_min['IS_NG'].iloc[::-1].rolling(window=PREDICTION_WINDOW, min_periods=1).sum().iloc[::-1]
    df_min['Y_TARGET'] = (future_ng_sum >= ERROR_THRESHOLD).astype(int)
    
    feature_cols = [
        'X_ng_sum_30m', 'X_ng_sum_10m', 'X_ng_mean_30m', 'X_ng_std_30m', 'X_ng_max_30m',
        'X_vel_30m', 'X_vel_10m', 'X_vel_5m', 'X_accel_5_30',
        'X_vol_upper_band', 'X_band_break_intensity', 'X_momentum_3m',
        'X_macd', 'X_macd_signal', 'X_macd_hist'
    ]
    
    df_ml = df_min.dropna()
    return df_ml[feature_cols], df_ml['Y_TARGET'], feature_cols

# ==============================================================================
# [가상 데이터 생성] 실 데이터 부재 시 백업 훈련 데이터셋 구성 엔진
# ==============================================================================
def make_synthetic_dataset():
    np.random.seed(42)
    n_samples = 3000
    X_list, y_list = [], []
    
    for _ in range(n_samples):
        is_hazard = np.random.rand() < 0.15
        if not is_hazard:
            # 정상 수치 시뮬레이션
            row = [
                np.random.randint(0, 3), np.random.randint(0, 2), np.random.uniform(0.01, 0.1), np.random.uniform(0.0, 0.2), np.random.randint(0, 1),
                np.random.uniform(-0.01, 0.01), np.random.uniform(-0.02, 0.02), np.random.uniform(-0.03, 0.03), np.random.uniform(-0.02, 0.02),
                np.random.uniform(0.1, 0.5), 0.0, np.random.randint(-1, 2),
                np.random.uniform(-0.05, 0.05), np.random.uniform(-0.03, 0.03), np.random.uniform(-0.02, 0.02)
            ]
            target = 0
        else:
            # 급격한 에러 전조 징후 시뮬레이션
            row = [
                np.random.randint(5, 18), np.random.randint(3, 10), np.random.uniform(0.2, 0.6), np.random.uniform(0.4, 1.2), np.random.randint(2, 5),
                np.random.uniform(0.05, 0.2), np.random.uniform(0.1, 0.4), np.random.uniform(0.2, 0.7), np.random.uniform(0.1, 0.5),
                np.random.uniform(0.5, 1.5), np.random.uniform(0.2, 2.0), np.random.randint(1, 4),
                np.random.uniform(0.3, 1.2), np.random.uniform(0.1, 0.8), np.random.uniform(0.1, 0.6)
            ]
            target = 1
        X_list.append(row)
        y_list.append(target)
        
    feature_cols = [
        'X_ng_sum_30m', 'X_ng_sum_10m', 'X_ng_mean_30m', 'X_ng_std_30m', 'X_ng_max_30m',
        'X_vel_30m', 'X_vel_10m', 'X_vel_5m', 'X_accel_5_30',
        'X_vol_upper_band', 'X_band_break_intensity', 'X_momentum_3m',
        'X_macd', 'X_macd_signal', 'X_macd_hist'
    ]
    return pd.DataFrame(X_list, columns=feature_cols), pd.Series(y_list), feature_cols

# ==============================================================================
# [3단계] 고정밀 XGBoost AI 모델 학습 및 저장
# ==============================================================================
def train_predictive_model():
    print(">> 1. 2/3/4월 과거 전체 데이터셋 로드 및 분석 시작...")
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    
    # 2월, 3월, 4월 과거 데이터 통합 로드 설정
    data_pattern = os.path.join(current_dir, 'DATA', '44호기_[234]월*.csv')
    file_path = sorted(glob.glob(data_pattern))
    
    if len(file_path) > 0:
        df_raw = load_and_label_dataset(file_path) 
        print(">> 2. 고정밀 시계열 예지보전 피처 스펙 추출 중...")
        X, y, feature_cols = create_ml_features_and_targets(df_raw)
    else:
        print(">> [안내] 실 물리 데이터 부재로 훈련용 전조 모의 데이터셋을 생성합니다.")
        X, y, feature_cols = make_synthetic_dataset()
    
    # 시간 순서에 맞게 시계열 데이터 분할 (셔플 안 함)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 클래스 가중치 불균형 보정값 연산
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    scale_weight = num_neg / num_pos if num_pos > 0 else 1.0
    
    print(f">> 3. XGBoost 고집적 예지 진단 모델 학습 수행 (샘플 개수: {X_train.shape[0]}행)...")
    
    # 전조 증상의 예리한 탐지와 노이즈 무시를 조율하기 위한 최적 하이퍼파라미터 구성
    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        scale_pos_weight=scale_weight,  # 불량 샘플 희소성 보완
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,                  # 과적합 방지 규제 강화
        reg_lambda=1.0,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # 성능 검증 출력
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) >= 2 else 1.0
    print(f">> 4. [검증 보고서] 모델 종합 예지 신뢰도 (XGBoost ROC-AUC): {auc_score:.4f}")
    
    # 불량이 발생하는 현장 가동 시나리오에 특화된 경보 임계값 설정
    alert_threshold = 0.40 
    y_pred_class = (y_pred_proba >= alert_threshold).astype(int)
    print(classification_report(y_test, y_pred_class, target_names=['안전', '에러 위험 감지'], zero_division=0))
    
    model_dir = os.path.join(current_dir, 'Model')
    os.makedirs(model_dir, exist_ok=True)
    
    # 파일명 변경 저장 (XGBoost 모델로 세이브)
    joblib.dump(model, os.path.join(model_dir, 'pm_xgb_model.pkl'))
    joblib.dump(feature_cols, os.path.join(model_dir, 'xgb_model_features.pkl'))
    print(">> 5. 경향성 기반 전조 진단 예측 모델 파일 저장 완료.\n")

# ==============================================================================
# [4단계] 실시간 유입 데이터 기반 예지 분석 엔진 (경향성 반영 고성능 추론기)
# ==============================================================================
class RealTimeInferenceEngine:
    def __init__(self, model_path, features_path):
        self.model = None
        self.feature_cols = None
        self.recent_minutes_buffer = []
        
        # 실시간 연속성 유지를 위한 EWMA/MACD 보존 변수들
        self.ewma_5 = 0.0
        self.ewma_30 = 0.0
        self.macd_val = 0.0
        self.macd_signal = 0.0
        self.is_first_ewma = True
        
        self.load_model_artifacts(model_path, features_path)

    def load_model_artifacts(self, model_path, features_path):
        if os.path.exists(model_path) and os.path.exists(features_path):
            try:
                self.model = joblib.load(model_path)
                self.feature_cols = joblib.load(features_path)
                print(">> [엔진] 전조진단 고도화 XGBoost 모델 적재 완료.")
            except Exception as e:
                print(f">> [엔진] 파일 로드 중 오류 발생: {e}")
                self.model = None
                self.feature_cols = None
        else:
            print(">> [엔진] 지정된 경로에 모델 아티팩트가 존재하지 않습니다.")

    def inject_minute_data_and_predict(self, raw_index_stream):
        if self.model is None or self.feature_cols is None:
            return 0.0
            
        current_minute_ng_count = sum(raw_index_stream)
        
        # 30분 타임윈도우 버퍼 유지
        self.recent_minutes_buffer.append(current_minute_ng_count)
        if len(self.recent_minutes_buffer) > 30:
            self.recent_minutes_buffer.pop(0)
            
        buffer_array = np.array(self.recent_minutes_buffer)
        
        # 1. 고밀도 시계열 기초 통계량 산출
        x_sum_30 = float(buffer_array.sum())
        x_sum_10 = float(buffer_array[-10:].sum()) if len(buffer_array) >= 10 else x_sum_30
        x_mean = float(buffer_array.mean())
        x_std = float(buffer_array.std()) if len(buffer_array) > 1 else 0.0
        x_max = float(buffer_array.max())
        
        # 2. 다중 시점 추세 기울기(속도) 연산
        x_vel_30 = float(calculate_trend_slope(self.recent_minutes_buffer))
        x_vel_10 = float(calculate_trend_slope(self.recent_minutes_buffer[-10:])) if len(self.recent_minutes_buffer) >= 10 else x_vel_30
        x_vel_5 = float(calculate_trend_slope(self.recent_minutes_buffer[-5:])) if len(self.recent_minutes_buffer) >= 5 else x_vel_30
        
        # 가속도 정의
        x_accel_5_30 = x_vel_5 - x_vel_30
        
        # 3. 변동성 채널 및 이격도 (Bollinger Band 아이디어)
        x_vol_upper_band = x_mean + (2.0 * x_std)
        x_band_break_intensity = float(max(0.0, current_minute_ng_count - x_vol_upper_band))
        
        # 4. 순간 모멘텀 연산 (3분 전 버퍼 데이터와 대비)
        x_momentum = float(current_minute_ng_count - self.recent_minutes_buffer[-4]) if len(self.recent_minutes_buffer) >= 4 else 0.0
        
        # 5. 실시간 EWMA 및 MACD 상태 업데이트
        if self.is_first_ewma:
            self.ewma_5 = float(current_minute_ng_count)
            self.ewma_30 = float(current_minute_ng_count)
            self.macd_val = 0.0
            self.macd_signal = 0.0
            self.is_first_ewma = False
        else:
            alpha_5 = 2.0 / (5.0 + 1.0)
            alpha_30 = 2.0 / (30.0 + 1.0)
            alpha_sig = 2.0 / (5.0 + 1.0)
            
            self.ewma_5 = alpha_5 * current_minute_ng_count + (1.0 - alpha_5) * self.ewma_5
            self.ewma_30 = alpha_30 * current_minute_ng_count + (1.0 - alpha_30) * self.ewma_30
            
            self.macd_val = self.ewma_5 - self.ewma_30
            self.macd_signal = alpha_sig * self.macd_val + (1.0 - alpha_sig) * self.macd_signal
            
        x_macd = float(self.macd_val)
        x_macd_signal = float(self.macd_signal)
        x_macd_hist = float(self.macd_val - self.macd_signal)
        
        # 컬럼 구조 일관성 정렬 매핑
        input_data = pd.DataFrame([[
            x_sum_30, x_sum_10, x_mean, x_std, x_max,
            x_vel_30, x_vel_10, x_vel_5, x_accel_5_30,
            x_vol_upper_band, x_band_break_intensity, x_momentum,
            x_macd, x_macd_signal, x_macd_hist
        ]], columns=self.feature_cols)
        
        # 추론 확률 반환
        risk_probability = self.model.predict_proba(input_data)[0][1]
        return risk_probability

# ==============================================================================
# [5단계] 실시간 시뮬레이션 진입점
# ==============================================================================
if __name__ == "__main__":
    main_start_time = time.perf_counter()    
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
        abs_model_path = os.path.join(current_dir, 'Model', 'pm_xgb_model.pkl')
        abs_features_path = os.path.join(current_dir, 'Model', 'xgb_model_features.pkl')
        
        print(">> [시스템] 최신 추론을 위해 기존 아티팩트 정리(Clean Reset)를 진행합니다...")
        for file_path in [abs_model_path, abs_features_path]:
            if os.path.exists(file_path):
                os.remove(file_path)
        print(">> [시스템] 기존 모델 청소 완료.\n")
        
        # 실시간 예지 추론기 초기화
        print(">> [시스템] 실시간 추론 엔진 구성 준비...")
        inference_engine = RealTimeInferenceEngine(
            model_path=abs_model_path,
            features_path=abs_features_path
        )
     
        # 모델이 없을 시 가상 대체 훈련 자동 유도
        if inference_engine.model is None or inference_engine.feature_cols is None:
            print("\n>> [시스템 안내] 기 학습 모델 부재로 인공 전조 트레이닝 시퀀스를 호출합니다...")
            train_predictive_model()
            inference_engine.load_model_artifacts(abs_model_path, abs_features_path)
            
            if inference_engine.model is None:
                raise RuntimeError("XGBoost 모델 세대 교체 및 초기화 실패.")

        # 시계열 가동 조건 테스트 데이터 스트림
        time_series_stream = [
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 1분 (총합 1)
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 2분 (총합 3)           
            [0, 0, 0, 3, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 3분 (총합 7)          
            [0, 2, 0, 3, 0, 0, 1, 0, 0, 2, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 4분 (총합 12)
        ]
        
        print("\n>> 6. 실시간 현점 검사 유입 데이터 전조 시그널 및 예측 추론 시뮬레이션 시작:")

        inference_start_time = time.perf_counter()
        
        for t_min, raw_index_stream in enumerate(time_series_stream, start=1):
            current_minute_ng_count = sum(raw_index_stream) 
            prob = inference_engine.inject_minute_data_and_predict(raw_index_stream)
            
            # 예측에 활용된 가속도 및 변동성 변수를 시각화하여 로깅 출력
            current_slope = calculate_trend_slope(inference_engine.recent_minutes_buffer)
            current_macd_hist = inference_engine.macd_val - inference_engine.macd_signal
            
            print(f"[{t_min}분 경과] 분당 불량수: {current_minute_ng_count}ea (불량 발생 가속도 기울기: {current_slope:+.3f} | 에너지 강도: {current_macd_hist:+.3f})")
            print(f"-> [XGBoost 예지결과] 향후 1시간 설비 이상 발생 확률: {prob*100:.2f}%")
            
            # 예측 신뢰 수준에 따른 경고 판단 제어
            if prob >= 0.60:
                print(f"[경보 수신장치 알림] 예지보전 신호 수신! 설비 정비 지시 자동 발송 (위험률: {prob*100:.1f}%)\n")
        
        inference_end_time = time.perf_counter()       

        print("-" * 50)
        print(f">> [성능 리포트] 초고속 4단계 실시간 추론 소요 시간: {inference_end_time - inference_start_time:.6f}초")                  
        
    except Exception as e:
        print(f"\n[오류 발생] {e}")

    finally:
        main_end_time = time.perf_counter()
        print(f">> [시스템] 전체 예지 루틴 수행 완료 (총 소요 시간: {main_end_time - main_start_time:.4f}초)")
        print("-" * 50)
    