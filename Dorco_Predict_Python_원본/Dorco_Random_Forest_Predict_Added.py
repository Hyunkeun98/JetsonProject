import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib
import time

# 공차 및 판단 기준 설정
ANGLE_MIN = 107.0
ANGLE_MAX = 113.0
ANGLE_DIFF_MAX = 2.5

X_MIN = 0.82
X_MAX = 0.92
X_DIFF_MAX = 0.1

ERROR_THRESHOLD = 4    # 에러 판단 기준 (불량 4번 이상)
PREDICTION_WINDOW = 60 # 예측 단위 시간 (향후 1시간 = 60분)

# ==============================================================================
# [1단계] 데이터 로드 및 전수 양불 판정
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
# [2단계] AI 모델 데이터셋 구성 (전조 증상/가속도/경향성 수치화 고도화)
# ==============================================================================
def create_ml_features_and_targets(df, observation_window='30T', prediction_window='1H'):
    df_time = df.set_index('DATETIME')
    
    def get_max_single_shot(series):
        return series.max() if len(series) > 0 else 0
        
    def get_ng_ratio(series):
        return (series > 0).mean() if len(series) > 0 else 0.0
        
    def get_max_consecutive_ng(series):
        if len(series) == 0: return 0
        is_ng_seq = (series > 0).astype(int).values
        max_consec = 0
        current_consec = 0
        for val in is_ng_seq:
            if val == 1:
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 0
        return max_consec

    resampled_ng = df_time['IS_NG'].resample('1min')

    df_min = pd.DataFrame()
    df_min['IS_NG'] = resampled_ng.sum()
    df_min['RAW_max_single_shot'] = resampled_ng.apply(get_max_single_shot)
    df_min['RAW_ng_ratio'] = resampled_ng.apply(get_ng_ratio)
    df_min['RAW_max_consecutive'] = resampled_ng.apply(get_max_consecutive_ng)
    
    df_min = df_min.interpolate(method='linear')

    # [기존 피처] 최근 30분 기본 통계량
    df_min['X_ng_sum_30m'] = df_min['IS_NG'].rolling(window=30, min_periods=1).sum()
    df_min['X_ng_mean_30m'] = df_min['IS_NG'].rolling(window=30, min_periods=1).mean()
    df_min['X_ng_std_30m'] = df_min['IS_NG'].rolling(window=30, min_periods=1).std().fillna(0)
    df_min['X_ng_max_30m'] = df_min['IS_NG'].rolling(window=30, min_periods=1).max()
    df_min['X_raw_max_shot_30m'] = df_min['RAW_max_single_shot'].rolling(window=30, min_periods=1).max()
    df_min['X_raw_ratio_30m'] = df_min['RAW_ng_ratio'].rolling(window=30, min_periods=1).mean()
    df_min['X_raw_consec_30m'] = df_min['RAW_max_consecutive'].rolling(window=30, min_periods=1).max()

    # [신규 고도화 피처] 전조 증상 수치화를 위한 경향성 및 가속도 추가
    # 1) 최근 데이터에 민감하게 반응하는 지수이동평균(EWMA)
    df_min['X_ng_ewma_30m'] = df_min['IS_NG'].ewm(span=30, adjust=False).mean()
    
    # 2) 불량 발생 가속도(Velocity): 최근 5분/10분 전 대비 현재 얼마나 불량이 폭증하고 있는가?
    df_min['X_ng_velocity_5m'] = df_min['IS_NG'] - df_min['IS_NG'].shift(5).fillna(0)
    df_min['X_ng_velocity_10m'] = df_min['IS_NG'] - df_min['IS_NG'].shift(10).fillna(0)
    
    # 3) 불량 누적 속도의 변화량 (Diff)
    df_min['X_ng_sum_diff_1m'] = df_min['X_ng_sum_30m'].diff(1).fillna(0)

    # 정답 라벨(Y) 생성
    future_ng_sum = df_min['IS_NG'].iloc[::-1].rolling(window=PREDICTION_WINDOW, min_periods=1).sum().iloc[::-1]
    df_min['Y_TARGET'] = (future_ng_sum >= ERROR_THRESHOLD).astype(int)
    
    # 고도화된 최종 피처 리스트
    feature_cols = [
        'X_ng_sum_30m', 'X_ng_mean_30m', 'X_ng_std_30m', 'X_ng_max_30m',
        'X_raw_max_shot_30m', 'X_raw_ratio_30m', 'X_raw_consec_30m',
        'X_ng_ewma_30m', 'X_ng_velocity_5m', 'X_ng_velocity_10m', 'X_ng_sum_diff_1m'
    ]
    
    df_ml = df_min.dropna()
    X = df_ml[feature_cols]
    y = df_ml['Y_TARGET']
    
    return X, y, feature_cols

# ==============================================================================
# [3단계] AI 모델 학습 (고도화된 Random Forest 알고리즘 파이프라인)
# ==============================================================================
def train_predictive_model():
    print(">> 1. 44호기 기존 과거 데이터셋 로드 및 전처리 시작...")
    current_dir = os.path.dirname(os.path.abspath(__file__))    
    target_pattern = os.path.join(current_dir, "DATA", "44호기_*월.csv")
    file_path = sorted(glob.glob(target_pattern))
    
    df_raw = load_and_label_dataset(file_path)    
    
    print(">> 2. 시계열 경향성/가속도 기반 머신러닝 피처 생성 중...")
    X, y, feature_cols = create_ml_features_and_targets(df_raw)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)   
    
    print(f">> 3. 고도화된 Random Forest 모델 학습 시작 (데이터 크기: {X_train.shape[0]}행)...")
    
    # [Random Forest 고도화 세팅]
    # - n_estimators: 300 (의사결정 나무 모델 수 확장으로 다수결 신뢰도 및 확률 해상도 강화)
    # - class_weight='balanced_subsample': 희소하게 일어나는 불량 전조 가중치 불균형 보정
    # - min_samples_leaf=3: 오버피팅 제어 및 확률 스무딩 효과 안정화
    model = RandomForestClassifier(
        n_estimators=150,                   # 생성할 트리 개수 300 > 150
        max_depth=8,                        # 트리 깊이 제한   10 > 8
        min_samples_leaf=3,                 # 부스팅 학습률  3 > 0.05
        class_weight='balanced',            # balanced_subsample > balanced
        random_state=42, 
        n_jobs=-1                           # 모든 CPU 코어 사용
    )
        
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    if len(np.unique(y_test)) < 2:
        score = 0.5
    else:
        score = roc_auc_score(y_test, y_pred_proba)
        
    print(f">> 4. 모델 검증 완료. 예측 신뢰도(AUC Score): {score:.4f}")
    
    # 훈련 변수 중요도 시각화 보조 피드백
    importances = model.feature_importances_
    print("\n[고도화 피처 중요도 파악]")
    for name, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        print(f" - {name}: {imp:.4f}")
    print("-" * 40)

    model_dir = os.path.join(current_dir, 'Model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_save_path = os.path.join(model_dir, 'pm_rf_model.pkl')
    features_save_path = os.path.join(model_dir, 'rf_model_features.pkl')
    
    joblib.dump(model, model_save_path)
    joblib.dump(feature_cols, features_save_path)
    print(f">> 5. 고도화 Random Forest 예지보전 모델 파일 저장 완료.\n")

# ==============================================================================
# [4단계] 실시간 데이터 기반 확률 추론 (인덱스 에러 방어형 가속도 버퍼 로직 내장)
# ==============================================================================
class RealTimeInferenceEngine:
    def __init__(self, model_path, features_path):    
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.feature_cols = joblib.load(features_path)
            print(">> [엔진] 저장된 고도화 Random Forest 모델을 로드했습니다.")
        else:
            self.model = None
            print(">> [엔진] 학습된 모델 파일이 없습니다.")
            
        self.buffer = []

    def inject_minute_data_and_predict(self, raw_minute_list):
        if self.model is None: return 0.0
        raw_arr = np.array(raw_minute_list)
        if len(raw_arr) == 0: return 0.0
        
        ng_sum = int(raw_arr.sum())
        max_single_shot = int(raw_arr.max())
        ng_ratio = float((raw_arr > 0).mean())
        
        is_ng_seq = (raw_arr > 0).astype(int)
        max_consec = 0
        current_consec = 0
        for val in is_ng_seq:
            if val == 1:
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 0
                
        minute_summary = {
            'ng_count': ng_sum,
            'max_single_shot': max_single_shot,
            'ng_ratio': ng_ratio,
            'max_consec': max_consec
        }
        
        self.buffer.append(minute_summary)
        if len(self.buffer) > 30:
            self.buffer.pop(0)
            
        df_buf = pd.DataFrame(self.buffer)
        
        # 기본 30분 통계량 계산
        x_ng_sum = df_buf['ng_count'].sum()
        x_ng_mean = df_buf['ng_count'].mean()
        x_ng_std = df_buf['ng_count'].std() if len(df_buf) > 1 else 0.0
        x_ng_max = df_buf['ng_count'].max()
        x_raw_max_shot = df_buf['max_single_shot'].max()
        x_raw_ratio = df_buf['ng_ratio'].mean()
        x_raw_consec = df_buf['max_consec'].max()
        
        # [실시간 엔진 계산] 시계열 패턴 분석을 위한 가속도 결합
        x_ng_ewma = df_buf['ng_count'].ewm(span=30, adjust=False).mean().iloc[-1]
        
        # [IndexError 방어] 초기 기동 시점 데이터가 부족할 때 슬라이싱 에러 원천 차단
        x_ng_velocity_5m = ng_sum - df_buf['ng_count'].iloc[-6] if len(df_buf) >= 6 else ng_sum
        x_ng_velocity_10m = ng_sum - df_buf['ng_count'].iloc[-11] if len(df_buf) >= 11 else ng_sum
        x_ng_sum_diff_1m = df_buf['ng_count'].diff(1).fillna(0).iloc[-1]
        
        # 모델 피처 컬럼명 순서 맞춤
        input_features = [
            x_ng_sum, x_ng_mean, x_ng_std, x_ng_max,
            x_raw_max_shot, x_raw_ratio, x_raw_consec,
            x_ng_ewma, x_ng_velocity_5m, x_ng_velocity_10m, x_ng_sum_diff_1m
        ]
        
        input_data = pd.DataFrame([input_features], columns=self.feature_cols)
        
        probabilities = self.model.predict_proba(input_data)
        return probabilities[0][1]

# ==============================================================================
# [5단계] 전체 시뮬레이션 및 실행 테스트
# ==============================================================================
if __name__ == "__main__":
    # 전체 메인 루틴 시작 시간 기록
    main_start_time = time.perf_counter()  
    #  
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        abs_model_path = os.path.join(current_dir, 'Model', 'pm_rf_model.pkl')
        abs_features_path = os.path.join(current_dir, 'Model', 'rf_model_features.pkl')

        # 0. 기존 잔재 모델 아티팩트 삭제 (Clean Reset 루틴)
        print(">> [시스템] 신규 추론 및 폴백 제어를 위해 기존 잔재 모델 파일 클리닝을 시작합니다...")
        for file_path in [abs_model_path, abs_features_path]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"기존 파일 삭제 완료: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"파일 삭제 중 오류 발생 ({os.path.basename(file_path)}): {e}")
        print(">> [시스템] 기존 모델 디렉터리 클리닝 완료.\n")
        
        # 1. 실시간 추론 엔진 초기화
        # 엔진 생성 시 절대 경로를 매개변수로 명시해 주어야 내부 __init__에서 파일을 안정적으로 로드합니다.        
        print(">> [시스템] 실시간 추론 엔진 초기화 중...")
        inference_engine = RealTimeInferenceEngine(model_path=abs_model_path, features_path=abs_features_path)
     
        if inference_engine.model is None:
            print("\n>> [시스템 안내] 기존 모델이 없어 고도화 Random Forest Fallback 학습을 시작합니다...")
            train_predictive_model()
            inference_engine = RealTimeInferenceEngine(model_path=abs_model_path, features_path=abs_features_path)

        # 가상의 가동 상황 데이터 (1분: 1개 -> 2분: 3개 -> 3분: 7개 -> 4분: 12개 폭증 상황)
        time_series_stream = [
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],            
            [0, 0, 0, 3, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],            
            [0, 2, 0, 3, 0, 0, 1, 0, 0, 2, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
        ]
        
        print("\n>> 6. 실시간 현장 데이터 유입에 따른 고도화 RF 예지 확률 추론 테스트 시작:")

        # [시간 측정] 추론 루프 소요 시간 측정
        inference_start_time = time.perf_counter()        
        #
        for t_min, raw_index_stream in enumerate(time_series_stream, start=1):
            current_minute_ng_count = sum(raw_index_stream) 
            prob = inference_engine.inject_minute_data_and_predict(raw_index_stream)
            
            print(f"[{t_min}분 경과] 현재분 불량 수: {current_minute_ng_count}ea "
                  f"-> 예지분석: '향후 1시간 내 설비 에러 발생 확률은 {prob*100:.1f}% 입니다.'")            
            if prob >= 0.60:
                print(f"[알람 제어 단말] 위험 수준 도달 예보 ({prob*100:.0f}%) -> 조기 정비 지시\n")
        inference_end_time = time.perf_counter()        

        print("-" * 50)
        print(f">> [성능 리포트] 실시간 4분 추론 연산 소요 시간: {inference_end_time - inference_start_time:.6f}초")
        
    except Exception as e:
        print(f"\n[오류 발생] 테스트 진행 중 예외 발생: {e}")
        
    finally:
        # 최종 메인 루틴 종료 시간 기록 및 출력
        main_end_time = time.perf_counter()
        print(f">> [시스템] 전체 메인 루틴 최종 실행 시간: {main_end_time - main_start_time:.4f}초")
        print("-" * 50)        
