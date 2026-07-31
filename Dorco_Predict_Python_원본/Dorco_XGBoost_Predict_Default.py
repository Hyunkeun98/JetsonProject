import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
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
    slope, _ = np.polyfit(x, y, 1)
    return slope

# ==============================================================================
# [1단계] 데이터 로드 및 전수 양불 판정 (0/1)
# ==============================================================================
def load_and_label_dataset(file_pattern):
    file_list = []
    
    # 1. 입력된 인자가 리스트(list) 형태인 경우 처리
    if isinstance(file_pattern, list):
        for pattern in file_pattern:
            file_list.extend(glob.glob(pattern))
    # 2. 단일 문자열(str) 형태인 경우 처리
    else:
        file_list = glob.glob(file_pattern)
        
    # 중복 경로 제거 및 파일명 순서대로 정렬
    file_list = sorted(list(set(file_list)))

    if not file_list:
        raise FileNotFoundError(f"학습에 사용할 CSV 파일이 없습니다. 입력 경로: {file_pattern}")
        
    df_list = [pd.read_csv(f) for f in file_list]
    df = pd.concat(df_list, ignore_index=True)
    
    # 시간축 정렬
    df['DATETIME'] = pd.to_datetime(
        df['INSP_DATE'].astype(str) + df['INSP_TIME'].astype(str).str.zfill(6), 
        format='%Y%m%d%H%M%S', errors='coerce'
    )
    df = df.dropna(subset=['DATETIME']).sort_values(by='DATETIME').reset_index(drop=True)
    
    # 스케일 다운 (1000배)
    target_cols = [c for c in df.columns if c.startswith(('A_L_', 'A_R_', 'X_L_', 'X_R_'))]
    df[target_cols] = df[target_cols] / 1000.0

    # 종합 불량 판정
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
# [2단계] AI 모델 데이터셋 구성 (최근 30분 경향성 기반 -> 향후 1시간 예측)
# ==============================================================================
def create_ml_features_and_targets(df, observation_window='30T', prediction_window='1H'):
    df_time = df.set_index('DATETIME')
    
    # 1. 1분 단위로 불량 개수 1차 집계
    df_min = df_time['IS_NG'].resample('1min').sum().to_frame()
    
    # 2. 특징량(X) 생성
    df_min['X_ng_sum_30m'] = df_min['IS_NG'].rolling(window=30, min_periods=1).sum()
    df_min['X_ng_mean_30m'] = df_min['IS_NG'].rolling(window=30, min_periods=1).mean()
    df_min['X_ng_std_30m'] = df_min['IS_NG'].rolling(window=30, min_periods=1).std().fillna(0)
    df_min['X_ng_max_30m'] = df_min['IS_NG'].rolling(window=30, min_periods=1).max()
    
    # [수정] 엔진 피처와 개수를 일치시키기 위해 기울기 및 EWMA 피처를 학습용 데이터셋에 추가 반영합니다.
    df_min['X_velocity'] = df_min['IS_NG'].rolling(window=30, min_periods=2).apply(calculate_trend_slope, raw=True).fillna(0)
    
    ewma_s = df_min['IS_NG'].ewm(span=5, adjust=False).mean()
    ewma_l = df_min['IS_NG'].ewm(span=30, adjust=False).mean()
    df_min['X_ewma_ratio'] = ewma_s - ewma_l
    
    # 3. 정답 라벨(Y) 생성
    future_ng_sum = df_min['IS_NG'].iloc[::-1].rolling(window=PREDICTION_WINDOW, min_periods=1).sum().iloc[::-1]
    df_min['Y_TARGET'] = (future_ng_sum >= ERROR_THRESHOLD).astype(int)
    
    # 모델 학습에 사용할 피처 컬럼 세팅 정의 (엔진의 input_data 순서와 동일)
    feature_cols = ['X_ng_sum_30m', 'X_ng_mean_30m', 'X_ng_std_30m', 'X_ng_max_30m', 'X_velocity', 'X_ewma_ratio']
    
    df_ml = df_min.dropna()
    X = df_ml[feature_cols]
    y = df_ml['Y_TARGET']
    
    return X, y, feature_cols

# ==============================================================================
# [3단계] AI 모델 학습 (Training Pipeline)
# ==============================================================================
def train_predictive_model():
    print(">> 1. 44호기 기존 과거 데이터셋 로드 및 전처리 시작...")
    file_path = sorted(glob.glob("./DATA/44호기_*월.csv"))        
    df_raw = load_and_label_dataset(file_path) 
    
    print(">> 2. 시계열 기반 머신러닝 피처/타겟 스펙 생성 중...")
    X, y, feature_cols = create_ml_features_and_targets(df_raw)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    scale_weight = num_neg / num_pos if num_pos > 0 else 1.0
    
    print(f">> [참고] 불량 데이터 가중치(scale_pos_weight) 계산 결과: {scale_weight:.2f}배")

    params = {
        'objective': 'binary:logistic',     
        'eval_metric': 'auc',              
        'learning_rate': 0.05,              
        'max_depth': 6,                    
        'scale_pos_weight': scale_weight,   
        'tree_method': 'hist',              
        'random_state': 42,
        'verbosity': 0                      
    }
    
    print(f">> 3. XGBoost AI 모델 학습 시작 (데이터 크기: {X_train.shape[0]}행)...")
    
    model = xgb.XGBClassifier(**params, n_estimators=500, early_stopping_rounds=30)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False  
    )
    
    print(f">> 조기 종료 최적 반복 횟수(Best Iteration): {model.best_iteration}")
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\n>> 4. 모델 검증 완료. 예측 신뢰도(XGBoost AUC Score): {auc_score:.4f}")
    
    custom_threshold = 0.15
    y_pred_class = (y_pred_proba >= custom_threshold).astype(int)
    
    print(f"\n[상세 검증 지표 - 예지보전 임계값: {custom_threshold}]")
    print(classification_report(y_test, y_pred_class, target_names=['정상', '위험(에러4회이상)'], zero_division=0))
    
    # [수정] 로드 환경과 일치시키기 위해 모델 파일을 절대 경로 상의 'Model' 디렉터리 내에 물리적으로 저장합니다.
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    save_dir = os.path.join(current_dir, 'Model')
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, 'pm_xgboost_model.json')
    features_path = os.path.join(save_dir, 'xgboost_model_features.pkl')
    
    model.save_model(model_path)
    joblib.dump(feature_cols, features_path)
    print(f">> 5. 고급 XGBoost 예측 모델 및 피처 명세서 저장 완료 -> 저장경로: {save_dir}\n")

# ==============================================================================
# [4단계] 실시간 데이터 기반 확률 추론 (Inference Engine)
# ==============================================================================
class RealTimeInferenceEngine:
    def __init__(self, model_path='Model/pm_xgboost_model.json', features_path='Model/xgboost_model_features.pkl'):
        self.model = None
        self.feature_cols = None
        self.recent_minutes_buffer = []
        # EWMA의 연속성 유지를 위한 실시간 상태 변수
        self.ewma_short = 0.0
        self.ewma_long = 0.0
        self.is_first_ewma = True
        
        self.load_model_artifacts(model_path, features_path)

    def load_model_artifacts(self, model_path, features_path):
        import os
        import joblib
        import xgboost as xgb
        
        if os.path.exists(model_path) and os.path.exists(features_path):
            try:
                self.model = xgb.XGBClassifier()
                self.model.load_model(model_path)
                self.feature_cols = joblib.load(features_path)
                print(">> [엔진] 전조진단 고도화 XGBoost 모델 적재 완료.")
            except Exception as e:
                print(f">> [엔진] 파일 로드 중 오류 발생: {e}")
                self.model = None
                self.feature_cols = None
        else:
            print(">> [엔진] 지정된 경로에 아티팩트가 존재하지 않습니다.")

    def inject_minute_data_and_predict(self, raw_index_stream):
        if self.model is None or self.feature_cols is None:
            return 0.0
            
        current_minute_ng_count = sum(raw_index_stream)
        
        # 30분 타임윈도우 버퍼 유지
        self.recent_minutes_buffer.append(current_minute_ng_count)
        if len(self.recent_minutes_buffer) > 30:
            self.recent_minutes_buffer.pop(0)
            
        buffer_array = np.array(self.recent_minutes_buffer)
        
        # 1. 기본 통계량 계산
        x_sum = buffer_array.sum()
        x_mean = buffer_array.mean()
        x_std = buffer_array.std() if len(buffer_array) > 1 else 0.0
        if np.isnan(x_std): 
            x_std = 0.0
        x_max = buffer_array.max()
        
        # 2. 전조증상 고도화 수치 1: 기울기(속도) 추세 연산
        x_velocity = calculate_trend_slope(self.recent_minutes_buffer)
        
        # 3. 전조증상 고도화 수치 2: EWMA 실시간 필터링
        if self.is_first_ewma:
            self.ewma_short = float(current_minute_ng_count)
            self.ewma_long = float(current_minute_ng_count)
            self.is_first_ewma = False
        else:
            alpha_s = 2.0 / (5.0 + 1.0)
            alpha_l = 2.0 / (30.0 + 1.0)
            self.ewma_short = alpha_s * current_minute_ng_count + (1.0 - alpha_s) * self.ewma_short
            self.ewma_long = alpha_l * current_minute_ng_count + (1.0 - alpha_l) * self.ewma_long
            
        x_ewma_ratio = self.ewma_short - self.ewma_long
        
        # 데이터프레임 빌딩 및 컬럼 매핑 순서 일치화 (6개 피처)
        input_data = pd.DataFrame([[
            x_sum, x_mean, x_std, x_max, x_velocity, x_ewma_ratio
        ]], columns=self.feature_cols)
        
        probabilities = self.model.predict_proba(input_data)
        risk_probability = probabilities[0][1]
        
        return risk_probability

# ==============================================================================
# [5단계] 전체 시뮬레이션 및 실행 테스트
# ==============================================================================
if __name__ == "__main__":
    # 전체 메인 루틴 시작 시간 기록
    main_start_time = time.perf_counter()    
    #    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
        abs_model_path = os.path.join(current_dir, 'Model', 'pm_xgboost_model.json')
        abs_features_path = os.path.join(current_dir, 'Model', 'xgboost_model_features.pkl')

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
        print(">> [시스템] 실시간 추론 엔진 초기화를 시작합니다...")
        inference_engine = RealTimeInferenceEngine(
            model_path=abs_model_path,
            features_path=abs_features_path
        )
     
        if inference_engine.model is None or inference_engine.feature_cols is None:
            print("\n>> [시스템 안내] 엔진 로드 실패로 가상 Fallback 학습을 유도합니다...")
            train_predictive_model()
            inference_engine.load_model_artifacts(abs_model_path, abs_features_path)
            
            if inference_engine.model is None:
                raise RuntimeError("모델 학습 및 복구 적재 프로세스 최종 실패.")

        # -----------------------------------------------------------------------------------------
        # 가상의 가동 상황 시뮬레이션 데이터
        time_series_stream = [
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 1분 (총합 1)
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 2분 (총합 3)           
            [0, 0, 0, 3, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 3분 (총합 7)          
            [0, 2, 0, 3, 0, 0, 1, 0, 0, 2, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 4분 (총합 12)
        ]
        
        print("\n>> 6. 실시간 현장 데이터 유입에 따른 AI 예지 확률 추론 테스트 시작:")

        # [시간 측정] 추론 루프 소요 시간 측정
        inference_start_time = time.perf_counter()
        #          
        for t_min, raw_index_stream in enumerate(time_series_stream, start=1):
            current_minute_ng_count = sum(raw_index_stream) 
            prob = inference_engine.inject_minute_data_and_predict(raw_index_stream)
            
            current_slope = calculate_trend_slope(inference_engine.recent_minutes_buffer)
            
            print(f"[{t_min}분 경과] 불량 발생 수: {current_minute_ng_count}ea (불량 증가 가속도 기울기: {current_slope:+.3f})")
            print(f"-> [AI 예지분석 결과] 향후 1시간 내 설비 에러(불량 4회 폭발) 발생 위험률: {prob*100:.2f}%")
            
            if prob >= 0.60:
                print(f"[알람 제어 단말 알림] 위험 수준 도달 ({prob*100:.0f}%) -> 예지보전 가동 (정비 지시)\n")
        inference_end_time = time.perf_counter()       

        print("-" * 50)
        print(f">> [성능 리포트] 실시간 4분 추론 연산 소요 시간: {inference_end_time - inference_start_time:.6f}초") 

    except Exception as e:
        print(f"\n[오류 발생] {e}")
        
    finally:
        # 최종 메인 루틴 종료 시간 기록 및 출력
        main_end_time = time.perf_counter()
        print(f">> [시스템] 전체 메인 루틴 최종 실행 시간: {main_end_time - main_start_time:.4f}초")
        print("-" * 50)        
