import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import lightgbm as lgb
import joblib
import time

# 공차 및 판단 기준 상수 정의
ANGLE_MIN = 107.0
ANGLE_MAX = 113.0
ANGLE_DIFF_MAX = 2.5

X_MIN = 0.82
X_MAX = 0.92
X_DIFF_MAX = 0.1

ERROR_THRESHOLD = 4    # 에러 판단 기준 (불량 4번 이상)
PREDICTION_WINDOW = 60 # 예측 단위 시간 (향후 1시간 = 60분)

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
        
    df_list = [pd.read_csv(f) for f in file_list]
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
        
        is_ng = np.where(~(cond_ang & cond_x), 1, is_ng)
        
    df['IS_NG'] = is_ng
    return df

# ==============================================================================
# [2단계] AI 모델 데이터셋 구성
# ==============================================================================
def create_ml_features_and_targets(df, observation_window='30T', prediction_window='1H'):
    df_time = df.set_index('DATETIME')
    df_min = df_time['IS_NG'].resample('1min').sum().to_frame()
    
    df_min['X_ng_sum_30m'] = df_min['IS_NG'].rolling(window=30, min_periods=1).sum()
    df_min['X_ng_mean_30m'] = df_min['IS_NG'].rolling(window=30, min_periods=1).mean()
    df_min['X_ng_std_30m'] = df_min['IS_NG'].rolling(window=30, min_periods=1).std().fillna(0)
    df_min['X_ng_max_30m'] = df_min['IS_NG'].rolling(window=30, min_periods=1).max()
    
    future_ng_sum = df_min['IS_NG'].iloc[::-1].rolling(window=PREDICTION_WINDOW, min_periods=1).sum().iloc[::-1]
    df_min['Y_TARGET'] = (future_ng_sum >= ERROR_THRESHOLD).astype(int)
    
    feature_cols = ['X_ng_sum_30m', 'X_ng_mean_30m', 'X_ng_std_30m', 'X_ng_max_30m']
    df_ml = df_min.dropna()
    X = df_ml[feature_cols]
    y = df_ml['Y_TARGET']
    
    return X, y, feature_cols

# ==============================================================================
# [3단계] AI 모델 학습 (Pure Native LightGBM 적용)
# ==============================================================================
def train_predictive_model():
    print(">> 1. 44호기 기존 과거 데이터셋 로드 및 전처리 시작...")
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    
    data_pattern = os.path.join(current_dir, 'DATA', '44호기_*월.csv')
    file_path = sorted(glob.glob(data_pattern))        
    df_raw = load_and_label_dataset(file_path) 
    
    print(">> 2. 시계열 기반 머신러닝 피처/타겟 스펙 생성 중...")
    X, y, feature_cols = create_ml_features_and_targets(df_raw)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    scale_weight = num_neg / num_pos if num_pos > 0 else 1.0
    
    # [순정 포인트 1] 데이터를 LightGBM 전용 Dataset 객체로 변환
    train_dataset = lgb.Dataset(X_train, label=y_train)
    test_dataset = lgb.Dataset(X_test, label=y_test, reference=train_dataset)
    
    # LightGBM 순정 파라미터 구성
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'max_depth': 6,
        'scale_pos_weight': scale_weight,
        'random_state': 42,
        'verbose': -1
    }
    
    print(f">> 3. LightGBM AI 모델 학습 시작 (데이터 크기: {X_train.shape[0]}행)...")
    
    # 조기 종료(Early Stopping) 콜백 정의
    callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]
    
    # [순정 포인트 2] lgb.train() 순정 학습 API 호출
    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=500,
        valid_sets=[train_dataset, test_dataset],
        callbacks=callbacks
    )
    
    # 저장 경로를 절대 경로로 고정
    model_dir = os.path.join(current_dir, 'Model')
    os.makedirs(model_dir, exist_ok=True)
    
    # [순정 포인트 3] pickle 대신 가볍고 호환성 극대화된 순정 텍스트 포맷(.txt)으로 저장
    model.save_model(os.path.join(model_dir, 'pm_lightgbm_model.txt')) 
    joblib.dump(feature_cols, os.path.join(model_dir, 'model_features.pkl'))
    print(">> 5. 고급 예지 예측 모델 파일 저장 완료 (pm_lightgbm_model.txt).\n")

# ==============================================================================
# [4단계] 실시간 데이터 기반 확률 추론 엔진 (Native Booster 로드형)
# ==============================================================================
class RealTimeInferenceEngine:
    def __init__(self, model_path, features_path):
        self.model = None
        self.feature_cols = None
        self.recent_minutes_buffer = []
        
        self.load_model_artifacts(model_path, features_path)

    def load_model_artifacts(self, model_path, features_path):
        if os.path.exists(model_path) and os.path.exists(features_path):
            try:
                # 🌟 [순정 포인트 4] lgb.Booster를 직접 호출하여 텍스트 모델 로드 (매우 빠름)
                self.model = lgb.Booster(model_file=model_path)
                self.feature_cols = joblib.load(features_path)
                print(">> [엔진] 순정 LightGBM (Booster) 모델 및 피처 목록 적재 완료.")
            except Exception as e:
                print(f">> [엔진] 파일 로드 중 오류 발생: {e}")
                self.model = None
                self.feature_cols = None
        else:
            print(">> [엔진] 지정된 경로에 모델 또는 피처 파일이 존재하지 않습니다.")
            self.model = None
            self.feature_cols = None

    def inject_minute_data_and_predict(self, raw_index_stream):
        if self.model is None or self.feature_cols is None:
            return 0.0
            
        current_minute_ng_count = sum(raw_index_stream)
        self.recent_minutes_buffer.append(current_minute_ng_count)
        if len(self.recent_minutes_buffer) > 30:
            self.recent_minutes_buffer.pop(0)
            
        buffer_array = np.array(self.recent_minutes_buffer)
        x_sum = buffer_array.sum()
        x_mean = buffer_array.mean()
        
        x_std = buffer_array.std() if len(buffer_array) > 1 else 0.0
        if np.isnan(x_std): 
            x_std = 0.0
            
        x_max = buffer_array.max()
        
        input_data = pd.DataFrame([[x_sum, x_mean, x_std, x_max]], columns=self.feature_cols)
        
        # [순정 포인트 5] 순정 predict()는 이진 분류에서 기본적으로 '클래스 1의 확률값'을 바로 반환합니다.
        risk_probability = self.model.predict(input_data)[0]
        
        return risk_probability

# ==============================================================================
# [5단계] 수정한 실시간 시뮬레이션 진입점
# ==============================================================================
if __name__ == "__main__":
    # 전체 메인 루틴 시작 시간 기록
    main_start_time = time.perf_counter()    
    #    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
        # 확장자를 .txt로 변경
        abs_model_path = os.path.join(current_dir, 'Model', 'pm_lightgbm_model.txt') 
        abs_features_path = os.path.join(current_dir, 'Model', 'model_features.pkl')

        print(">> [시스템] 신규 추론 및 폴백 제어를 위해 기존 잔재 모델 파일 클리닝을 시작합니다...")
        for file_path in [abs_model_path, abs_features_path]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"기존 파일 삭제 완료: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"파일 삭제 중 오류 발생 ({os.path.basename(file_path)}): {e}")
        print(">> [시스템] 기존 모델 디렉터리 클리닝 완료.\n")
        
        print(">> [시스템] 실시간 추론 엔진 초기화를 시작합니다...")
        inference_engine = RealTimeInferenceEngine(
            model_path=abs_model_path,
            features_path=abs_features_path
        )
     
        if inference_engine.model is None or inference_engine.feature_cols is None:
            print("\n>> [시스템 안내] 엔진 내부 자동 로드에 실패했습니다. 메인 루틴에서 강제 학습을 실행합니다...")
            train_predictive_model()
            
            print(">> [시스템] 강제 학습 완료. 엔진에 아티팩트를 다시 로드합니다.")
            inference_engine.load_model_artifacts(abs_model_path, abs_features_path)
            
            if inference_engine.model is None:
                raise RuntimeError("모델 학습을 완료했으나 파일 적재(Load)에 최종 실패했습니다. 저장 경로 및 파일명을 점검하세요.")

        # -----------------------------------------------------------------------------------------
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
            
            print(f"[{t_min}분 경과] 불량 수: {current_minute_ng_count}ea "
                  f"-> 예지분석: '향후 1시간 내 에러 발생 확률 {prob*100:.1f}%'")            
            if prob >= 0.60:
                print(f"[알람 제어 단말 알림] 위험 수준 도달 ({prob*100:.0f}%) -> 예지보전 가동 (정비 지시)\n")
        inference_end_time = time.perf_counter()       

        print("-" * 50)
        print(f">> [성능 리포트] 실시간 4분 추론 연산 소요 시간: {inference_end_time - inference_start_time:.6f}초") 

    except FileNotFoundError as e:
        print(f"\n[경로 오류] CSV 데이터 소스 또는 모델 저장 디렉토리를 확인해 주세요. {e}")
    except RuntimeError as e:
        print(f"\n[엔진 치명오류] {e}")
    except Exception as e:
        print(f"\n[런타임 오류] 테스트 진행 중 예외 발생: {e}")
    finally:
        # 최종 메인 루틴 종료 시간 기록 및 출력
        main_end_time = time.perf_counter()
        print(f">> [시스템] 전체 메인 루틴 최종 실행 시간: {main_end_time - main_start_time:.4f}초")
        print("-" * 50)          
    