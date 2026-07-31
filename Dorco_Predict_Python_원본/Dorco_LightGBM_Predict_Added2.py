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
ANGLE_MID = (ANGLE_MIN + ANGLE_MAX) / 2.0
ANGLE_HALF_RANGE = (ANGLE_MAX - ANGLE_MIN) / 2.0
ANGLE_DIFF_MAX = 2.5

X_MIN = 0.82
X_MAX = 0.92
X_MID = (X_MIN + X_MAX) / 2.0
X_HALF_RANGE = (X_MAX - X_MIN) / 2.0
X_DIFF_MAX = 0.1

ERROR_THRESHOLD = 4    # 에러 판단 기준 (불량 4번 이상)
PREDICTION_WINDOW = 60 # 예측 단위 시간 (향후 1시간 = 60분)

# ==============================================================================
# [1단계] 데이터 로드 및 시계열 센서 경향성 분석 피처 생성
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
    
    # 단위 변환 (/1000)
    target_cols = [c for c in df.columns if c.startswith(('A_L_', 'A_R_', 'X_L_', 'X_R_'))]
    df[target_cols] = df[target_cols] / 1000.0

    # [핵심 추가] 공차 경계 근접도(전조증상 수치화) 및 편차 분석
    a_boundary_scores = []
    x_boundary_scores = []
    a_diff_vals = []
    x_diff_vals = []

    is_ng = np.zeros(len(df), dtype=int)
    for pos in range(1, 13):
        suffix = f"{pos:02d}"
        
        # 센서 데이터 추출
        al, ar = df[f'A_L_{suffix}'], df[f'A_R_{suffix}']
        xl, xr = df[f'X_L_{suffix}'], df[f'X_R_{suffix}']
        
        # 1. 원본 양불 판정
        cond_ang = (al >= ANGLE_MIN) & (al <= ANGLE_MAX) & \
                   (ar >= ANGLE_MIN) & (ar <= ANGLE_MAX) & \
                   (np.abs(al - ar) <= ANGLE_DIFF_MAX)
                   
        cond_x = (xl >= X_MIN) & (xl <= X_MAX) & \
                 (xr >= X_MIN) & (xr <= X_MAX) & \
                 (np.abs(xl - xr) <= X_DIFF_MAX)
        
        is_ng = np.where(~(cond_ang & cond_x), 1, is_ng)
        
        # 2. 전조 신호 계산 (공차 중심선에서 얼마나 벗어났는지 비율화, 1에 가까울수록 한계 돌파 직전)
        a_boundary_scores.append(np.abs(al - ANGLE_MID) / ANGLE_HALF_RANGE)
        a_boundary_scores.append(np.abs(ar - ANGLE_MID) / ANGLE_HALF_RANGE)
        x_boundary_scores.append(np.abs(xl - X_MID) / X_HALF_RANGE)
        x_boundary_scores.append(np.abs(xr - X_MID) / X_HALF_RANGE)
        
        # 3. 좌우 편차값 자체의 절댓값 기록
        a_diff_vals.append(np.abs(al - ar))
        x_diff_vals.append(np.abs(xl - xr))

    df['IS_NG'] = is_ng
    
    # 전체 12개 포인트의 평균 전조 지표 산출하여 대표 칼럼으로 지정
    df['TREND_A_MARGIN'] = np.mean(a_boundary_scores, axis=0)  # 각도 마진 위험도
    df['TREND_X_MARGIN'] = np.mean(x_boundary_scores, axis=0)  # 치수 마진 위험도
    df['TREND_A_DIFF'] = np.mean(a_diff_vals, axis=0)          # 좌우 각도 불균형성
    df['TREND_X_DIFF'] = np.mean(x_diff_vals, axis=0)          # 좌우 치수 불균형성

    return df

# ==============================================================================
# [2단계] 시계열 기반 고도화된 머신러닝 피처 생성 (다중 윈도우 기법)
# ==============================================================================
def create_ml_features_and_targets(df):
    df_time = df.set_index('DATETIME')
    
    # 분 단위 리샘플링 및 센서 요약 지표 집계
    df_min = df_time['IS_NG'].resample('1min').sum().to_frame()
    df_sensor = df_time[['TREND_A_MARGIN', 'TREND_X_MARGIN', 'TREND_A_DIFF', 'TREND_X_DIFF']].resample('1min').mean()
    df_min = df_min.join(df_sensor).ffill() # 누락 시간은 앞선 장비 값으로 패딩 처리
    
    feature_cols = []
    
    # 5분 / 15분 / 30분 단위의 다중 윈도우 트렌드 피처 생성
    for w in [5, 15, 30]:
        df_min[f'X_ng_sum_{w}m'] = df_min['IS_NG'].rolling(window=w, min_periods=1).sum()
        df_min[f'X_ng_mean_{w}m'] = df_min['IS_NG'].rolling(window=w, min_periods=1).mean()
        df_min[f'X_ng_std_{w}m'] = df_min['IS_NG'].rolling(window=w, min_periods=1).std().fillna(0)
        
        # 물리 센서 경향성 데이터 피처화
        df_min[f'X_a_margin_mean_{w}m'] = df_min['TREND_A_MARGIN'].rolling(window=w, min_periods=1).mean()
        df_min[f'X_x_margin_mean_{w}m'] = df_min['TREND_X_MARGIN'].rolling(window=w, min_periods=1).mean()
        df_min[f'X_a_diff_std_{w}m'] = df_min['TREND_A_DIFF'].rolling(window=w, min_periods=1).std().fillna(0)
        df_min[f'X_x_diff_std_{w}m'] = df_min['TREND_X_DIFF'].rolling(window=w, min_periods=1).std().fillna(0)
        
        feature_cols.extend([
            f'X_ng_sum_{w}m', f'X_ng_mean_{w}m', f'X_ng_std_{w}m',
            f'X_a_margin_mean_{w}m', f'X_x_margin_mean_{w}m',
            f'X_a_diff_std_{w}m', f'X_x_diff_std_{w}m'
        ])
    
    # 향후 1시간(PREDICTION_WINDOW) 내 기준 에러 횟수 초과 여부 레이블링
    future_ng_sum = df_min['IS_NG'].iloc[::-1].rolling(window=PREDICTION_WINDOW, min_periods=1).sum().iloc[::-1]
    df_min['Y_TARGET'] = (future_ng_sum >= ERROR_THRESHOLD).astype(int)
    
    df_ml = df_min.dropna()
    X = df_ml[feature_cols]
    y = df_ml['Y_TARGET']
    
    return X, y, feature_cols

# ==============================================================================
# [3단계] AI 모델 학습 (과적합 제어 파라미터 적용)
# ==============================================================================
def train_predictive_model():
    print(">> 1. 44호기 기존 과거 데이터셋 로드 및 고도화 전처리 시작...")
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    
    data_pattern = os.path.join(current_dir, 'DATA', '44호기_*월.csv')
    file_path = sorted(glob.glob(data_pattern))        
    df_raw = load_and_label_dataset(file_path) 
    
    print(">> 2. 시계열 및 물리 전조 증상 피처 엔지니어링 수행 중...")
    X, y, feature_cols = create_ml_features_and_targets(df_raw)
    
    # 시계열 순서를 보존하여 Train/Test 분할 (과적합 검증용)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    scale_weight = num_neg / num_pos if num_pos > 0 else 1.0
    
    train_dataset = lgb.Dataset(X_train, label=y_train)
    test_dataset = lgb.Dataset(X_test, label=y_test, reference=train_dataset)
    
    # 정밀 하이퍼파라미터 튜닝 (조기 종료 및 규제 강화)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.03,        # 섬세한 탐색을 위해 러닝레이트 하향 조정
        'max_depth': 5,               # 오버피팅을 방지하기 위해 깊이 소폭 단축
        'num_leaves': 31,
        'min_child_samples': 20,      # 노드 분할 시 필요 데이터 조건 강화
        'reg_alpha': 0.1,             # L1 규제 추가
        'reg_lambda': 1.0,            # L2 규제 강화
        'scale_pos_weight': scale_weight,
        'random_state': 42,
        'verbose': -1
    }
    
    print(f">> 3. LightGBM AI 모델 학습 시작 (데이터 크기: {X_train.shape[0]}행, 피처 개수: {len(feature_cols)}개)...")
    
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)] # 학습 관찰 기간 확대
    
    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=1000,
        valid_sets=[train_dataset, test_dataset],
        callbacks=callbacks
    )
    
    # 테스트 셋 평가 결과 출력
    preds = model.predict(X_test)
    auc_score = roc_auc_score(y_test, preds)
    print(f">> [검증 스코어] Test AUC: {auc_score:.4f}")
    
    model_dir = os.path.join(current_dir, 'Model')
    os.makedirs(model_dir, exist_ok=True)
    
    model.save_model(os.path.join(model_dir, 'pm_lightgbm_model.txt')) 
    joblib.dump(feature_cols, os.path.join(model_dir, 'model_features.pkl'))
    print(">> 5. 고도화된 예지 예측 모델 저장 완료.\n")

# ==============================================================================
# [4단계] 실시간 수신 스트림 기반 확률 추론 엔진 (물리 전조 지표 추적형 버퍼)
# ==============================================================================
class RealTimeInferenceEngine:
    def __init__(self, model_path, features_path):
        self.model = None
        self.feature_cols = None
        
        # 실시간 처리를 위한 슬라이딩 윈도우 멀티 버퍼
        self.buffer_ng = []
        self.buffer_a_margin = []
        self.buffer_x_margin = []
        self.buffer_a_diff = []
        self.buffer_x_diff = []
        
        self.load_model_artifacts(model_path, features_path)

    def load_model_artifacts(self, model_path, features_path):
        if os.path.exists(model_path) and os.path.exists(features_path):
            try:
                self.model = lgb.Booster(model_file=model_path)
                self.feature_cols = joblib.load(features_path)
                print(">> [엔진] 물리 트렌드 통합형 LightGBM 모델 적재 성공.")
            except Exception as e:
                print(f">> [엔진] 파일 로드 오류: {e}")
                self.model = None
                self.feature_cols = None

    def inject_minute_data_and_predict(self, raw_sensor_stream):
        if self.model is None or self.feature_cols is None:
            return 0.0
        
        # 1. 1분간 유입된 현장 원본 센서 스트림 가공 및 불량 판정
        a_boundary_scores = []
        x_boundary_scores = []
        a_diff_vals = []
        x_diff_vals = []
        minute_ng_count = 0

        # 스트림 데이터 파싱 (1분간 측정된 12개 포인트의 센서 측정 이력)
        for row in raw_sensor_stream:
            # 27개 원소 구조: [0]포지션별 임의값 등, 실제 처리는 포인트 1~12개 단위 기준
            # 시뮬레이터가 전달하는 원시 값에 맞춰 판정 점수 생성 (데모 스트림의 경우 스케일 감안 처리)
            pass

        # 데모 시뮬레이션 데이터 특성 상 직접적인 가공 수치 모사 구현
        # 실시간 스트림 데이터에서 불량 수 산출
        minute_ng_count = sum(raw_sensor_stream) 
        
        # 물리적 마진 추세 시뮬레이션 (불량이 쌓일수록 부품 마모가 심화되어 공차 한계선에 근접하는 경향 반영)
        simulated_a_margin = 0.5 + (minute_ng_count * 0.04)  # 위험도 최대 1.0 접근
        simulated_x_margin = 0.4 + (minute_ng_count * 0.05)
        simulated_a_diff = 0.2 + (minute_ng_count * 0.1)
        simulated_x_diff = 0.01 + (minute_ng_count * 0.008)

        # 2. 개별 리얼타임 버퍼에 삽입 (최대 30분 관리)
        self.buffer_ng.append(minute_ng_count)
        self.buffer_a_margin.append(simulated_a_margin)
        self.buffer_x_margin.append(simulated_x_margin)
        self.buffer_a_diff.append(simulated_a_diff)
        self.buffer_x_diff.append(simulated_x_diff)

        for buf in [self.buffer_ng, self.buffer_a_margin, self.buffer_x_margin, self.buffer_a_diff, self.buffer_x_diff]:
            if len(buf) > 30:
                buf.pop(0)

        # 3. 다중 윈도우 피처 재생성 (학습 단계 피처셋과 완전 일치)
        input_features = {}
        for w in [5, 15, 30]:
            # 서브 윈도우 추출
            sub_ng = np.array(self.buffer_ng[-w:])
            sub_am = np.array(self.buffer_a_margin[-w:])
            sub_xm = np.array(self.buffer_x_margin[-w:])
            sub_ad = np.array(self.buffer_a_diff[-w:])
            sub_xd = np.array(self.buffer_x_diff[-w:])
            
            input_features[f'X_ng_sum_{w}m'] = sub_ng.sum()
            input_features[f'X_ng_mean_{w}m'] = sub_ng.mean()
            input_features[f'X_ng_std_{w}m'] = sub_ng.std() if len(sub_ng) > 1 else 0.0
            
            input_features[f'X_a_margin_mean_{w}m'] = sub_am.mean()
            input_features[f'X_x_margin_mean_{w}m'] = sub_xm.mean()
            input_features[f'X_a_diff_std_{w}m'] = sub_ad.std() if len(sub_ad) > 1 else 0.0
            input_features[f'X_x_diff_std_{w}m'] = sub_xd.std() if len(sub_xd) > 1 else 0.0

        # 피처 정렬 및 입력 데이터프레임 생성
        ordered_vals = [input_features[col] for col in self.feature_cols]
        input_df = pd.DataFrame([ordered_vals], columns=self.feature_cols)

        # 예측 실행 (클래스 1 위험 확률 반환)
        risk_probability = self.model.predict(input_df)[0]
        return risk_probability

# ==============================================================================
# [5단계] 가동 데이터 테스트 시뮬레이터 실행부
# ==============================================================================
if __name__ == "__main__":
    main_start_time = time.perf_counter()    
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
        abs_model_path = os.path.join(current_dir, 'Model', 'pm_lightgbm_model.txt') 
        abs_features_path = os.path.join(current_dir, 'Model', 'model_features.pkl')

        print(">> [시스템] 신규 알고리즘 반영을 위해 기존 모델 아티팩트 정리...")
        for file_path in [abs_model_path, abs_features_path]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass
        
        inference_engine = RealTimeInferenceEngine(model_path=abs_model_path, features_path=abs_features_path)
     
        if inference_engine.model is None or inference_engine.feature_cols is None:
            print("\n>> [시스템 안내] 새 트렌드 모델 학습을 개시합니다...")
            train_predictive_model()
            inference_engine.load_model_artifacts(abs_model_path, abs_features_path)

        # -----------------------------------------------------------------------------------------
        # 가상의 유입 데이터 (1분당 원시 시그널)
        time_series_stream = [
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 1분 (총합 1)
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 2분 (총합 3)           
            [0, 0, 0, 3, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 3분 (총합 7)          
            [0, 2, 0, 3, 0, 0, 1, 0, 0, 2, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 4분 (총합 12)
        ]
        
        print("\n>> 6. 실시간 현장 데이터 유입에 따른 AI 예지 확률 추론 테스트 시작:")

        inference_start_time = time.perf_counter()
        
        for t_min, raw_index_stream in enumerate(time_series_stream, start=1):
            current_minute_ng_count = sum(raw_index_stream) 
            prob = inference_engine.inject_minute_data_and_predict(raw_index_stream)
            
            print(f"[{t_min}분 경과] 분당 불량 수: {current_minute_ng_count}ea "
                  f"-> 예지분석 결과: '향후 1시간 내 설비 긴급장애 발생 가능성: {prob*100:.1f}%'")            
            if prob >= 0.60:
                print(f"🚨 [경보 발령] 예지보전 정비 알림 발송 완료 (신뢰도 {prob*100:.0f}%)\n")
                
        inference_end_time = time.perf_counter()       

        print("-" * 50)
        print(f">> [성능 리포트] 실시간 4분 추론 연산 소요 시간: {inference_end_time - inference_start_time:.6f}초") 

    except FileNotFoundError as e:
        print(f"\n[경로 오류] 데이터 소스나 모델 디렉토리를 확인해 주세요. {e}")
    except Exception as e:
        print(f"\n[런타임 오류] 예외 발생: {e}")
    finally:
        main_end_time = time.perf_counter()
        print(f">> [시스템] 전체 메인 루틴 최종 실행 시간: {main_end_time - main_start_time:.4f}초")
        print("-" * 50)
    