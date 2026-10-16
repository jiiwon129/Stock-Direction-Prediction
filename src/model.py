"""최종 v3 모델: daily/price 피처 7개 + 데이터 가용성 확인용 ret1.

피처 생성: build_features(day)
예측: Model.predict(day), predict_features(bundle, features)
채점기 진입점: load_model()
원래 모델의 가중치와 예측 수식을 유지하고 사용하지 않는 실험 코드를 제거했다.
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    'pre_gap', 'post_gap', 'rv20', 'intra_vol20',
    'error_vol20', 'error_down60', 'error_up60',
]
GUARD_COLUMN = 'ret1'
ALL_COLUMNS = FEATURE_COLUMNS + [GUARD_COLUMN]
LOOKBACK = 120
CUTS = np.array([-.025, -.006, .006, .025])
MOMENTS = np.c_[(np.arange(5)-2)**3, (np.arange(5)-2)**2]


def build_features(day):
    """known_at < 당일 09:30인 정보로 7개 피처와 ret1만 계산한다.

    day.y는 읽지 않는다. 과거의 장전 이후 수익률 통계에도 당일 종가는 제외한다.
    day.symbols와 거래일은 교수님이 제공하는 예측 대상 메타데이터다.
    """
    symbols = list(day.symbols)
    daily = day.daily(days=LOOKBACK, columns=[
        'symbol','date_et','open','close','known_at'])
    daily = daily[(daily.known_at < day.cutoff) & (daily.date_et < day.target)]
    daily = daily[daily.symbol.isin(symbols)].sort_values(['symbol','date_et'])
    price = day.price(days=LOOKBACK, columns=[
        'symbol','datetime','close','session','known_at'])
    price = price[(price.known_at < day.cutoff) & price.symbol.isin(symbols)].copy()
    price['date_et'] = price.datetime.dt.normalize()

    # 각 과거 거래일에도 09:30에 완성되는 봉은 사용할 수 없다.
    pre = price[(price.session == 'pre') &
                (price.known_at < price.date_et + pd.Timedelta(hours=9,minutes=30))]
    pre = pre.sort_values('datetime').groupby(['symbol','date_et']).tail(1)
    historical_pre = pre[['symbol','date_et','close']].rename(columns={'close':'historic_pre'})
    history = daily.merge(historical_pre,on=['symbol','date_et'],how='left',validate='one_to_one')
    current_pre = pre[pre.date_et == day.target].set_index('symbol').close
    post = price[(price.session == 'post') & (price.date_et == day.date)]
    current_post = post.sort_values('datetime').groupby('symbol').tail(1).set_index('symbol').close

    rows = []
    for symbol, g in history.groupby('symbol'):
        if len(g) < 30:
            continue
        close, opening = g.close.to_numpy(dtype=float), g.open.to_numpy(dtype=float)
        returns = close[1:] / close[:-1] - 1
        pre_price = g.historic_pre.fillna(g.open).to_numpy(dtype=float)
        # error는 오분류율이 아니라 과거의 장전 이후 남은 수익률이다.
        residual = np.clip(close / pre_price - 1, -.5, .5)
        rows.append(dict(
            symbol=symbol,
            pre_gap=current_pre.get(symbol,np.nan) / close[-1] - 1,
            post_gap=current_post.get(symbol,np.nan) / close[-1] - 1,
            rv20=np.std(returns[-20:],ddof=1),
            intra_vol20=np.std((close/opening-1)[-20:],ddof=1),
            error_vol20=np.std(residual[-20:],ddof=1),
            error_down60=np.sqrt(2*np.mean(np.minimum(residual[-60:],0)**2)),
            error_up60=np.sqrt(2*np.mean(np.maximum(residual[-60:],0)**2)),
            ret1=returns[-1],
        ))
    computed = pd.DataFrame(rows,columns=['symbol',*ALL_COLUMNS])
    return (pd.DataFrame({'symbol':symbols}).merge(computed,on='symbol',how='left',validate='one_to_one')
            .replace([np.inf,-np.inf],np.nan))


def gap_inputs(features, scale_column):
    gap = features.pre_gap.fillna(features.post_gap).fillna(0).to_numpy(dtype=float)
    scale = features[scale_column].fillna(features.rv20).fillna(.02).to_numpy(dtype=float)
    return np.clip(gap,-.95,2.), np.clip(scale,.003,.20)


def scales(features, config):
    gap, default = gap_inputs(features,config.get('scale','error_vol20'))
    if config.get('two_sided'):
        down = np.clip(features.error_down60.fillna(pd.Series(default,index=features.index)).to_numpy(),.003,.20)
        up = np.clip(features.error_up60.fillna(pd.Series(default,index=features.index)).to_numpy(),.003,.20)
    else:
        down = up = default
    return gap, down, up


def ecdf_probability(component, features):
    """과거 표준화 잔차의 경험분포에서 5개 클래스 확률을 계산한다."""
    gap, scale = gap_inputs(features,component['scale'])
    threshold = ((1+CUTS[None,:])/(1+gap[:,None])-1)/scale[:,None]
    residuals = component['residuals']
    cdf = np.searchsorted(residuals,threshold,side='right')/len(residuals)
    return np.diff(np.c_[np.zeros(len(features)),cdf,np.ones(len(features))],axis=1)


def component_moments(component, features):
    if component['kind'] == 'legacy_blend':
        probability = np.mean([ecdf_probability(c,features) for c in component['components']],axis=0)
        return probability @ MOMENTS
    if component['kind'] != 'distribution':
        raise ValueError('이 정리본은 최종 v3 분포 모델만 지원합니다.')
    gap, down, up = scales(features,component['config'])
    threshold = (1+CUTS)/(1+gap[:,None])-1
    z = threshold/np.where(threshold<0,down[:,None],up[:,None])
    residuals = component['residuals']
    cdf = np.searchsorted(residuals,z,side='right')/len(residuals)
    probability = np.diff(np.c_[np.zeros(len(features)),cdf,np.ones(len(features))],axis=1)
    return probability @ MOMENTS


def raw_prediction(bundle, features):
    return sum(w*component_moments(c,features) for w,c in zip(bundle['weights'],bundle['components']))


def decode(raw, rule):
    """공식 점수의 비대칭 오분류 비용과 학습 구간의 사전분포를 반영한다."""
    action = np.arange(-2,3)
    B,C = raw[:,0,None],raw[:,1,None]
    prior_B,prior_C = rule['prior_moments']
    costs = -2*action*B + action**2*C - rule['alpha']*(-2*action*prior_B + action**2*prior_C)
    return costs.argmin(axis=1).astype(int)


def predict_features(bundle, features):
    labels = decode(raw_prediction(bundle,features),bundle['rule'])
    labels[features.ret1.isna().to_numpy()] = 2
    return labels


class Model:
    def __init__(self,bundle):
        if bundle.get('format_version') != 3:
            raise ValueError('v3 가중치 파일이 필요합니다.')
        self.bundle = bundle

    def predict(self,day):
        features = build_features(day)
        return pd.DataFrame({'symbol':features.symbol,'label':predict_features(self.bundle,features)})


def load_model():
    """교수님 채점기가 호출하는 함수. 재학습 없이 제출용 가중치만 읽는다."""
    path = Path(__file__).resolve().parents[1]/'artifacts'/'feature_model.pkl'
    with path.open('rb') as stream:
        return Model(pickle.load(stream))
