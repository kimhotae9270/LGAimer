# -*- coding: utf-8 -*-
import os, gc, time, warnings, re
from glob import glob
from datetime import timedelta
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor, early_stopping, log_evaluation
import xgboost as xgb

warnings.filterwarnings("ignore")

# ===================== Paths =====================
TRAIN_PATH = "../train/train.csv"
TEST_GLOB = "../test/TEST_*.csv"
SAMPLE_SUB_PATH = "../sample_submission.csv"
OUTPUT_PATH = "../result/submission_ensemble_nbeatsGEN_emb_zipGLOBAL_sarimaxLOCAL_nogroup.csv"

# ===================== Core config =====================
HORIZONS = [1, 2, 3, 4, 5, 6, 7]
LAG_LIST = [1, 7, 14, 28]
ROLLS_MEAN = [7, 14, 28]
ROLLS_ZERO = [7, 28]

USE_TIME_FEATURES = False          # 요일/월/주말 피처 비활성
EXCLUDE_TIME_IN_LGBM = True        # LGBM 입력에서 시간피처 제거
USE_GROUP_FEATURES = True         # 완전 비활성: 그룹 피처 생성/사용 안 함

# ===== Ensemble weights (기본 합=1.00) =====
W_XGB     = 0.30
W_LGBM    = 0.30
W_NBEATS  = 0.30
W_SARIMAX = 0.05   # SARIMAX는 튜닝 제외 (고정)
W_ZIP     = 0  # ZIP은 글로벌 & 튜닝 포함

RANDOM_STATE = 2025
EARLY_STOPPING_ROUNDS = 500

# ===================== Tree model params =====================
LGBM_PARAMS = dict(
    objective="regression",
    learning_rate=0.035,
    n_estimators=10000,        # 🔼 3000 → 10000 (더 길게 학습)
    num_leaves=127,
    feature_fraction=0.80,
    bagging_fraction=0.80,
    bagging_freq=1,
    min_data_in_leaf=32,
    max_depth=-1,
    max_bin=255,
    device="gpu",
    gpu_platform_id=0,
    gpu_device_id=0,
    random_state=RANDOM_STATE
)

XGB_TRAIN_PARAMS = {
    "objective": "reg:squarederror",
    "eta": 0.03,
    "max_depth": 8,
    "min_child_weight": 2,
    "subsample": 0.8,
    "colsample_bytree": 0.75,
    "alpha": 0.05,
    "lambda": 0.90,
    "tree_method": "gpu_hist",
    "predictor": "gpu_predictor",
    "eval_metric": "rmse",
    "seed": RANDOM_STATE,
}

# ===================== Group features (disabled) =====================
CLUSTER_MEMBERS_GLOB = ""  # 비활성 (필요시 경로 지정)
GLOBAL_GROUP_MAP: Dict[str, str] = {}
GLOBAL_GROUP_LETTERS: List[str] = []

def _load_cluster_groups(pattern: str = CLUSTER_MEMBERS_GLOB):
    # 유지하되, USE_GROUP_FEATURES=False 이면 그냥 빈 맵 반환
    if not USE_GROUP_FEATURES:
        return {}, []
    group_map = {}
    letters = []
    for p in sorted(glob(pattern)):
        bn = os.path.basename(p)
        m = re.search(r"final_members_([A-Za-z]+)\.csv$", bn)
        if not m:
            continue
        letter = m.group(1)
        try:
            dfm = pd.read_csv(p)
            col = "영업장명_메뉴명" if "영업장명_메뉴명" in dfm.columns else dfm.columns[0]
            for it in dfm[col].astype(str).tolist():
                group_map[it] = letter
            letters.append(letter)
        except Exception:
            pass
    return group_map, sorted(set(letters))

# ===================== Feature utils =====================
def _time_feats(df: pd.DataFrame) -> pd.DataFrame:
    if not USE_TIME_FEATURES: return df
    df = df.copy()
    dt = df["영업일자"]
    df["dow"] = dt.dt.weekday
    df["month"] = dt.dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(np.int8)
    return df

def _zero_streak_from_past(s: pd.Series) -> pd.Series:
    out = np.zeros(len(s), dtype=np.float32)
    streak = 0
    vals = s.to_numpy()
    for i in range(len(vals)):
        prev = vals[i-1] if i-1 >= 0 else np.nan
        if i == 0 or np.isnan(prev) or prev != 0:
            streak = 0
        else:
            streak += 1
        out[i] = streak
    return pd.Series(out, index=s.index)

def _add_train_feats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["영업장명_메뉴명", "영업일자"])
    df = _time_feats(df)

    for k in [1,7,14,28]:
        df[f"lag_{k}"] = df.groupby("영업장명_메뉴명")["매출수량"].shift(k)

    s_shift = df.groupby("영업장명_메뉴명")["매출수량"].shift(1)

    for w in [7,14,28]:
        df[f"rmean_{w}"] = s_shift.rolling(w, min_periods=1).mean()

    for w in [7,28]:
        z = (s_shift == 0).astype(float)
        df[f"zero_count_{w}"] = z.rolling(w, min_periods=1).sum()

    df["zero_streak"] = (
        df.groupby("영업장명_메뉴명")["매출수량"]
          .apply(_zero_streak_from_past)
          .reset_index(level=0, drop=True)
          .astype(float)
    )

    # 그룹 피처 완전 비활성: 아무 것도 생성하지 않음

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].astype(np.float32)
    return df

def build_datasets_by_horizon(train: pd.DataFrame):
    base = _add_train_feats(train)
    datasets = {}
    for h in [1,2,3,4,5,6,7]:
        use = base.copy()
        use[f"target_h{h}"] = use.groupby("영업장명_메뉴명")["매출수량"].shift(-h)
        safe_lags = [f"lag_{k}" for k in [1,7,14,28] if k >= h]

        feat_cols = []
        if USE_TIME_FEATURES:
            feat_cols += ["dow", "month", "is_weekend"]
        feat_cols += safe_lags + [f"rmean_{w}" for w in [7,14,28]] + \
                     [f"zero_count_{w}" for w in [7,28]] + ["zero_streak"] + \
                     ["영업장명_메뉴명"]

        # 그룹 피처 비활성: 추가하지 않음

        use = use.dropna(subset=safe_lags + [f"target_h{h}"]).copy()

        use["영업장명_메뉴명"] = use["영업장명_메뉴명"].astype("category")
        cat_categories = use["영업장명_메뉴명"].cat.categories
        use["영업장명_메뉴명_code"] = use["영업장명_메뉴명"].cat.codes.astype("int32")

        X = use[feat_cols + ["영업장명_메뉴명_code"]].copy()
        y = use[f"target_h{h}"].astype(np.float32).copy()
        dates = use["영업일자"].copy()

        num_cols = X.select_dtypes(exclude=["category","object"]).columns
        X[num_cols] = X[num_cols].fillna(0.0).astype(np.float32)

        datasets[h] = (X, y, feat_cols, dates, cat_categories)
    return datasets

def _time_split_train_valid(dates: pd.Series, valid_ratio=0.10):
    dates_sorted = np.array(sorted(dates.unique()))
    split_idx = int(len(dates_sorted) * (1 - valid_ratio))
    split_idx = max(1, min(split_idx, len(dates_sorted) - 1))
    split_date = dates_sorted[split_idx]
    train_mask = dates <= split_date
    valid_mask = dates > split_date
    if valid_mask.sum() == 0:
        valid_mask = dates == split_date
        train_mask = dates < split_date
    return train_mask, valid_mask

# ===================== N-BEATS (Generic-only + Item Embedding) =====================
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

BACKCAST_LENGTH = 28
FORECAST_LENGTH = 7
NBEATS_WIDTH = 1024
NBEATS_DEPTH = 6
NBEATS_GENERIC_BLOCKS = 10
NBEATS_DROPOUT = 0.1
NBEATS_LR = 1e-3
NBEATS_WD = 1e-4
NBEATS_BATCH = 4096
NBEATS_MAX_EPOCHS = 500
NBEATS_MIN_EPOCHS = 60
NBEATS_PATIENCE = 35
NBEATS_USE_SMAPE_LOSS = True
NBEATS_EMB_DIM = 32
NBEATS_USE_ROBUST_SCALE = True  # True -> scale by P90(|backcast|), else max-abs
SMAPE_EPS = 1e-3

# torch.compile 옵션: 안전하게 기본 꺼짐
USE_TORCH_COMPILE = False  # Set True only if Triton/Inductor available

class WindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, backcast: int, forecast: int,
                 valid: bool, valid_ratio: float, item_to_id: Dict[str,int]):
        self.backcasts, self.forecasts, self.scales, self.item_ids = [], [], [], []
        for item, g in df.groupby("영업장명_메뉴명"):
            y = g.sort_values("영업일자")["매출수량"].to_numpy(dtype=np.float32)
            n = len(y); L, H = backcast, forecast
            if n < L + H: continue
            pairs = []
            for t in range(0, n - (L + H) + 1):
                b = y[t:t+L]; f = y[t+L:t+L+H]
                pairs.append((b, f))
            if not pairs: continue
            split = max(1, int(len(pairs) * (1 - valid_ratio)))
            use_pairs = pairs[split:] if valid else pairs[:split]
            iid = item_to_id[str(item)]
            for b, f in use_pairs:
                if NBEATS_USE_ROBUST_SCALE:
                    s = np.percentile(np.abs(b), 90)
                else:
                    s = np.max(np.abs(b))
                s = 1.0 if s < 1e-6 else s
                self.backcasts.append((b/s).astype(np.float32))
                self.forecasts.append((f/s).astype(np.float32))
                self.scales.append(s)
                self.item_ids.append(iid)
        self.backcasts = torch.tensor(np.stack(self.backcasts, axis=0))
        self.forecasts = torch.tensor(np.stack(self.forecasts, axis=0))
        self.scales = torch.tensor(np.asarray(self.scales, dtype=np.float32))
        self.item_ids = torch.tensor(np.asarray(self.item_ids, dtype=np.int64))
    def __len__(self): return self.backcasts.shape[0]
    def __getitem__(self, idx): return self.backcasts[idx], self.forecasts[idx], self.scales[idx], self.item_ids[idx]

class FCStack(nn.Module):
    def __init__(self, in_dim, width, depth, p=0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            if p > 0: layers += [nn.Dropout(p)]
            d = width
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class GenericBlock(nn.Module):
    def __init__(self, in_dim, backcast_len, forecast_len, width, depth, p=0.0):
        super().__init__()
        self.fc = FCStack(in_dim, width, depth, p=p)
        self.backcast_proj = nn.Linear(width, backcast_len)
        self.forecast_proj = nn.Linear(width, forecast_len)
    def forward(self, x):
        h = self.fc(x)
        return self.backcast_proj(h), self.forecast_proj(h)

class NBeatsGEN(nn.Module):
    def __init__(self, backcast_len, forecast_len, n_items, emb_dim=32,
                 width=512, depth=4, blocks=8, p=0.0, device="cpu"):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, emb_dim)
        in_dim = backcast_len + emb_dim
        self.blocks = nn.ModuleList([GenericBlock(in_dim, backcast_len, forecast_len, width, depth, p=p)
                                     for _ in range(blocks)])
        self.device_ref = device
    def forward(self, bcast, item_ids):
        emb = self.item_emb(item_ids)               # (B, E)
        x = torch.cat([bcast, emb], dim=1)          # (B, L+E)
        residual = bcast
        forecast_sum = 0.0
        for blk in self.blocks:
            backcast, forecast = blk(x)
            residual = residual - backcast
            x = torch.cat([residual, emb], dim=1)
            forecast_sum = forecast_sum + forecast
        return residual, forecast_sum

def smape_loss(yhat, ytrue, eps=SMAPE_EPS):
    num = torch.abs(ytrue - yhat)
    denom = torch.abs(ytrue) + torch.abs(yhat) + eps
    return torch.mean(2.0 * num / denom)

def _train_nbeats_global(train_df: pd.DataFrame, device: torch.device):
    # Item id mapping
    items = sorted(train_df["영업장명_메뉴명"].astype(str).unique().tolist())
    item_to_id = {it:i for i,it in enumerate(items)}
    n_items = len(items)

    tr_ds = WindowDataset(train_df, BACKCAST_LENGTH, FORECAST_LENGTH, valid=False, valid_ratio=0.20, item_to_id=item_to_id)
    va_ds = WindowDataset(train_df, BACKCAST_LENGTH, FORECAST_LENGTH, valid=True,  valid_ratio=0.20, item_to_id=item_to_id)
    print(f"[NBEATS-GEN] train_windows={len(tr_ds)} valid_windows={len(va_ds)} items={n_items}")

    model = NBeatsGEN(BACKCAST_LENGTH, FORECAST_LENGTH, n_items=n_items, emb_dim=NBEATS_EMB_DIM,
                      width=NBEATS_WIDTH, depth=NBEATS_DEPTH, blocks=NBEATS_GENERIC_BLOCKS,
                      p=NBEATS_DROPOUT, device=device).to(device)
    model.item_to_id = item_to_id  # 저장해서 추론 때 사용

    opt = torch.optim.AdamW(model.parameters(), lr=NBEATS_LR, weight_decay=NBEATS_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NBEATS_MAX_EPOCHS, eta_min=1e-5)

    loss_fn = smape_loss if NBEATS_USE_SMAPE_LOSS else nn.SmoothL1Loss()

    tr_loader = DataLoader(tr_ds, batch_size=NBEATS_BATCH, shuffle=True, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=NBEATS_BATCH, shuffle=False, drop_last=False)

    scaler = GradScaler(enabled=(device.type == "cuda"))
    best_state, best_val, bad = None, float("inf"), 0
    t0 = time.perf_counter()

    # (옵션) torch.compile: only when Triton (Inductor) is available on CUDA
    if USE_TORCH_COMPILE and device.type == 'cuda':
        try:
            import triton  # noqa: F401
            model = torch.compile(model, backend='inductor')
            print('[NBEATS-GEN] torch.compile enabled (inductor)')
        except Exception as e:
            print(f'[NBEATS-GEN] torch.compile disabled: {e}')

    for epoch in range(1, NBEATS_MAX_EPOCHS + 1):
        model.train(); tr_loss_sum = n_tr = 0
        for xb, yb, _s, iid in tr_loader:
            xb = xb.to(device); yb = yb.to(device); iid = iid.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                _, yhat = model(xb, iid)
                loss = loss_fn(yhat, yb)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            tr_loss_sum += loss.item() * xb.size(0); n_tr += xb.size(0)

        model.eval(); va_loss_sum = n_va = 0
        with torch.no_grad():
            for xb, yb, _s, iid in va_loader:
                xb = xb.to(device); yb = yb.to(device); iid = iid.to(device)
                with autocast(enabled=(device.type == "cuda")):
                    _, yhat = model(xb, iid)
                    val_loss = loss_fn(yhat, yb)
                va_loss_sum += val_loss.item() * xb.size(0); n_va += xb.size(0)
        va_loss = va_loss_sum / max(1, n_va)
        scheduler.step()

        if va_loss < best_val - 1e-7:
            best_val = va_loss; bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[NBEATS-GEN best@{epoch}] val={best_val:.6f}, lr={opt.param_groups[0]['lr']:.2e}")
        else:
            bad += 1
            if epoch % 5 == 0:
                print(f"[NBEATS-GEN plateau] epoch={epoch}, bad={bad}, lr={opt.param_groups[0]['lr']:.2e}, val={va_loss:.6f}")
        if (epoch >= NBEATS_MIN_EPOCHS) and (bad >= NBEATS_PATIENCE):
            break

    print(f"[NBEATS-GEN] done in {time.perf_counter()-t0:.1f}s best_val={best_val:.6f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

@torch.no_grad()
def _predict_nbeats_7(model, last_values: np.ndarray, device: torch.device, item_name: str) -> np.ndarray:
    b = np.asarray(last_values, dtype=np.float32)
    if len(b) < BACKCAST_LENGTH:
        pad = np.zeros(BACKCAST_LENGTH - len(b), dtype=np.float32)
        b = np.concatenate([pad, b], axis=0)
    else:
        b = b[-BACKCAST_LENGTH:]
    if NBEATS_USE_ROBUST_SCALE:
        s = np.percentile(np.abs(b), 90)
    else:
        s = np.max(np.abs(b))
    s = 1.0 if s < 1e-6 else s
    xb = torch.from_numpy((b/s)[None, :]).to(device)
    iid = model.item_to_id.get(str(item_name), 0)
    iid = torch.tensor([iid], dtype=torch.long, device=device)
    with autocast(enabled=(device.type == "cuda")):
        _, yhat = model(xb, iid)
    y = yhat.squeeze(0).cpu().numpy() * s
    return np.clip(y, 0.0, None)

# ===================== Tree models (per-horizon) =====================
def train_models_by_horizon_all(datasets: dict):
    models = {}
    cat_ref = None
    for _, (_, _, _, _, cats) in datasets.items():
        cat_ref = cats
        break

    for h, (X, y, feat_cols, dates, _) in datasets.items():
        print(f"\n[+] Start Horizon +{h}d: n={len(X)}")
        X = X.copy()
        X["영업장명_메뉴명"] = X["영업장명_메뉴명"].astype("category").cat.set_categories(cat_ref)

        tr_mask, va_mask = _time_split_train_valid(dates, valid_ratio=0.10)
        X_tr, y_tr = X.loc[tr_mask], y.loc[tr_mask]
        X_va, y_va = X.loc[va_mask], y.loc[va_mask]

        feat_cols_lgbm = [c for c in feat_cols if c != "영업장명_메뉴명"] + ["영업장명_메뉴명_code"]
        if EXCLUDE_TIME_IN_LGBM:
            feat_cols_lgbm = [c for c in feat_cols_lgbm if c not in ("dow","month","is_weekend")]
        xgb_feats = feat_cols_lgbm[:]

        # LGBM
        reg_lgbm = LGBMRegressor(**LGBM_PARAMS)
        t0 = time.perf_counter()
        reg_lgbm.fit(
            X_tr[feat_cols_lgbm], y_tr,
            eval_set=[(X_va[feat_cols_lgbm], y_va)],
            eval_metric="rmse",
            callbacks=[early_stopping(EARLY_STOPPING_ROUNDS), log_evaluation(50)]
        )
        print(f"[H{h}] LGBM in {time.perf_counter()-t0:.1f}s (best_iter={getattr(reg_lgbm,'best_iteration_',None)})")

        # XGB
        dtrain = xgb.DMatrix(X_tr[xgb_feats], label=y_tr)
        dval   = xgb.DMatrix(X_va[xgb_feats], label=y_va)
        watchlist = [(dtrain, "train"), (dval, "eval")]
        t2 = time.perf_counter()
        reg_xgb = xgb.train(
            params=XGB_TRAIN_PARAMS,
            dtrain=dtrain,
            num_boost_round=10000,
            evals=watchlist,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=50
        )
        print(f"[H{h}] XGB in {time.perf_counter()-t2:.1f}s (best_iter={reg_xgb.best_iteration})")

        models[h] = dict(
            lgbm=(reg_lgbm, feat_cols_lgbm),
            xgb=(reg_xgb, xgb_feats),
            valid=(X_va.reset_index(drop=True), y_va.reset_index(drop=True))
        )

        del X_tr, y_tr, X_va, y_va, dtrain, dval
        gc.collect()

    return models

# ===================== SARIMAX (로컬, 비계절) & ZIP (글로벌, intercept-only) =====================
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", FutureWarning)

def _fit_sarimax_best_aic_noseason(y: np.ndarray):
    ORDERS = [(1,1,1), (2,1,1), (1,0,1), (0,1,1), (1,1,0)]
    best = None
    for o in ORDERS:
        try:
            res = SARIMAX(y, order=o, seasonal_order=(0,0,0,0),
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            aic = float(getattr(res, "aic", np.inf))
            if not np.isfinite(aic): continue
            if (best is None) or (aic < best[-1]): best = (res, o, (0,0,0,0), aic)
        except Exception:
            pass
    if best is None:
        res = SARIMAX(y, order=(1,1,0), seasonal_order=(0,0,0,0),
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        return res, (1,1,0), (0,0,0,0), float(res.aic)
    return best

def _aic_select_items_sarimax_noseason(train: pd.DataFrame, items: List[str]) -> Tuple[Dict[str, Tuple[tuple,tuple]], List[str]]:
    rows = []
    order_by_item: Dict[str, Tuple[tuple,tuple]] = {}
    for it in items:
        g = train[train["영업장명_메뉴명"] == it].sort_values("영업일자")
        y = g["매출수량"].to_numpy(dtype=float)
        if len(y) < 30:
            continue
        y = np.maximum(y, 0.0)
        try:
            res, o, so, aic = _fit_sarimax_best_aic_noseason(y)
            order_by_item[it] = (o, so)
            rows.append((it, aic))
        except Exception:
            pass
    if not rows:
        return {}, []
    df = pd.DataFrame(rows, columns=["item","aic"]).sort_values("aic")
    cutoff = np.nanpercentile(df["aic"].values, 100.0)
    chosen = df.loc[df["aic"] <= cutoff, "item"].astype(str).tolist()
    return order_by_item, chosen

def _forecast_sarimax_for_block(train: pd.DataFrame, block_df: pd.DataFrame,
                                item: str, order: tuple, sorder: tuple, horizon: int = 7) -> np.ndarray:
    g_tr = train[train["영업장명_메뉴명"] == item].sort_values("영업일자")
    y_tr = g_tr["매출수량"].to_numpy(dtype=float)
    g_te = block_df[block_df["영업장명_메뉴명"] == item].sort_values("영업일자")
    y_all = np.concatenate([y_tr, g_te["매출수량"].to_numpy(dtype=float)], axis=0) if len(g_te) else y_tr
    y_all = np.maximum(y_all, 0.0)
    try:
        res = SARIMAX(y_all, order=order, seasonal_order=sorder,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = np.asarray(res.forecast(steps=horizon), dtype=np.float32)
        return np.clip(fc, 0.0, None)
    except Exception:
        w = min(28, len(y_all))
        mu = float(np.nanmean(y_all[-w:])) if w > 0 else 0.0
        return np.repeat(mu, horizon).astype(np.float32)

# ----- ZIP (global, intercept-only) -----
try:
    from statsmodels.discrete.count_model import ZeroInflatedPoisson
    HAVE_ZIP = True
except Exception:
    HAVE_ZIP = False
    warnings.warn("[ZIP] statsmodels ZeroInflatedPoisson unavailable. ZIP will be skipped.")

def fit_zip_global_intercept_only(train: pd.DataFrame) -> Optional[object]:
    if not HAVE_ZIP:
        return None
    y = train.sort_values(["영업일자","영업장명_메뉴명"])["매출수량"].to_numpy(dtype=float)
    y = np.maximum(y, 0.0)
    exog = np.ones((len(y), 1), dtype=float)
    try:
        model = ZeroInflatedPoisson(y, exog=exog, exog_infl=exog, inflation="logit")
        res = model.fit(disp=False, maxiter=300)
        return res
    except Exception as e:
        warnings.warn(f"[ZIP] global fit failed: {e}")
        return None

def zip_global_expected_mean(res_zip) -> Optional[float]:
    if res_zip is None:
        return None
    try:
        mu = float(res_zip.predict(exog=np.ones((1,1)), exog_infl=np.ones((1,1)))[0])
        return max(0.0, mu)
    except Exception:
        return None

# ===================== Utils: sMAPE & weight tuning =====================
def _smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = (np.abs(y_true) + np.abs(y_pred) + eps)
    diff = np.abs(y_true - y_pred)
    return float(np.mean(2.0 * diff / denom))

def _build_nbeats_val_preds_for_h(
    nbeats_model,
    device: torch.device,
    train_df: pd.DataFrame,
    X_va: pd.DataFrame,
    dates_va: pd.Series,
    items_va: pd.Series,
    h: int
) -> np.ndarray:
    preds = np.zeros(len(X_va), dtype=np.float32)
    series_by_item = {}
    for item, g in train_df.groupby("영업장명_메뉴명"):
        series_by_item[str(item)] = g.sort_values("영업일자")[["영업일자","매출수량"]].reset_index(drop=True)

    for i in range(len(X_va)):
        item = str(items_va.iloc[i])
        t = pd.to_datetime(dates_va.iloc[i])
        g = series_by_item.get(item, None)
        if g is None:
            preds[i] = 0.0
            continue
        hist = g[g["영업일자"] < t]["매출수량"].to_numpy(dtype=np.float32)
        y7 = _predict_nbeats_7(nbeats_model, hist, device, item)
        preds[i] = float(y7[h-1]) if len(y7) >= h else float(y7[-1])
    return preds

ZIP_WEIGHT_FLOOR = 0.0  # ZIP 가중치 하한 (0.0이면 자유 튜닝)

def auto_tune_weights_scipy(models_by_h: Dict, datasets: Dict, nbeats_model,
                            zip_mu: Optional[float], train_df: pd.DataFrame, device: torch.device):
    try:
        from scipy.optimize import minimize
    except Exception as e:
        print(f"[AUTO WEIGHTS] scipy 사용 불가: {e}. 기본 가중치 유지합니다.")
        return None

    preds_stack = []
    y_all = []

    for h in HORIZONS:
        mdl = models_by_h[h]
        X_va, y_va = mdl["valid"]
        _, _, _, dates, _ = datasets[h]
        _, va_mask = _time_split_train_valid(dates, valid_ratio=0.10)
        dates_va = dates.loc[va_mask].reset_index(drop=True)
        items_va = X_va["영업장명_메뉴명"].astype("category")

        reg_lgbm, feat_cols_lgbm = mdl["lgbm"]
        best_iter_lgbm = getattr(reg_lgbm, "best_iteration_", None)
        p_lgbm = reg_lgbm.predict(X_va[feat_cols_lgbm], num_iteration=best_iter_lgbm)

        reg_xgb, xgb_feats = mdl["xgb"]
        dval = xgb.DMatrix(X_va[xgb_feats])
        if getattr(reg_xgb, "best_iteration", None) is not None:
            ntree_limit = int(reg_xgb.best_iteration) + 1
            p_xgb = reg_xgb.predict(dval, iteration_range=(0, ntree_limit))
        else:
            p_xgb = reg_xgb.predict(dval)

        p_nbeats = _build_nbeats_val_preds_for_h(
            nbeats_model, device, train_df, X_va, dates_va, items_va, h
        )

        rows = [
            np.asarray(p_xgb, dtype=np.float64),
            np.asarray(p_lgbm, dtype=np.float64),
            np.asarray(p_nbeats, dtype=np.float64),
        ]

        if zip_mu is not None:
            rows.append(np.repeat(float(zip_mu), len(X_va)).astype(np.float64))

        P = np.vstack(rows)
        preds_stack.append(P)
        y_all.append(np.asarray(y_va, dtype=np.float64))

    P_full = np.hstack(preds_stack)
    y_full = np.concatenate(y_all)

    mask = np.isfinite(y_full)
    for i in range(P_full.shape[0]):
        mask &= np.isfinite(P_full[i])
    P_full = P_full[:, mask]
    y_full = y_full[mask]

    if y_full.size == 0 or P_full.shape[1] == 0:
        print("[AUTO WEIGHTS] 유효한 검증 예측이 없어 튜닝을 건너뜁니다.")
        return None

    M = P_full.shape[0]
    def objective(w):
        w = np.asarray(w, dtype=np.float64)
        y_pred = np.dot(w, P_full)
        return _smape(y_full, y_pred)

    x0 = np.ones(M, dtype=np.float64) / M
    cons = ({'type':'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    if M == 4:
        bnds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (ZIP_WEIGHT_FLOOR, 1.0)]
    else:
        bnds = [(0.0, 1.0)] * M

    res = minimize(objective, x0, method="SLSQP", bounds=bnds, constraints=cons, options={'maxiter': 500, 'ftol': 1e-9})
    if not res.success:
        print(f"[AUTO WEIGHTS] 최적화 수렴 실패: {res.message}. 기본 가중치 유지합니다.")
        return None

    w_opt = np.clip(res.x, 0.0, 1.0)
    w_opt = w_opt / (w_opt.sum() + 1e-12)

    result = {"xgb": w_opt[0], "lgbm": w_opt[1], "nbeats": w_opt[2]}
    if M == 4:
        result["zip"] = w_opt[3]

    print(f"[AUTO WEIGHTS] tuned order(XGB,LGBM,NBEATS,ZIP?) w={w_opt}  smape={objective(w_opt):.6f}")
    return result

# ===================== Inference (Weighted Ensemble) =====================
def _compute_infer_stats(hist: pd.DataFrame, items: np.ndarray):
    stats = {"rmean": {7: [], 14: [], 28: []}, "zero_count": {7: [], 28: []}, "zero_streak": []}
    for item in items:
        s = hist[hist["영업장명_메뉴명"] == item].sort_values("영업일자")["매출수량"].to_numpy()
        z_streak = 0
        for v in s[::-1]:
            if v == 0: z_streak += 1
            else: break
        stats["zero_streak"].append(float(z_streak))
        for w in [7,14,28]:
            win = s[-w:] if len(s) >= w else s
            stats["rmean"][w].append(float(np.mean(win)) if len(win) else np.nan)
        for w in [7,28]:
            win = s[-w:] if len(s) >= w else s
            stats["zero_count"][w].append(float((np.array(win) == 0).sum()) if len(win) else np.nan)
    return stats

def predict_block_ensemble(models_by_h, nbeats_model, train: pd.DataFrame,
                           sarimax_orders: Dict[str, Tuple[tuple,tuple]], chosen_sarimax: List[str],
                           zip_mu: Optional[float],
                           test_block_df: pd.DataFrame, items: np.ndarray, device: torch.device):
    hist = test_block_df.copy()
    hist["영업일자"] = pd.to_datetime(hist["영업일자"])
    hist["영업장명_메뉴명"] = hist["영업장명_메뉴명"].astype(str)
    last_date = hist["영업일자"].max()

    nbeats_by_item = {}
    for item in items:
        s = hist[hist["영업장명_메뉴명"] == item].sort_values("영업일자")["매출수량"].to_numpy(dtype=np.float32)
        nbeats_by_item[item] = _predict_nbeats_7(nbeats_model, s, device, item)

    sarimax_by_item = {}
    for item in items:
        if item in sarimax_orders and item in chosen_sarimax:
            o, so = sarimax_orders[item]
            sarimax_by_item[item] = _forecast_sarimax_for_block(train, hist, item, o, so, horizon=7)

    stats = _compute_infer_stats(hist, items)

    all_rows = []
    for h in HORIZONS:
        mdl_lgbm, feat_cols_lgbm = models_by_h[h]["lgbm"]
        mdl_xgb, xgb_feats = models_by_h[h]["xgb"]

        target_date = last_date + timedelta(days=h)
        tmp = pd.DataFrame({
            "영업일자": [target_date] * len(items),
            "영업장명_메뉴명": items
        })
        tmp["영업일자"] = pd.to_datetime(tmp["영업일자"])
        tmp = _time_feats(tmp)

        for k in LAG_LIST:
            vals = []
            for item in items:
                ref_date = target_date - timedelta(days=k)
                if (k >= h) and (ref_date <= last_date):
                    v = hist[(hist["영업장명_메뉴명"] == item) & (hist["영업일자"] == ref_date)]["매출수량"]
                    vals.append(float(v.iloc[0]) if len(v) else np.nan)
                else:
                    vals.append(np.nan)
            tmp[f"lag_{k}"] = vals

        for w in ROLLS_MEAN:
            tmp[f"rmean_{w}"] = stats["rmean"][w]
        for w in ROLLS_ZERO:
            tmp[f"zero_count_{w}"] = stats["zero_count"][w]
        tmp["zero_streak"] = stats["zero_streak"]

        tmp["영업장명_메뉴명"] = tmp["영업장명_메뉴명"].astype("category")
        tmp["영업장명_메뉴명_code"] = tmp["영업장명_메뉴명"].cat.codes.astype("int32")

        base_feats = []
        if USE_TIME_FEATURES:
            base_feats += ["dow", "month", "is_weekend"]
        base_feats += [f"lag_{k}" for k in LAG_LIST] + \
                      [f"rmean_{w}" for w in ROLLS_MEAN] + \
                      [f"zero_count_{w}" for w in ROLLS_ZERO] + \
                      ["zero_streak", "영업장명_메뉴명", "영업장명_메뉴명_code"]

        X_tmp = tmp[base_feats].copy()
        num_cols = X_tmp.select_dtypes(exclude=["category","object"]).columns
        if "rmean_7" in tmp.columns:
            X_tmp[num_cols] = X_tmp[num_cols].fillna(tmp["rmean_7"]).astype(np.float32)
        else:
            X_tmp[num_cols] = X_tmp[num_cols].fillna(0.0).astype(np.float32)

        best_iter_lgbm = getattr(mdl_lgbm, "best_iteration_", None)
        preds_lgbm = mdl_lgbm.predict(X_tmp[feat_cols_lgbm], num_iteration=best_iter_lgbm)

        X_xgb = X_tmp[mdl_xgb[1]] if isinstance(mdl_xgb, tuple) else X_tmp
        dtest = xgb.DMatrix(X_tmp[xgb_feats])
        if getattr(mdl_xgb, "best_iteration", None) is not None:
            ntree_limit = int(mdl_xgb.best_iteration) + 1
            preds_xgb = mdl_xgb.predict(dtest, iteration_range=(0, ntree_limit))
        else:
            preds_xgb = mdl_xgb.predict(dtest)

        preds_nbeats = np.array([nbeats_by_item[item][h-1] for item in items], dtype=np.float32)

        preds_sarimax = np.array([sarimax_by_item.get(item, [np.nan]*7)[h-1] for item in items], dtype=np.float32)
        if zip_mu is not None:
            preds_zip = np.full(len(items), float(zip_mu), dtype=np.float32)
        else:
            preds_zip = np.full(len(items), np.nan, dtype=np.float32)

        stack = np.vstack([
            preds_xgb, preds_lgbm, preds_nbeats, preds_sarimax, preds_zip
        ])

        base_w = np.array([W_XGB, W_LGBM, W_NBEATS, W_SARIMAX, W_ZIP], dtype=float)[:, None]

        mask = np.isfinite(stack)
        w = base_w * mask
        wsum = w.sum(axis=0)
        weighted = np.nansum(stack * w, axis=0)
        preds = np.where(wsum > 0, weighted / wsum, np.nan_to_num(weighted))

        preds = np.clip(preds, 0, None).astype(np.float32)

        all_rows.append(pd.DataFrame({
            "영업일자": [target_date] * len(items),
            "영업장명_메뉴명": items,
            "매출수량": preds,
            "h": h
        }))

    return pd.concat(all_rows, ignore_index=True)

# ===================== Main =====================
def main():
    t_all = time.perf_counter()

    train = pd.read_csv(TRAIN_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    test_paths = sorted(glob(TEST_GLOB))

    train["영업일자"] = pd.to_datetime(train["영업일자"])
    train = train.sort_values(["영업장명_메뉴명", "영업일자"])

    items = sample_sub.columns[1:].astype(str).to_list()

    global GLOBAL_GROUP_MAP, GLOBAL_GROUP_LETTERS
    GLOBAL_GROUP_MAP, GLOBAL_GROUP_LETTERS = _load_cluster_groups(CLUSTER_MEMBERS_GLOB)
    print(f"[GROUP] Loaded groups: {GLOBAL_GROUP_LETTERS if (USE_GROUP_FEATURES and GLOBAL_GROUP_LETTERS) else 'DISABLED'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[NBEATS-GEN] device: {device}")
    nbeats_model = _train_nbeats_global(train, device)

    t0 = time.perf_counter()
    datasets = build_datasets_by_horizon(train)
    print(f"[Prep] build_datasets_by_horizon in {time.perf_counter()-t0:.1f}s")

    t1 = time.perf_counter()
    models_by_h = train_models_by_horizon_all(datasets)
    print(f"[Train] trees per horizon in {time.perf_counter()-t1:.1f}s")

    res_zip = fit_zip_global_intercept_only(train)
    zip_mu = zip_global_expected_mean(res_zip)
    print(f"[ZIP] global intercept-only mean = {zip_mu if zip_mu is not None else 'None'}")

    global W_XGB, W_LGBM, W_NBEATS, W_SARIMAX, W_ZIP
    tuned = auto_tune_weights_scipy(models_by_h, datasets, nbeats_model, zip_mu, train, device)
    if tuned is not None:
        scale = max(0.0, 1.0 - float(W_SARIMAX))
        w_xgb = float(tuned.get("xgb", 0.0))
        w_lgbm = float(tuned.get("lgbm", 0.0))
        w_nbeats = float(tuned.get("nbeats", 0.0))
        w_zip = float(tuned.get("zip", 0.0)) if ("zip" in tuned) else 0.0
        s = w_xgb + w_lgbm + w_nbeats + w_zip
        if s <= 1e-12:
            print("[AUTO WEIGHTS] tuned weights sum=0, keep defaults.")
        else:
            W_XGB, W_LGBM, W_NBEATS, W_ZIP = [scale * w / s for w in (w_xgb, w_lgbm, w_nbeats, w_zip)]
        print(f"[AUTO WEIGHTS] applied: "
              f"xgb={W_XGB:.3f}, lgbm={W_LGBM:.3f}, nbeats={W_NBEATS:.3f}, zip={W_ZIP:.3f}, "
              f"sarimax={W_SARIMAX:.3f}  (sum={W_XGB+W_LGBM+W_NBEATS+W_ZIP+W_SARIMAX:.3f})")
    else:
        print(f"[AUTO WEIGHTS] skipped. keep defaults: "
              f"xgb={W_XGB:.3f}, lgbm={W_LGBM:.3f}, nbeats={W_NBEATS:.3f}, zip={W_ZIP:.3f}, "
              f"sarimax={W_SARIMAX:.3f}")

    print("[SARIMAX] selecting items by AIC (no-season)...")
    sarimax_orders, chosen_sarimax = _aic_select_items_sarimax_noseason(train, items)
    print(f"[SARIMAX] chosen {len(chosen_sarimax)}/{len(items)}")

    submission = sample_sub.copy()
    for c in submission.columns[1:]:
        submission[c] = submission[c].astype(float)

    if not test_paths:
        print("[경고] TEST 파일이 발견되지 않았습니다. (경로 확인)")

    for path in test_paths:
        t_block = time.perf_counter()
        tdf = pd.read_csv(path)
        tdf["영업일자"] = pd.to_datetime(tdf["영업일자"])
        tdf = tdf[tdf["영업장명_메뉴명"].isin(items)].copy()

        block_id = os.path.splitext(os.path.basename(path))[0]
        preds_block = predict_block_ensemble(
            models_by_h, nbeats_model, train,
            sarimax_orders, chosen_sarimax, zip_mu,
            tdf, np.array(items, dtype=str), device
        )

        for h in HORIZONS:
            row_label = f"{block_id}+{h}일"
            row_vals = (preds_block.loc[preds_block["h"] == h]
                        .set_index("영업장명_메뉴명")["매출수량"]
                        .reindex(items).fillna(0.0))
            submission.loc[submission["영업일자"] == row_label, items] = row_vals.values

        print(f"[Predict] {os.path.basename(path)} in {time.perf_counter()-t_block:.1f}s")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[완료] 저장: {OUTPUT_PATH}")
    print(f"[Total] end-to-end in {time.perf_counter()-t_all:.1f}s")

if __name__ == "__main__":
    main()
