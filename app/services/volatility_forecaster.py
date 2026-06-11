from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from arch import arch_model
from sklearn.preprocessing import StandardScaler

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

SEQ_LEN = 30
TEST_DAYS = 60
ROLL_TARGET = 21
ROLL_5 = 5
ROLL_21 = 21
EPOCHS = 50
PATIENCE = 10
MODEL_TTL = 86400


class VolLSTM(nn.Module):
    def __init__(self, input_size: int = 4, hidden_size: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


@dataclass
class VolatilityForecastMetrics:
    predicted_vol: float
    garch_vol: float
    lstm_mae: float
    lstm_rmse: float
    garch_mae: float
    garch_rmse: float
    direction_accuracy: float
    vol_regime: str
    historical_vol_mean: float
    lstm_mae_high: float | None = None
    garch_mae_high: float | None = None


@dataclass
class VolatilityForecastResult:
    metrics: VolatilityForecastMetrics
    adjusted_var_95: float | None = None


class VolatilityForecaster:
    def __init__(self, returns: pd.Series, ticker: str = ""):
        self.ticker = ticker.upper()
        self.returns = returns.dropna().astype(float)
        self._scaler: StandardScaler | None = None
        self._model: VolLSTM | None = None
        self._device = torch.device("cpu")

    @staticmethod
    def _realized_vol(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window).std() * np.sqrt(252)

    def _build_frame(self) -> pd.DataFrame:
        r = self.returns
        df = pd.DataFrame(
            {
                "return": r,
                "vol_5": self._realized_vol(r, ROLL_5),
                "vol_21": self._realized_vol(r, ROLL_21),
                "sq_return": r**2,
                "target_vol": self._realized_vol(r, ROLL_TARGET),
            }
        )
        return df.dropna()

    def _sequences(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feature_cols = ["return", "vol_5", "vol_21", "sq_return"]
        feats = df[feature_cols].values
        targets = df["target_vol"].values
        xs, ys, indices = [], [], []
        for i in range(SEQ_LEN, len(df)):
            xs.append(feats[i - SEQ_LEN : i])
            ys.append(targets[i])
            indices.append(df.index[i])
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32), np.array(indices)

    def _load_model_from_redis(self) -> bool:
        try:
            import redis

            client = redis.from_url(settings.redis_url, decode_responses=False)
            raw = client.get(f"vol_model:{self.ticker}")
            if not raw:
                return False
            buf = io.BytesIO(raw)
            state = torch.load(buf, map_location=self._device, weights_only=True)
            self._model = VolLSTM().to(self._device)
            self._model.load_state_dict(state)
            self._model.eval()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("vol_model_cache_load_failed", extra={"ticker": self.ticker, "error": str(exc)})
            return False

    def _save_model_to_redis(self) -> None:
        if self._model is None:
            return
        try:
            import redis

            buf = io.BytesIO()
            torch.save(self._model.state_dict(), buf)
            client = redis.from_url(settings.redis_url, decode_responses=False)
            client.setex(f"vol_model:{self.ticker}", MODEL_TTL, buf.getvalue())
        except Exception as exc:  # noqa: BLE001
            logger.warning("vol_model_cache_save_failed", extra={"ticker": self.ticker, "error": str(exc)})

    def train(self, skip_if_cached: bool = True) -> None:
        if skip_if_cached and self._load_model_from_redis():
            return
        df = self._build_frame()
        if len(df) < SEQ_LEN + TEST_DAYS + 10:
            raise ValueError("Insufficient return history for volatility model")
        x_all, y_all, _ = self._sequences(df)
        split = len(x_all) - TEST_DAYS
        if split < 20:
            raise ValueError("Insufficient training rows for LSTM")
        x_train, y_train = x_all[:split], y_all[:split]
        x_val, y_val = x_all[split - 20 : split], y_all[split - 20 : split]

        self._scaler = StandardScaler()
        n_train, seq, feat = x_train.shape
        x_train_flat = x_train.reshape(-1, feat)
        self._scaler.fit(x_train_flat)
        x_train = self._scaler.transform(x_train_flat).reshape(n_train, seq, feat)
        x_val_flat = x_val.reshape(-1, feat)
        x_val = self._scaler.transform(x_val_flat).reshape(x_val.shape[0], seq, feat)

        model = VolLSTM().to(self._device)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        best_val = float("inf")
        stale = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        xt = torch.tensor(x_train, device=self._device)
        yt = torch.tensor(y_train, device=self._device)
        xv = torch.tensor(x_val, device=self._device)
        yv = torch.tensor(y_val, device=self._device)

        for _ in range(EPOCHS):
            model.train()
            opt.zero_grad()
            pred = model(xt)
            loss = loss_fn(pred, yt)
            loss.backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(xv), yv).item())
            if val_loss < best_val:
                best_val = val_loss
                stale = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
        model.load_state_dict(best_state)
        model.eval()
        self._model = model
        self._save_model_to_redis()

    def _predict_lstm_batch(self, x: np.ndarray) -> np.ndarray:
        if self._model is None or self._scaler is None:
            raise RuntimeError("Model not trained")
        n, seq, feat = x.shape
        flat = self._scaler.transform(x.reshape(-1, feat)).reshape(n, seq, feat)
        with torch.no_grad():
            t = torch.tensor(flat, device=self._device)
            return self._model(t).cpu().numpy()

    def garch_forecast(self) -> float:
        am = arch_model(self.returns * 100, vol="Garch", p=1, q=1, rescale=False)
        res = am.fit(disp="off", show_warning=False)
        f = res.forecast(horizon=1)
        var = f.variance.values[-1, -1]
        return float(np.sqrt(var) / 100 * np.sqrt(252))

    @staticmethod
    def get_vol_adjusted_var(historical_var_95: float, predicted_vol: float, historical_vol_mean: float) -> float:
        if historical_vol_mean <= 0:
            return historical_var_95
        return historical_var_95 * (predicted_vol / historical_vol_mean)

    def _regime_label(self, vol_value: float, historical: pd.Series) -> str:
        p10 = float(historical.quantile(0.10))
        p90 = float(historical.quantile(0.90))
        if vol_value < p10:
            return "LOW"
        if vol_value > p90:
            return "HIGH"
        return "MEDIUM"

    def evaluate_and_forecast(self, historical_var_95: float | None = None) -> VolatilityForecastResult:
        self.train()
        df = self._build_frame()
        x_all, y_all, _ = self._sequences(df)
        split = len(x_all) - TEST_DAYS
        x_test, y_test = x_all[split:], y_all[split:]
        if len(x_test) == 0:
            raise ValueError("No test sequences for evaluation")

        lstm_preds = self._predict_lstm_batch(x_test)
        garch_vol = self.garch_forecast()
        train_end = max(80, len(self.returns) - TEST_DAYS - 1)
        stale = self.returns.iloc[:train_end]
        try:
            am = arch_model(stale * 100, vol="Garch", p=1, q=1, rescale=False)
            res = am.fit(disp="off", show_warning=False)
            var = res.forecast(horizon=1).variance.values[-1, -1]
            stale_garch = float(np.sqrt(var) / 100 * np.sqrt(252))
        except Exception:  # noqa: BLE001
            stale_garch = garch_vol
        garch_preds = np.full_like(y_test, stale_garch, dtype=float)

        lstm_mae = float(np.mean(np.abs(lstm_preds - y_test)))
        lstm_rmse = float(np.sqrt(np.mean((lstm_preds - y_test) ** 2)))
        garch_mae = float(np.mean(np.abs(garch_preds - y_test)))
        garch_rmse = float(np.sqrt(np.mean((garch_preds - y_test) ** 2)))

        prev = np.concatenate([[y_test[0]], y_test[:-1]])
        lstm_dir = np.sign(lstm_preds - prev) == np.sign(y_test - prev)
        direction_accuracy = float(np.mean(lstm_dir))

        hist_vol = df["target_vol"]
        hist_mean = float(hist_vol.mean())
        current_regime = self._regime_label(float(y_test[-1]), hist_vol)

        high_mask = np.array([self._regime_label(v, hist_vol) == "HIGH" for v in y_test])
        if high_mask.any():
            lstm_mae_high = float(np.mean(np.abs(lstm_preds[high_mask] - y_test[high_mask])))
            garch_mae_high = float(np.mean(np.abs(garch_preds[high_mask] - y_test[high_mask])))
        else:
            lstm_mae_high = None
            garch_mae_high = None

        predicted_vol = float(lstm_preds[-1])
        adjusted = None
        if historical_var_95 is not None:
            adjusted = self.get_vol_adjusted_var(float(historical_var_95), predicted_vol, hist_mean)

        metrics = VolatilityForecastMetrics(
            predicted_vol=predicted_vol,
            garch_vol=garch_vol,
            lstm_mae=lstm_mae,
            lstm_rmse=lstm_rmse,
            garch_mae=garch_mae,
            garch_rmse=garch_rmse,
            direction_accuracy=direction_accuracy,
            vol_regime=current_regime,
            historical_vol_mean=hist_mean,
            lstm_mae_high=lstm_mae_high,
            garch_mae_high=garch_mae_high,
        )
        return VolatilityForecastResult(metrics=metrics, adjusted_var_95=adjusted)


def forecast_is_fresh(computed_at: datetime | None, hours: int = 24) -> bool:
    if computed_at is None:
        return False
    if computed_at.tzinfo is None:
        computed_at = computed_at.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - computed_at < timedelta(hours=hours)
