"""
Aftershock-rate prediction — final model.
Trains an XGBoost regressor in LOG space on M>=5.0 mainshocks.
Honest evaluation via 5-fold CV + leave-one-out.
"""
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_predict, LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

RADIUS_KM, PRE_DAYS, POST_DAYS, MIN_MAG = 100, 30, 30, 5.0
FEATURES = ["mag","depth","latitude","longitude",
            "pre_rate","pre_mean_m","pre_max_m","pre_std_m","b_est"]
PARAMS = dict(n_estimators=50, max_depth=3, learning_rate=0.10, reg_lambda=0.5,
              subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
              random_state=42, verbosity=0)

def haversine_km(lat1, lon1, lat2, lon2):
    R=6371.0; p1,p2=np.radians(lat1),np.radians(lat2)
    dphi=np.radians(lat2-lat1); dlmb=np.radians(lon2-lon1)
    a=np.sin(dphi/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dlmb/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def build_features(df, radius_km=RADIUS_KM, pre_days=PRE_DAYS,
                   post_days=POST_DAYS, min_mag=MIN_MAG):
    df=df.sort_values("time").reset_index(drop=True)
    # Drop tz info so per-row np.datetime64(m.time) conversions don't warn;
    # all timestamps are UTC, so this is a no-op on the actual values.
    if pd.api.types.is_datetime64tz_dtype(df["time"]):
        df["time"]=df["time"].dt.tz_localize(None)
    lat,lon=df.latitude.values,df.longitude.values
    t=df.time.values.astype("datetime64[ns]"); mag=df.mag.values
    rows=[]
    for _,m in df[df.mag>=min_mag].iterrows():
        t0=np.datetime64(m.time); d=haversine_km(m.latitude,m.longitude,lat,lon)
        win=d<=radius_km
        pre =win&(t>=t0-np.timedelta64(pre_days,"D"))&(t<t0)
        post=win&(t>t0)&(t<=t0+np.timedelta64(post_days,"D"))
        pm=mag[pre]; n_pre=int(pre.sum()); n_aft=int(post.sum())
        if n_pre>=3:
            b_est=np.log10(np.e)/(pm.mean()-pm.min()+0.05)
            pre_mean,pre_max,pre_std=pm.mean(),pm.max(),pm.std()
        else:
            b_est,pre_mean,pre_max,pre_std=1.0,0.0,0.0,0.0
        rows.append(dict(mag=m.mag,depth=m.depth,latitude=m.latitude,longitude=m.longitude,
            pre_rate=n_pre/pre_days,pre_mean_m=pre_mean,pre_max_m=pre_max,
            pre_std_m=pre_std,b_est=b_est,n_aftershocks=n_aft))
    return pd.DataFrame(rows)

def train_model(df, outdir="."):
    feat=build_features(df); feat.to_csv(f"{outdir}/features.csv",index=False)
    X=feat[FEATURES].values; y=feat["n_aftershocks"].values; ylog=np.log1p(y)
    print(f"Feature matrix: {feat.shape}  (M>={MIN_MAG} mainshocks)")

    model=xgb.XGBRegressor(**PARAMS)

    # ---- 5-fold CV (headline metrics) ----
    cv=KFold(n_splits=5,shuffle=True,random_state=42)
    yl=cross_val_predict(model,X,ylog,cv=cv); yp=np.expm1(yl)
    count_rmse=np.sqrt(mean_squared_error(y,yp))
    print("\n--- 5-fold cross-validation ---")
    print(f"  log-RMSE : {np.sqrt(mean_squared_error(ylog,yl)):.3f}")
    print(f"  log-R^2  : {r2_score(ylog,yl):+.3f}")
    print(f"  count-RMSE: {count_rmse:.0f} aftershocks")
    print(f"  count-MAE : {mean_absolute_error(y,yp):.0f} aftershocks")

    # ---- Leave-one-out (robust for small n) ----
    yl_loo=cross_val_predict(model,X,ylog,cv=LeaveOneOut())
    print("\n--- Leave-one-out ---")
    print(f"  log-R^2  : {r2_score(ylog,yl_loo):+.3f}")
    print(f"  count-MAE: {mean_absolute_error(y,np.expm1(yl_loo)):.0f} aftershocks")

    # ---- naive baseline for context ----
    base=np.full_like(ylog,ylog.mean())
    print(f"\nBaseline (predict mean): count-RMSE={np.sqrt(mean_squared_error(y,np.expm1(base))):.0f}, "
          f"log-R^2={r2_score(ylog,base):+.3f}")

    # ---- residual diagnostics (log space + log-log actual/pred) ----
    resid=ylog-yl
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    ax[0].scatter(yl,resid,alpha=0.6,s=25,edgecolor="k",linewidth=.3)
    ax[0].axhline(0,color="r",lw=1)
    ax[0].set_xlabel("Predicted log1p(count)"); ax[0].set_ylabel("Residual")
    ax[0].set_title("Residuals vs Predicted (log space)")
    lo,hi=min(y.min(),yp.min()),max(y.max(),yp.max())
    ax[1].scatter(y,np.clip(yp,1,None),alpha=0.6,s=25,edgecolor="k",linewidth=.3)
    ax[1].plot([1,hi],[1,hi],"r--"); ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("Actual aftershocks"); ax[1].set_ylabel("Predicted aftershocks")
    ax[1].set_title("Actual vs Predicted (log-log)")
    plt.tight_layout(); plt.savefig(f"{outdir}/model_residuals.png",dpi=150); plt.close()

    # ---- feature importance ----
    model.fit(X,ylog); joblib.dump(model,f"{outdir}/xgb_model.pkl")
    imp=pd.Series(model.feature_importances_,index=FEATURES).sort_values()
    imp.plot(kind="barh",figsize=(7,4),color="#2c7fb8")
    plt.title("XGBoost Feature Importance (log-aftershock model)")
    plt.tight_layout(); plt.savefig(f"{outdir}/feature_importance.png",dpi=150); plt.close()
    print("\nTop features:", ", ".join(imp.sort_values(ascending=False).index[:4]))
    return model, count_rmse

if __name__=="__main__":
    df=pd.read_csv("catalog.csv")
    df["time"]=pd.to_datetime(df["time"],utc=True,format="ISO8601").dt.tz_localize(None)
    train_model(df, outdir="/tmp/out")
