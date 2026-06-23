"""Adiciona baseline 'prior simples': EOS = POS_obs + mediana(POS->EOS do banco, mesmo season)."""
import sys; from pathlib import Path
import numpy as np, pandas as pd, warnings, statistics as st
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from fenologia.phenophase import extract_phenometrics, adaptive_smoothing, extrapolate_terminal_cycle

PARTS=Path("data/output/_parts"); col="evi_medio_soja"
dfs=[]
for p in sorted(PARTS.glob("*.parquet")):
    d=pd.read_parquet(p,columns=["id_hexagono","data",col]).dropna(subset=[col])
    if len(d): dfs.append(d)
df=pd.concat(dfs,ignore_index=True).drop_duplicates(["id_hexagono","data"])
counts=df.groupby("id_hexagono").size(); good=counts[counts>=80].index.tolist()
rng=np.random.default_rng(3); sample=list(rng.choice(good,size=40,replace=False))
run=lambda ts: extract_phenometrics(ts,ndvi_column="NDVI_mean",min_cycle_length_days=120,smoothing_method="both",quality_threshold=0.60)

rows=[]
for h in sample:
    grp=df[df.id_hexagono==h].sort_values("data")
    ts=grp.rename(columns={"data":"datetime",col:"NDVI_mean"})[["datetime","NDVI_mean"]]
    try: full=run(ts)
    except: continue
    if not full.get("success"): continue
    comp=[c for c in full["cycles"] if c.get("fit_success") and not c.get("at_series_end") and not c.get("at_series_start")]
    for c in comp:
        pos_date=pd.Timestamp(c["phenophase_dates"]["pos"]); eos_true=pd.Timestamp(c["phenophase_dates"]["eos"])
        for label,trunc in [("POS+10",pos_date+pd.Timedelta(days=10)),("POS+30",pos_date+pd.Timedelta(days=30))]:
            if trunc>=eos_true: continue
            sub=ts[ts.datetime<=trunc]
            if len(sub)<30: continue
            try: trr=run(sub)
            except: continue
            if not trr.get("success"): continue
            term=[x for x in trr["cycles"] if x.get("at_series_end") or x.get("cycle",{}).get("at_series_end")]
            if not term: continue
            t=term[-1]
            if not t.get("fit_success"): continue
            tstype=t["season_type"]
            # banco do truncado (mesmo season, completos)
            bank=[x for x in trr["cycles"] if x.get("fit_success") and not x.get("at_series_end") and not x.get("at_series_start")]
            bank_same=[x for x in bank if x["season_type"]==tstype] or bank
            if not bank_same: continue
            pos2eos=[x["phenophase_days"]["eos_days"]-x["phenophase_days"]["pos_days"] for x in bank_same]
            med=float(np.median(pos2eos))
            pos_obs=pd.Timestamp(t["phenophase_dates"]["pos"])
            eos_prior=pos_obs+pd.Timedelta(days=med)
            eos_fit=pd.Timestamp(t["phenophase_dates"]["eos"])
            ndvi=sub["NDVI_mean"].values.astype(float); dts=sub["datetime"].values
            sm=adaptive_smoothing(ndvi,dts,method="both")
            ex=extrapolate_terminal_cycle(sm,dts,trr["cycles"])
            eos_sm=pd.Timestamp(ex["forecast_eos_date"]) if ex.get("success") else None
            rows.append((label,(eos_fit-eos_true).days,(eos_sm-eos_true).days if eos_sm is not None else None,(eos_prior-eos_true).days))

print(f"N={len(rows)}")
for nm,idx in [("FIT logistico",1),("SHAPE matching",2),("PRIOR simples",3)]:
    for filt,fl in [("todos",lambda r:True),("POS+10",lambda r:r[0]=="POS+10"),("POS+30",lambda r:r[0]=="POS+30")]:
        vals=[abs(r[idx]) for r in rows if r[idx] is not None and fl(r)]
        if vals:
            # remove outliers >120d (re-segmentação errada) p/ ver sinal
            clean=[v for v in vals if v<=120]
            print(f"{nm:16s} [{filt:7s}] n={len(vals):3d}  MAE={st.mean(vals):6.1f}d  med={st.median(vals):5.1f}d  | sem_outlier(<120d) n={len(clean)} MAE={st.mean(clean) if clean else 0:.1f}d")
    print()
