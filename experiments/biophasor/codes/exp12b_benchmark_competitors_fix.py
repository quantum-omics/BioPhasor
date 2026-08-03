import sys, os, json, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
np.random.seed(0)
from experiments._shared import common
from experiments._shared.revision_data import load_matched_cptac

SUITE = "biophasor"
RESULTS = common.results_dir(SUITE)
from sklearn.metrics import silhouette_score, roc_auc_score, adjusted_rand_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---- sklearn 1.9 compat shim for snfpy (force_all_finite -> ensure_all_finite) ----
import sklearn.utils.validation as skv
_orig_check_array = skv.check_array
def _patched(*a, **k):
    if "force_all_finite" in k:
        k["ensure_all_finite"] = k.pop("force_all_finite")
    return _orig_check_array(*a, **k)
skv.check_array = _patched
# also patch the name imported inside sklearn.metrics.pairwise
try:
    import sklearn.metrics.pairwise as smp
    if hasattr(smp,"check_array"): smp.check_array = _patched
except Exception: pass

rna, prot, y, _genes = load_matched_cptac(complete_case=True)
R = rna.values.astype(float); P = prot.values.astype(float)
out = json.load(open(os.path.join(RESULTS,"multiomics_benchmark_results.json")))

def eval_embedding(Z, name):
    Z = StandardScaler().fit_transform(Z)
    sil = float(silhouette_score(Z, y))
    ari = float(adjusted_rand_score(y, KMeans(2,n_init=10,random_state=0).fit_predict(Z)))
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    proba = cross_val_predict(LogisticRegression(max_iter=2000,class_weight="balanced"),Z,y,cv=skf,method="predict_proba")[:,1]
    auc = float(roc_auc_score(y, proba))
    out["methods"][name] = {"n_factors":int(Z.shape[1]),"silhouette_tumor_normal":round(sil,4),
                            "kmeans_ARI":round(ari,4),"logreg_AUC_5fold":round(auc,4)}
    print(f"{name:20s} k={Z.shape[1]:>3} sil={sil:+.3f} ARI={ari:.3f} AUC={auc:.3f}", flush=True)

# ---- MOFA+ (fixed Z extraction) ----
try:
    t1=time.time()
    from mofapy2.run.entry_point import entry_point
    views=[]
    for vname,M in [("RNA",R),("protein",P)]:
        df=pd.DataFrame(M, columns=rna.columns).copy(); df["sample"]=[f"s{i}" for i in range(len(y))]
        long=df.melt(id_vars="sample",var_name="feature",value_name="value"); long["view"]=vname; long["group"]="g1"
        views.append(long)
    data=pd.concat(views,ignore_index=True)[["sample","group","feature","view","value"]]
    ep=entry_point(); ep.set_data_options(scale_views=True); ep.set_data_df(data)
    ep.set_model_options(factors=10)
    ep.set_train_options(iter=300, convergence_mode="fast", seed=0, verbose=False, quiet=True)
    ep.build(); ep.run()
    # Z extraction: model node expectation -> list per group
    Znode = ep.model.nodes["Z"].getExpectation()
    Zm = np.asarray(Znode) if not isinstance(Znode,list) else np.asarray(Znode[0])
    print("MOFA+ Z shape", Zm.shape, flush=True)
    eval_embedding(Zm, "MOFA+"); out["methods"]["MOFA+"]["train_s"]=round(time.time()-t1,1)
except Exception as e:
    import traceback; out["methods"]["MOFA+"]={"error":str(e)[:200]}; print("MOFA+ ERR", traceback.format_exc()[-400:], flush=True)

# ---- SNF (with shim) ----
try:
    t2=time.time()
    import snf
    affs = snf.make_affinity([StandardScaler().fit_transform(R), StandardScaler().fit_transform(P)],
                             metric="euclidean", K=20, mu=0.5)
    fused = snf.snf(affs, K=20)
    from sklearn.manifold import SpectralEmbedding
    Zs = SpectralEmbedding(n_components=10, affinity="precomputed", random_state=0).fit_transform(fused)
    eval_embedding(Zs, "SNF"); out["methods"]["SNF"]["train_s"]=round(time.time()-t2,1)
    # native SNF spectral clustering
    best_k,_ = snf.get_n_clusters(fused)
    from sklearn.cluster import SpectralClustering
    lab = SpectralClustering(2, affinity="precomputed", random_state=0).fit_predict(fused)
    out["methods"]["SNF"]["snf_native_ARI_2clust"]=round(float(adjusted_rand_score(y,lab)),4)
    out["methods"]["SNF"]["snf_estimated_k"]=int(best_k)
except Exception as e:
    import traceback; out["methods"]["SNF"]={"error":str(e)[:200]}; print("SNF ERR", traceback.format_exc()[-400:], flush=True)

json.dump(out, open(os.path.join(RESULTS,"multiomics_benchmark_results.json"),"w"), indent=2)
print("DONE", flush=True)
