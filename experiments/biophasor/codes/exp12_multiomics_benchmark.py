import sys, os, json, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
np.random.seed(0)
from experiments._shared import common
from experiments._shared.revision_data import load_matched_cptac

SUITE = "biophasor"
RESULTS = common.results_dir(SUITE)
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

rna, prot, y, _genes = load_matched_cptac(complete_case=True)
R = rna.values.astype(float); P = prot.values.astype(float)
out = {"n_samples":len(y),"n_genes":R.shape[1],"tumor":int(y.sum()),"normal":int((1-y).sum()),"methods":{}}

def eval_embedding(Z, name, n_factors=None):
    Z = StandardScaler().fit_transform(Z)
    # unsupervised separability by true label
    sil = float(silhouette_score(Z, y)) if len(np.unique(y))>1 else np.nan
    # unsupervised 2-cluster recovery of tumor/normal
    km = KMeans(2, n_init=10, random_state=0).fit_predict(Z)
    ari = float(adjusted_rand_score(y, km))
    # downstream classification (LogReg, stratified 5-fold, AUC)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    try:
        proba = cross_val_predict(LogisticRegression(max_iter=2000,class_weight="balanced"),
                                  Z, y, cv=skf, method="predict_proba")[:,1]
        auc = float(roc_auc_score(y, proba))
    except Exception as e:
        auc = np.nan
    out["methods"][name] = {"n_factors": int(Z.shape[1]) if n_factors is None else n_factors,
                            "silhouette_tumor_normal": round(sil,4),
                            "kmeans_ARI": round(ari,4), "logreg_AUC_5fold": round(auc,4)}
    print(f"{name:26s} k={Z.shape[1]:>4} sil={sil:+.3f} ARI={ari:.3f} AUC={auc:.3f}", flush=True)

t0=time.time()
# ---- BioPhasor: phasor encode + fusion variants ----
import biophasor as bp
from biophasor.core.operators import coherence
import biophasor.integration.multiomics as mm
phi_rna = bp.tanh_phase_encode(R, log_transform=True)
phi_prot = bp.tanh_phase_encode(P, log_transform=False)
ig = mm.MultiOmicsIntegrator(['RNA','protein'])
fused_orig = ig.fuse({'RNA':phi_rna,'protein':phi_prot}, method='circular_mean')
fused_gated = ig.fuse({'RNA':phi_rna,'protein':phi_prot}, method='coherence_gated')
# For a fair low-dim comparison, PCA the fused phasor cos/sin features to k=10
from sklearn.decomposition import PCA
def phasor_embed(phi, k=10):
    feat = np.concatenate([np.cos(phi), np.sin(phi)], axis=1)
    return PCA(k, random_state=0).fit_transform(feat)
eval_embedding(phasor_embed(fused_orig), "BioPhasor_fusion_orig")
eval_embedding(phasor_embed(fused_gated), "BioPhasor_fusion_gated")
# coherence-score based (training-free) 1D
tcs = coherence(np.concatenate([phi_rna,phi_prot],axis=1), axis=1).reshape(-1,1)
eval_embedding(tcs, "BioPhasor_TorusCoherence1D")
print("biophasor done %.1fs"%(time.time()-t0), flush=True)

# ---- MOFA+ ----
try:
    t1=time.time()
    from mofapy2.run.entry_point import entry_point
    # MOFA expects features x samples per view; build long df
    views=[]
    for vname,M in [("RNA",R),("protein",P)]:
        df=pd.DataFrame(M, columns=rna.columns).copy()
        df["sample"]=[f"s{i}" for i in range(len(y))]
        long=df.melt(id_vars="sample",var_name="feature",value_name="value")
        long["view"]=vname; long["group"]="g1"
        views.append(long)
    data=pd.concat(views,ignore_index=True)[["sample","group","feature","view","value"]]
    ep=entry_point()
    ep.set_data_options(scale_views=True)
    ep.set_data_df(data)
    ep.set_model_options(factors=10)
    ep.set_train_options(iter=200, convergence_mode="fast", seed=0, verbose=False, quiet=True)
    ep.build(); ep.run()
    Zm = ep.model.getExpectations()["Z"]["group0"] if hasattr(ep.model,"getExpectations") else None
    if Zm is None:
        import h5py
    eval_embedding(np.asarray(Zm), "MOFA+")
    out["methods"]["MOFA+"]["train_s"]=round(time.time()-t1,1)
except Exception as e:
    out["methods"]["MOFA+"]={"error":str(e)[:200]}
    print("MOFA+ ERR", str(e)[:200], flush=True)

# ---- SNF ----
try:
    t2=time.time()
    import snf
    affs = snf.make_affinity([StandardScaler().fit_transform(R), StandardScaler().fit_transform(P)],
                             metric="euclidean", K=20, mu=0.5)
    fused = snf.snf(affs, K=20)
    # spectral embedding of the fused network
    from sklearn.manifold import SpectralEmbedding
    Zs = SpectralEmbedding(n_components=10, affinity="precomputed", random_state=0).fit_transform(fused)
    eval_embedding(Zs, "SNF")
    out["methods"]["SNF"]["train_s"]=round(time.time()-t2,1)
    # SNF native clustering
    from snf import metrics
    labels_snf = snf.spectral_clustering(fused, n_clusters=2)
    out["methods"]["SNF"]["snf_native_ARI"]=round(float(adjusted_rand_score(y,labels_snf)),4)
except Exception as e:
    out["methods"]["SNF"]={"error":str(e)[:200]}
    print("SNF ERR", str(e)[:200], flush=True)

json.dump(out, open(os.path.join(RESULTS,"multiomics_benchmark_results.json"),"w"), indent=2)
print("TOTAL %.1fs"%(time.time()-t0), flush=True)
