import warnings; warnings.filterwarnings("ignore")
import os; import numpy as np, pandas as pd, json
from experiments._shared import common
from experiments._shared.revision_data import load_matched_cptac, load_clinical

SUITE = "biophasor"
RESULTS = common.results_dir(SUITE)
np.random.seed(0)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

clin = load_clinical()
rna, prot, _y, _genes = load_matched_cptac(complete_case=True)
import biophasor as bp
from biophasor.core.operators import coherence
import biophasor.integration.multiomics as mm

def make_task(name, col, pos, neg):
    common=[i for i in rna.index if i in clin.index]
    c=clin.loc[common, col].astype(str)
    mask=c.str.contains(pos,case=False,na=False)|c.str.contains(neg,case=False,na=False)
    ids=[common[k] for k in range(len(common)) if mask.iloc[k]]
    yv=np.array([1 if pos.lower() in str(clin.loc[i,col]).lower() else 0 for i in ids])
    return name, ids, yv

tasks=[ make_task("grade_G1_vs_G3","histologic_grade","G3","G1"),
        make_task("type_endometrioid_vs_serous","histologic_type","Serous","Endometrioid") ]

def features(ids):
    R=rna.loc[ids].values.astype(float); P=prot.loc[ids].values.astype(float)
    phi_r=bp.tanh_phase_encode(R,log_transform=True); phi_p=bp.tanh_phase_encode(P,log_transform=False)
    ig=mm.MultiOmicsIntegrator(['RNA','protein'])
    fused=ig.fuse({'RNA':phi_r,'protein':phi_p},method='coherence_gated')
    phasor_feat=np.concatenate([np.cos(fused),np.sin(fused)],axis=1)
    Zbp=PCA(20,random_state=0).fit_transform(phasor_feat)
    # raw concatenated omics baseline (Euclidean) with matched PCA-20
    raw=np.concatenate([StandardScaler().fit_transform(R),StandardScaler().fit_transform(P)],axis=1)
    Zraw=PCA(20,random_state=0).fit_transform(raw)
    # training-free torus coherence score (1 feature)
    tcs=coherence(np.concatenate([phi_r,phi_p],axis=1),axis=1).reshape(-1,1)
    return {"BioPhasor_gated_PCA20":Zbp, "RawOmics_PCA20":Zraw, "TorusCoherence1D":tcs}

def boot_ci(y,p,fn,n=2000):
    rng=np.random.default_rng(0); vals=[]
    idx=np.arange(len(y))
    for _ in range(n):
        b=rng.choice(idx,len(idx),replace=True)
        if len(np.unique(y[b]))<2: continue
        try: vals.append(fn(y[b],p[b]))
        except: pass
    return (round(float(np.percentile(vals,2.5)),3), round(float(np.percentile(vals,97.5)),3)) if vals else (None,None)

out={}
for name,ids,yv in tasks:
    feats=features(ids)
    out[name]={"n":len(ids),"pos":int(yv.sum()),"neg":int((yv==0).sum()),"models":{}}
    print(f"\n### {name}: n={len(ids)} pos={int(yv.sum())} neg={int((yv==0).sum())}",flush=True)
    skf=StratifiedKFold(5,shuffle=True,random_state=0)
    for fname,Z in feats.items():
        Zs=StandardScaler().fit_transform(Z)
        for mdl_name,mdl in [("logreg",LogisticRegression(max_iter=2000,class_weight="balanced")),
                             ("mlp",MLPClassifier((32,),max_iter=500,random_state=0))]:
            if fname=="TorusCoherence1D" and mdl_name=="mlp": continue
            try:
                proba=cross_val_predict(mdl,Zs,yv,cv=skf,method="predict_proba")[:,1]
                auc=roc_auc_score(yv,proba); auprc=average_precision_score(yv,proba)
                bacc=balanced_accuracy_score(yv,(proba>0.5).astype(int))
                key=f"{fname}|{mdl_name}"
                out[name]["models"][key]={"AUC":round(float(auc),3),"AUC_CI":boot_ci(yv,proba,roc_auc_score),
                    "AUPRC":round(float(auprc),3),"bal_acc":round(float(bacc),3)}
                print(f"  {key:34s} AUC={auc:.3f} {out[name]['models'][key]['AUC_CI']} AUPRC={auprc:.3f} bAcc={bacc:.3f}",flush=True)
            except Exception as e:
                print(f"  {fname}|{mdl_name} ERR {str(e)[:80]}",flush=True)
json.dump(out,open(os.path.join(RESULTS,"hardtask_clinical_results.json"),"w"),indent=2)
print("\nDONE",flush=True)
