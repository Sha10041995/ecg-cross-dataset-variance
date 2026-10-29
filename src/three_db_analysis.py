"""Three-database analysis from saved per-seed scores. Run from the project root:
    python three_db_analysis.py > results/three_db_analysis.txt
"""
import sys, json, numpy as np
from scipy import stats
sys.argv=sys.argv[:1]; sys.path.insert(0,'.')
import classical_baseline as cb
U=''
new=json.load(open(U+'results/multiseed3_runs.json')); old=json.load(open(U+'results/multiseed_runs.json'))
print("REPRO vs original multiseed_runs.json")
for n_,o_ in zip(new,old):
    d=max(abs(n_[k]['AUC']-o_[k]['AUC']) for k in('cnn_ptbxl','cnn_mitbih','ae_ptbxl','ae_mitbih'))
    d=max(d,abs(n_['ae_threshold']-o_['ae_threshold']))
    print(f"  seed {n_['seed']}: max abs diff = {d:.3e}")
cl=np.load(U+'processed/classical_scores_chapman.npz'); clm=np.load(U+'processed/classical_scores.npz')
yb=cl['y_cha_broad']; ys=cl['y_cha']
S=[np.load(U+f'results/multiseed3_scores/seed{s}.npz') for s in range(5)]
assert (S[0]['y_cha']==ys).all()
def dl(y,a,b): return cb.delong_test(y.astype(int),a,b)
def sec(title,y,key_c,key_a,lr,rf,ytag):
    print(f"\n== {title} ==")
    diffs=[];ses=[];rows=[]
    for s,S_ in enumerate(S):
        c,a=S_[key_c],S_[key_a]
        d,z,p=dl(y,a,c); se=abs(d/z) if np.isfinite(z) and z!=0 else float('nan')
        dl_,zl,pl=dl(y,c,lr); dr,zr,pr=dl(y,c,rf)
        diffs.append(d);ses.append(se)
        print(f"  seed {s}: AE-CNN {d:+.4f} (DeLong SE {se:.4f}, p={p:.2g}) | CNN-LR {dl_:+.4f} p={pl:.2g} | CNN-RF {dr:+.4f} p={pr:.2g}")
    diffs=np.array(diffs);ses=np.array(ses)
    print(f"  AE-CNN across seeds: mean {diffs.mean():+.4f}, SD {diffs.std(ddof=1):.4f}; mean eval-set SE {np.nanmean(ses):.4f}; ratio SD/SE = {diffs.std(ddof=1)/np.nanmean(ses):.1f}")
sec("Chapman strict label",ys,'cnn_cha','ae_cha',cl['lr_cha'],cl['rf_cha'],'s')
sec("Chapman broad label",yb,'cnn_cha','ae_cha',cl['lr_cha'],cl['rf_cha'],'b')
ym=S[0]['y_mit']; assert (clm['y_out']==ym).all()
sec("MIT-BIH (for comparison; same scores as in the paper)",ym,'cnn_mit','ae_mit',clm['lr_out'],clm['rf_out'],'m')
# classical AUCs and CNN gap under broad, per seed
print("\nCNN/AE AUC on Chapman broad label per seed vs classical:")
from sklearn.metrics import roc_auc_score as A
print("  LR",round(A(yb,cl['lr_cha']),4),"RF",round(A(yb,cl['rf_cha']),4))
for s,S_ in enumerate(S): print("  seed",s,"CNN",round(A(yb,S_['cnn_cha']),4),"AE",round(A(yb,S_['ae_cha']),4))
