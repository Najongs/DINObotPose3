"""
Situation router (Phase 0 / N1). Fit a tiny logistic on the FROZEN DINOv3 global feature (dumped by
selfbbox_eval.py --dump-npz) to predict, per frame, whether the crop pose is a depth-FAILURE
(ADD > fail_thr) that render-compare can help. Render-compare is mask-accuracy-capped (~40mm) and
only helps frames WORSE than the mask; routing ONLY the predicted-hard tail to the expensive RC
(and leaving the ~0.94-AUROC-confident-good frames at the fast solver) concentrates the lever and
protects the bulk — the upstream fix for the accept-margin cap.

Fits on a contiguous adapt split (first --adapt-frac), reports held-out AUROC vs the reproj baseline,
and writes a route file (fid -> P(fail), is_eval) consumed cross-env by CtRNet-X/rc_refine_routed.py.
Reused by Phase A (routed render-compare) and Phase B (route the silhouette selector to the tail).
"""
import argparse
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True, help='selfbbox dump with feat, reproj, kp_cam, gt3d, fid')
    ap.add_argument('--out-pkl', required=True)
    ap.add_argument('--out-route', required=True)
    ap.add_argument('--fail-thr', type=float, default=0.05,
                    help='ADD (m) above which a frame is a depth-failure render-compare can help')
    ap.add_argument('--adapt-frac', type=float, default=0.7)
    ap.add_argument('--use-reproj', action='store_true', help='append log reproj as a 2nd feature')
    a = ap.parse_args()

    d = np.load(a.npz)
    feat, reproj, fid = d['feat'], d['reproj'], d['fid']
    kp_cam, gt3d = d['kp_cam'], d['gt3d']
    valid = (np.abs(gt3d).sum(-1) > 0)                 # (N,7)
    err = np.linalg.norm(kp_cam - gt3d, axis=-1)       # (N,7)
    add = np.array([err[i][valid[i]].mean() if valid[i].any() else np.nan for i in range(len(fid))])
    ok = ~np.isnan(add)
    feat, reproj, fid, add = feat[ok], reproj[ok], fid[ok], add[ok]
    y = (add > a.fail_thr).astype(int)
    N = len(fid); na = int(N * a.adapt_frac)
    tr = np.arange(na); te = np.arange(na, N)

    X = np.concatenate([feat, np.log1p(reproj)[:, None]], axis=1) if a.use_reproj else feat
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, class_weight='balanced').fit(sc.transform(X[tr]), y[tr])
    score = clf.predict_proba(sc.transform(X))[:, 1]

    enough = y[te].sum() >= 3 and (len(te) - y[te].sum()) >= 3
    auc = roc_auc_score(y[te], score[te]) if enough else float('nan')
    rp_auc = roc_auc_score(y[te], reproj[te]) if enough else float('nan')
    print(f"[router] N={N} fail-rate={y.mean()*100:.1f}% (thr={a.fail_thr*1000:.0f}mm)  "
          f"adapt={na} eval={len(te)}")
    print(f"[router] held-out AUROC  DINOv3-feat={auc:.3f}   baseline-reproj={rp_auc:.3f}")

    joblib.dump({'scaler': sc, 'clf': clf, 'use_reproj': a.use_reproj, 'fail_thr': a.fail_thr}, a.out_pkl)
    is_eval = np.zeros(N, dtype=bool); is_eval[te] = True
    np.savez(a.out_route, fid=fid, score=score, is_eval=is_eval, add=add)
    print(f"[router] saved {a.out_pkl} + {a.out_route}")


if __name__ == '__main__':
    main()
