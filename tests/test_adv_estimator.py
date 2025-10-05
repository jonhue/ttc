import numpy as np, torch
from scipy.special import comb
from verl.trainer.ppo.core_algos import _calc_passk_adv, compute_passk_combined_outcome_advantage

def ref_calc_adv(val, k):
    val = np.asarray(val)
    c = np.sum(val==1)
    n = len(val)
    rho = 1 - comb(n-c, k) / comb(n, k) if comb(n,k)>0 else 0.0
    sigma = np.sqrt(max(rho*(1-rho), 0.0))
    adv_p = (1 - rho) / (sigma + 1e-6)
    denom = comb(n-1, k-1)
    term = comb(n-c-1, k-1)/denom if denom>0 else 0.0
    adv_n = (1 - rho - term) / (sigma + 1e-6)
    out = np.where(val==1, adv_p, adv_n)
    return out.astype(np.float32)

rng = np.random.default_rng(0)
for n in [2,3,4,8]:
  for k in [1,2,3,4]:
    vals = (rng.random(n) < 0.4).astype(np.int32)
    a_ref = ref_calc_adv(vals, min(k,n))
    a_mine = _calc_passk_adv(vals, k)
    assert np.allclose(a_ref, a_mine, rtol=1e-5, atol=1e-5)
print("Pass@k term parity: OK")

# Combined check on a fake batch
uid = np.array(['u1']*4 + ['u2']*3, dtype=object)
scores = np.array([1,0,1,0, 0,1,0], dtype=np.float32)  # >0 means pass
tlr = torch.zeros((len(scores), 5), dtype=torch.float32)  # last token holds outcome
tlr[:, -1] = torch.from_numpy(scores)
mask = torch.zeros_like(tlr); mask[:, -1] = 1
adv, ret = compute_passk_combined_outcome_advantage(tlr, mask, uid, k=2)
assert torch.allclose(adv, ret)
assert adv.shape == mask.shape and (adv[:, :-1]==0).all()
print("Combined strategy shape/broadcasting: OK")
