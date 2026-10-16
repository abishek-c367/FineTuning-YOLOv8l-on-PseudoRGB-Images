"""T08 (zer0-data/GATE-IR): is the WeatherGate / RainRemover trained? Run the gate on synthetic clear vs blurred-low-contrast
vs streaky inputs with its shipped (random) initialization, and inspect what FogEnhancer/RainRemover do numerically."""
import _common, sys, torch, json, torch.nn.functional as F
from _common import env_info, save, ZREPO
sys.path.insert(0, ZREPO)
from gate.weather_gate import WeatherGate
from preprocessing.fog_enhancer import FogEnhancer
from preprocessing.rain_remover import RainRemover
res = {"env": env_info()}
torch.manual_seed(0)
g = WeatherGate().eval()
res["gate_params"] = sum(p.numel() for p in g.parameters())
res["gate_has_pretrained_weights_in_repo"] = False  # no .pt/.pth and no training script for the gate in the repo (grep)
base = torch.rand(1, 1, 128, 160)
clear = F.avg_pool2d(base, 3, 1, 1)
fog = F.avg_pool2d(base, 15, 1, 7) * 0.3 + 0.35
rain = (clear + (torch.rand_like(clear) > 0.97).float() * 0.5).clamp(0, 1)
outs = {}
for name, im in [("clear", clear), ("fog", fog), ("rain", rain)]:
    with torch.no_grad():
        cid, probs, feats = g(im, return_probs=True, return_features=True, is_normalized=True)
    outs[name] = {"class": int(cid), "probs": [round(float(p), 3) for p in probs[0]], "features": [round(float(f), 4) for f in feats[0]]}
res["gate_outputs_random_init"] = outs
fe = FogEnhancer()
with torch.no_grad():
    ef = fe(fog)
res["fog_enhancer"] = {"in_mean": float(fog.mean()), "out_mean": float(ef.mean()), "in_std": float(fog.std()), "out_std": float(ef.std()), "gamma": float(fe.compute_adaptive_gamma(fog).flatten()[0])}
rr = RainRemover().eval()
res["rain_remover_params"] = sum(p.numel() for p in rr.parameters())
with torch.no_grad():
    er = rr(rain)
res["rain_remover_random_init"] = {"in_mean": float(rain.mean()), "out_mean": float(er.mean()), "out_min": float(er.min()), "out_max": float(er.max()), "mean_abs_change": float((er - rain).abs().mean())}
save("T08_zer0_gate.json", res); print(json.dumps({k: v for k, v in res.items() if k != 'env'}, indent=1))
