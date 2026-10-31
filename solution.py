# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold

# ----------------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------------
SEARCH_DIRS = ["/kaggle/input", "."]          # add your local data folder here if not on Kaggle
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
SEEDS = [0, 1, 2, 3, 4]
RUN_CV = True                                 # profile-grouped validation (prints a realistic score)


def find_files(names):
    found = {}
    for root in SEARCH_DIRS:
        for dp, _, fns in os.walk(root):
            for fn in fns:
                for key in names:
                    if key in fn.lower() and key not in found:
                        found[key] = os.path.join(dp, fn)
    return found


F = find_files(["train_part1", "train_part2", "test.csv", "sample_submission"])
print(F)

# ----------------------------------------------------------------------------------
# LOAD (train is split over two xlsx files)
# ----------------------------------------------------------------------------------
def read_any(p):
    return pd.read_excel(p) if p.lower().endswith((".xlsx", ".xls")) else pd.read_csv(p)

parts = [read_any(F[k]) for k in ["train_part1", "train_part2"] if k in F]
train = pd.concat(parts, ignore_index=True)
test = pd.read_csv(F["test.csv"])
print("train", train.shape, "test", test.shape)

CLASSES = np.array(["Availability", "Benign", "Reconnaissance", "Spoofing"])
canon = {c.lower(): c for c in CLASSES}
y = train["Target"].astype(str).str.strip().str.lower().map(canon)
assert y.notna().all(), "unexpected label"
y = pd.Categorical(y, categories=CLASSES).codes


# ----------------------------------------------------------------------------------
# FEATURES
# The test set contains traffic "profiles" never seen in training (e.g. SYN floods that get no
# reply, nmap scans with other TCP windows). Raw victim-dependent / tool-specific values
# (reply counts, exact ports, exact window sizes) are therefore coarsened so the model learns
# the *behaviour* (probe vs payload, flags, service, volume) instead of memorising a profile.
# ----------------------------------------------------------------------------------
def clip_log(s):
    return np.log1p(s.clip(lower=0))


def build(tr, te, mode="coarse"):
    full = pd.concat([tr.drop(columns=["Target"], errors="ignore"), te], keys=["tr", "te"])
    full = full.drop(columns=["id"]).reset_index(level=0).rename(columns={"level_0": "_s"})
    s = full.pop("_s").values
    d = pd.DataFrame(index=full.index)
    d["proto"] = full.proto.astype(str).str.lower().map({"tcp": 0, "udp": 1, "icmp": 2}).fillna(3)
    svc = {"-": 0, "dns": 1, "mqtt": 2, "ssl": 3, "http": 4, "ntp": 5, "irc": 6, "dhcp": 7, "radius": 8}
    d["service"] = full.service.astype(str).str.lower().map(svc).fillna(9)
    p = full["id.resp_p"]
    d["port_zone"] = np.select([p < 1024, p < 49152], [0, 1], 2)
    d["port_low"] = (p < 1024).astype(int)
    for q in [21, 53, 80, 443, 1883, 5353, 123, 137]:
        d[f"p{q}"] = (p == q).astype(int)
    d["same_port"] = (full["id.orig_p"] == p).astype(int)
    f, b = full.fwd_pkts_tot, full.bwd_pkts_tot
    d["fwd_pkts_l"] = clip_log(f); d["bwd_pkts_l"] = clip_log(b); d["tot_pkts_l"] = clip_log(f + b)
    d["has_reply"] = (b > 0).astype(int)
    d["fwd_data_l"] = clip_log(full.fwd_data_pkts_tot); d["bwd_data_l"] = clip_log(full.bwd_data_pkts_tot)
    for c in ["flow_FIN_flag_count", "flow_SYN_flag_count", "flow_RST_flag_count", "fwd_PSH_flag_count",
              "bwd_PSH_flag_count", "flow_ACK_flag_count", "fwd_URG_flag_count"]:
        d[c] = full[c].clip(upper=3)
    d["xmas_like"] = ((full.fwd_URG_flag_count > 0) |
                      ((full.flow_FIN_flag_count > 0) & (full.fwd_PSH_flag_count > 0) &
                       (full.flow_ACK_flag_count == 0))).astype(int)
    d["syn_noack"] = ((full.flow_SYN_flag_count > 0) & (full.flow_ACK_flag_count == 0)).astype(int)
    d["syn_probe"] = ((f == 1) & (full["fwd_pkts_payload.tot"] == 0)).astype(int)
    d["syn_payload"] = ((f == 1) & (full["fwd_pkts_payload.tot"] > 0) & (full.flow_SYN_flag_count > 0)).astype(int)
    for c in ["fwd_pkts_payload.tot", "bwd_pkts_payload.tot", "fwd_pkts_payload.max", "bwd_pkts_payload.max",
              "fwd_pkts_payload.avg", "bwd_pkts_payload.avg"]:
        d[c.replace(".", "_") + "_l"] = clip_log(full[c])
    d["fwd_payload_zero"] = (full["fwd_pkts_payload.tot"] == 0).astype(int)
    d["dur_l"] = clip_log(full.flow_duration * 1e3)
    d["fwd_iat_avg_l"] = clip_log(full["fwd_iat.avg"] * 1e3)
    d["flow_iat_avg_l"] = clip_log(full["flow_iat.avg"] * 1e3)
    d["flow_iat_max_l"] = clip_log(full["flow_iat.max"] * 1e3)
    d["active_l"] = clip_log(full["active.tot"] * 1e3)
    d["idle_l"] = clip_log(full["idle.tot"] * 1e3)
    d["down_up"] = full.down_up_ratio.clip(upper=10)
    d["bytes_ratio"] = np.log((full["fwd_pkts_payload.tot"] + 1) / (full["bwd_pkts_payload.tot"] + 1))
    w, bw = full.fwd_init_window_size, full.bwd_init_window_size
    d["win_zero"] = (w == 0).astype(int); d["win_large"] = (w >= 60000).astype(int)
    d["bwin_large"] = (bw >= 60000).astype(int); d["bwin_zero"] = (bw == 0).astype(int)
    if mode == "full":  # adds exact window / header values (second view for the ensemble)
        d["win_raw"] = w
        d["fhdr"] = full.fwd_header_size_tot
        d["bhdr"] = full.bwd_header_size_tot
    X = d.astype(float)
    return X[s == "tr"].reset_index(drop=True), X[s == "te"].reset_index(drop=True)


def profile_signature(df):
    """Coarse traffic-profile id used to build 'unseen profile' validation folds."""
    p = df["id.resp_p"]
    pb = np.where(p.isin([21, 53, 80, 443, 1883, 5353, 123, 137, 67]), p.astype(int).astype(str),
                  np.where(p < 1024, "lo", "hi"))
    w = df.fwd_init_window_size
    wb = np.select([w == 0, w == 64, w <= 2048, w < 60000, w < 65535], ["w0", "w64", "wsmall", "wmid", "wlarge"], "wmax")
    return (df.proto.astype(str) + "/" + df.service.astype(str) + "/" + pb + "/" + wb +
            "/f" + df.flow_FIN_flag_count.clip(upper=1).astype(int).astype(str) +
            "u" + df.fwd_URG_flag_count.clip(upper=1).astype(int).astype(str) +
            "/b" + df.bwd_pkts_tot.clip(upper=2).astype(int).astype(str)).values


def make_model(seed):
    return lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20,
        class_weight="balanced", subsample=0.7, subsample_freq=1, colsample_bytree=0.5,
        reg_lambda=5, random_state=seed, n_jobs=-1, verbose=-1)


views = {m: build(train, test, m) for m in ["coarse", "full"]}

# ----------------------------------------------------------------------------------
# VALIDATION: hold out whole traffic profiles (mimics the hidden "unseen profile" test)
# ----------------------------------------------------------------------------------
if RUN_CV:
    sg = profile_signature(train)
    g = pd.factorize(sg)[0]
    giant = pd.Series(sg).value_counts().index[0]          # one huge flood profile -> split randomly
    gi = sg == giant
    g[gi] = 10000 + np.random.RandomState(0).randint(0, 10, gi.sum())
    X0 = views["coarse"][0]
    oof = np.zeros((len(X0), 4))
    for tri, vai in StratifiedGroupKFold(5, shuffle=True, random_state=1).split(X0, y, g):
        for mode, (Xa, _) in views.items():
            oof[vai] += make_model(0).fit(Xa.iloc[tri], y[tri]).predict_proba(Xa.iloc[vai]) / len(views)
    print("Profile-grouped CV macro-F1:", round(f1_score(y, oof.argmax(1), average="macro"), 5))
    print(classification_report(y, oof.argmax(1), target_names=CLASSES, digits=4))

# ----------------------------------------------------------------------------------
# FINAL FIT + PREDICT (ensemble of 2 feature views x several seeds)
# ----------------------------------------------------------------------------------
P = np.zeros((len(test), 4))
for mode, (Xa, Xt) in views.items():
    for sd in SEEDS:
        P += make_model(sd).fit(Xa, y).predict_proba(Xt)
P /= (len(views) * len(SEEDS))

sub = pd.DataFrame({"id": test["id"].values, "Target": CLASSES[P.argmax(1)]})
if "sample_submission" in F:
    sample = pd.read_csv(F["sample_submission"])
    assert len(sub) == len(sample) and set(sub["id"]) == set(sample["id"]), "id mismatch with sample"
    sub = sample[["id"]].merge(sub, on="id", how="left")
assert sub["Target"].isin(CLASSES).all()
out_path = os.path.join(OUT_DIR, "submission.csv")
sub.to_csv(out_path, index=False)
print("Saved", out_path, sub.shape)
print(sub["Target"].value_counts())
