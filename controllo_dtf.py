import uproot, numpy as np, matplotlib.pyplot as plt

path = "/eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/FTUPLE.ROOT/00412801/0000/00412801_00000001_1.ftuple.root"
f = uproot.open(path)
t = f["SpruceRD_B0ToHpHmMuMu/DecayTree"]
hlt1 = ["B_Hlt1TrackMVADecision_TOS", "B_Hlt1TrackMuonMVADecision_TOS", "B_Hlt1TwoTrackMVADecision_TOS"] # contiene i nomi delle tre decisioni di trigger hlt1
hlt2 = ["B_Hlt2_InclDetDiMuonDecision_TOS", "B_Hlt2_InclDetDiMuon_3BodyDecision_TOS",
      "B_Hlt2_InclDetDiMuon_4BodyDecision_TOS"] # contiene i nomi delle tre decisioni di trigger di hlt2


print([k for k in t.keys() if "DTF" in k and "MASS" in k])

print(np.allclose(d["Jps_MASS"], d["B_DTF_PV_Jps_MASS"]))
print(np.allclose(d["Jps_MASS"], d["B_DTF_HEAD_Jps_MASS"]))

# print([k for k in t.keys() if k.startswith("L1_P") or k.startswith("L1_E")])
E  = d["L1_ENERGY"].astype(float) + d["L2_ENERGY"]
px = d["L1_PX"].astype(float) + d["L2_PX"]
py = d["L1_PY"].astype(float) + d["L2_PY"]
pz = d["L1_PZ"].astype(float) + d["L2_PZ"]
m_mumu = np.sqrt(E**2 - px**2 - py**2 - pz**2)

# fa jps_mass - ricacolata e pv - ricalcolata e poi calcola media e dev std
diff_nodtf = d["Jps_MASS"] - m_mumu
diff_dtf   = d["B_DTF_PV_Jps_MASS"] - m_mumu
for nome, diff in [("Jps_MASS - ricalcolata", diff_nodtf),
                   ("DTF_PV - ricalcolata  ", diff_dtf)]:
    print(f"{nome}: media = {diff.mean():.4f} MeV, std = {diff.std():.3f} MeV, "
          f"entro 0.5 MeV: {(np.abs(diff) < 0.5).mean():.1%}")

masse = {
    "Jps_MASS":            d["Jps_MASS"],
    "ricalcolata (muoni)": m_mumu,
    "B_DTF_PV_Jps_MASS":   d["B_DTF_PV_Jps_MASS"],
    "B_DTF_HEAD_Jps_MASS": d["B_DTF_HEAD_Jps_MASS"],
    "B_DTF_Jps_Jps_MASS":  d["B_DTF_Jps_Jps_MASS"],
    "B_DTF_Psi_Jps_MASS":  d["B_DTF_Psi_Jps_MASS"],
} 

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for nome, m in masse.items():
    ax[0].hist(m, bins=200, range=(2900, 3800), histtype="step", label=nome)  # vista larga
    ax[1].hist(m, bins=100, range=(3000, 3200), histtype="step", label=nome)  # zoom sulla J/psi
ax[0].set_yscale("log")
for a in ax:
    a.set_xlabel("m(mumu) [MeV]")
    a.legend(fontsize=8)
fig.tight_layout()
fig.savefig("confronto_masse_dimuone.png")

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# sinistra: le tre masse, blu spessa sotto e arancione tratteggiata sopra
r = (3000, 3200)
ax[0].hist(d["Jps_MASS"], bins=100, range=r, histtype="step", lw=4, alpha=0.5, label="Jps_MASS")
ax[0].hist(m_mumu, bins=100, range=r, histtype="step", ls="--", label="ricalcolata (muoni)")
ax[0].hist(d["B_DTF_PV_Jps_MASS"], bins=100, range=r, histtype="step", label="B_DTF_PV_Jps_MASS")
ax[0].set_xlabel("m(mumu) [MeV]"); ax[0].legend(fontsize=8)

# destra: differenze candidato per candidato rispetto alla ricalcolata
ax[1].hist(d["Jps_MASS"] - m_mumu, bins=100, range=(-5, 5), histtype="step", label="Jps_MASS - ricalcolata")
ax[1].hist(d["B_DTF_PV_Jps_MASS"] - m_mumu, bins=100, range=(-5, 5), histtype="step", label="DTF_PV - ricalcolata")
ax[1].set_xlabel("differenza [MeV]"); ax[1].set_yscale("log"); ax[1].legend(fontsize=8)

fig.tight_layout()
fig.savefig("confronto_ricalcolata.png")