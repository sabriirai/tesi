import uproot, numpy as np, matplotlib.pyplot as plt

path = "/eos/lhcb/grid/prod/lhcb/anaprod/lhcb/LHCb/Collision24/FTUPLE.ROOT/00412689/0000/00412689_00000006_1.ftuple.root"
f = uproot.open(path)
t = f["SpruceRD_B0ToHpHmMuMu/DecayTree"]
hlt1 = ["B_Hlt1TrackMVADecision_TOS", "B_Hlt1TrackMuonMVADecision_TOS", "B_Hlt1TwoTrackMVADecision_TOS"]
hlt2 = ["B_Hlt2_InclDetDiMuonDecision_TOS", "B_Hlt2_InclDetDiMuon_3BodyDecision_TOS",
      "B_Hlt2_InclDetDiMuon_4BodyDecision_TOS"]

d = t.arrays(["B_MASS", "Jps_MASS", "Phi_MASS", "H1_PT", "H2_PT", "L1_PT", "L2_PT",
              "H1_PROBNN_K", "H2_PROBNN_K", "B_BPVDIRA", "B_CHI2NDOF", "H1_PID_K", "H2_PID_K", "L1_PROBNN_MU", "L2_PROBNN_MU", "L1_PID_MU", "L2_PID_MU"] + hlt1 + hlt2,
             library="np", entry_stop=200000)

print("eventi:", t.num_entries)

# nomi che compaiono nelle selezioni: quali esistono davvero?
voluti = ["Phi_MASS", "B_MVA", "B_QSQ", "TURBO_BLOCK", "H1_PROBNN_K", "L1_PT", "B_PT"]
for nome in voluti:
    print(nome, "OK" if nome in t.keys() else "MANCA")

# tutti i branch con MASS nel nome
print([k for k in t.keys() if "MASS" in k])


mask = np.abs(d["Phi_MASS"] - 1019.46) < 12     # dal file di testo. applico selezione sulla massa del phi e vedo che effetto ha su massaB e massaJpsi

# mini cut-flow: aggiungo un cut alla volta
c1 = np.abs(d["Phi_MASS"] - 1019.46) < 12              # finestra sul phi
c2 = np.minimum(d["H1_PT"], d["H2_PT"]) > 250          # PT minimo dei due kaoni
c3 = np.minimum(d["L1_PT"], d["L2_PT"]) > 100          # PT minimo dei due muoni
c4 = np.minimum(d["H1_PROBNN_K"], d["H2_PROBNN_K"]) > 0.025   # pid
c5 = d["B_BPVDIRA"] > 0.9995                                   # vertice (fid)
c6 = d["B_CHI2NDOF"] < 20                                      # qualità del fit (fid)
c7 = np.minimum(d["H1_PID_K"], d["H2_PID_K"]) > -10.0              # PID dei kaoni
c8 = np.minimum(d["L1_PROBNN_MU"], d["L2_PROBNN_MU"]) > 0.05       # ProbNN dei muoni
c9 = np.minimum(d["L1_PID_MU"], d["L2_PID_MU"]) > -3.0             # PID dei muoni
pid_tutti = c4 & c7 & c8 & c9                                      # c4 = ProbNN_K, già definito


n = len(d["B_MASS"])                                   # candidati di partenza
print("partenza:             ", n)
print("solo finestra phi:    ", c1.sum(), f"({c1.mean():.1%})")
print("solo PT kaoni:        ", c2.sum(), f"({c2.mean():.1%})")
print("solo PT muoni:        ", c3.sum(), f"({c3.mean():.1%})")
print("phi + PT kaoni:       ", (c1 & c2).sum(), f"({(c1 & c2).mean():.1%})")
print("phi + kaoni + muoni:  ", (c1 & c2 & c3).sum(), f"({(c1 & c2 & c3).mean():.1%})")
print("solo PROBNN_K:      ", c4.sum(), f"({c4.mean():.1%})")
print("solo DIRA:          ", c5.sum(), f"({c5.mean():.1%})")
print("solo CHI2NDOF:      ", c6.sum(), f"({c6.mean():.1%})")
print("tutti insieme + phi:", (c1 & c4 & c5 & c6).sum(), f"({(c1 & c4 & c5 & c6).mean():.1%})")
print("solo PID_K:          ", c7.sum(), f"({c7.mean():.1%})")
print("solo PROBNN_MU:      ", c8.sum(), f"({c8.mean():.1%})")
print("solo PID_MU:         ", c9.sum(), f"({c9.mean():.1%})")
print("gruppo pid completo: ", pid_tutti.sum(), f"({pid_tutti.mean():.1%})")
print("pid + phi:           ", (pid_tutti & c1).sum(), f"({(pid_tutti & c1).mean():.1%})")

# trigger: ogni decisione da sola, poi OR e AND dentro ciascun gruppo
for nome in hlt1 + hlt2:
    print(f"{nome:42s}", f"{d[nome].astype(bool).mean():.1%}")

tr1_or  = np.any([d[n].astype(bool) for n in hlt1], axis=0)
tr1_and = np.all([d[n].astype(bool) for n in hlt1], axis=0)
tr2_or  = np.any([d[n].astype(bool) for n in hlt2], axis=0)
tr2_and = np.all([d[n].astype(bool) for n in hlt2], axis=0)

print("Hlt1 in OR: ", f"{tr1_or.mean():.1%}", "  Hlt1 in AND:", f"{tr1_and.mean():.1%}")
print("Hlt2 in OR: ", f"{tr2_or.mean():.1%}", "  Hlt2 in AND:", f"{tr2_and.mean():.1%}")
print("tutto (OR) + phi + PROBNN_K:", f"{(c1 & c4 & tr1_or & tr2_or).mean():.1%}")

tutti = c1 & c4 & tr1_or & tr2_or
jpsi  = np.abs(d["Jps_MASS"] - 3097) < 100     # finestra larga attorno al J/psi (valore d'esempio)

tutti = c1 & c2 & c3 & c5 & c6 & pid_tutti & tr1_or & tr2_or # tutti i tagli messi insieme
k = tutti.sum()              # quanti passano
eff = k / n                  # efficienza
err = np.sqrt(eff * (1 - eff) / n)   # errore

print("passano:", k, "su", n)
print(f"efficienza = {eff:.4f} +- {err:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(d["Jps_MASS"], bins=80, range=(2500, 3500), histtype="step", label="prima")
ax[0].hist(d["Jps_MASS"][tutti], bins=80, range=(2500, 3500), histtype="step", label="dopo i cut")
ax[0].set_xlabel("Jps_MASS [MeV]"); ax[0].legend()
ax[1].hist(d["B_MASS"][tutti], bins=80, range=(5100, 5700), histtype="step", label="dopo i cut")
ax[1].hist(d["B_MASS"][tutti & jpsi], bins=80, range=(5100, 5700), histtype="step", label="dopo i cut + J/psi")
ax[1].set_xlabel("B_MASS [MeV]"); ax[1].legend()
fig.savefig("dati_spruce.png")