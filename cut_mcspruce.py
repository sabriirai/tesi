import uproot, numpy as np, matplotlib.pyplot as plt

path = "/eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/FTUPLE.ROOT/00412801/0000/00412801_00000001_1.ftuple.root"
f = uproot.open(path)
t = f["SpruceRD_B0ToHpHmMuMu/DecayTree"]
hlt1 = ["B_Hlt1TrackMVADecision_TOS", "B_Hlt1TrackMuonMVADecision_TOS", "B_Hlt1TwoTrackMVADecision_TOS"] # contiene i nomi delle tre decisioni di trigger hlt1
hlt2 = ["B_Hlt2_InclDetDiMuonDecision_TOS", "B_Hlt2_InclDetDiMuon_3BodyDecision_TOS",
      "B_Hlt2_InclDetDiMuon_4BodyDecision_TOS"] # contiene i nomi delle tre decisioni di trigger di hlt2
branches = [
    # ============================================================
        # CANDIDATO B_s
        # ============================================================
        "B_MASS",          # massa ricostruita del candidato B_s
        "B_PT",            # impulso trasverso del B_s
        "B_P",             # modulo dell'impulso del B_s

        "B_BPVDIRA",       # allineamento tra direzione del B_s e PV -> vertice
        "B_BPVFDCHI2",     # significativita della distanza di volo dal PV
        "B_BPVIPCHI2",     # IP chi2 del B_s rispetto al PV
        "B_CHI2NDOF",      # qualita del vertice del B_s

        # ============================================================
        # SISTEMA DIMUONICO: J/psi
        # ============================================================
        "Jps_MASS",        # massa dimuone senza DTF
        "Jps_PT",          # impulso trasverso del sistema dimuonico
        "Jps_DOCACHI2",    # DOCA chi2 tra i due muoni
        "Jps_CHI2NDOF",    # qualita del vertice dimuonico
        "Jps_BPVFDCHI2",   # significativita della distanza di volo dal PV

        # Massa del dimuone con diverse configurazioni DTF
        "B_DTF_PV_Jps_MASS",      # DTF con vincolo al PV
        "B_DTF_HEAD_Jps_MASS",    # DTF con vincolo sulla massa del B_s
        "B_DTF_Jps_Jps_MASS",     # DTF con vincolo sulla massa della J/psi
        "B_DTF_Psi_Jps_MASS",     # DTF con vincolo sulla massa della psi(2S)

        # ============================================================
        # CANDIDATO PHI -> K+ K-
        # ============================================================
        "Phi_MASS",        # massa invariante dei due kaoni
        "Phi_PT",          # impulso trasverso della phi
        "Phi_CHI2NDOF",    # qualita del vertice della phi
        "Phi_DOCACHI2",    # DOCA chi2 tra i due kaoni
        "Phi_DOCA",        # distanza di massimo avvicinamento dei due kaoni
        "Phi_BPVIPCHI2",   # IP chi2 della phi rispetto al PV

        # ============================================================
        # PRIMO KAONE: H1
        # ============================================================
        "H1_PT",             # impulso trasverso
        "H1_P",              # modulo dell'impulso
        "H1_ETA",            # pseudorapidita
        "H1_BPVIPCHI2",      # IP chi2 rispetto al PV
        "H1_CHI2NDOF",       # qualita della traccia
        "H1_PROBNN_GHOST",   # probabilita che la traccia sia ghost
        "H1_PROBNN_K",       # probabilita neurale di essere un kaone
        "H1_PID_K",          # identificazione come kaone

        # ============================================================
        # SECONDO KAONE: H2
        # ============================================================
        "H2_PT",
        "H2_P",
        "H2_ETA",
        "H2_BPVIPCHI2",
        "H2_CHI2NDOF",
        "H2_PROBNN_GHOST",
        "H2_PROBNN_K",
        "H2_PID_K",

        # ============================================================
        # PRIMO MUONE: L1
        # ============================================================
        "L1_PT",             # impulso trasverso
        "L1_P",              # modulo dell'impulso
        "L1_ETA",            # pseudorapidita
        "L1_BPVIPCHI2",      # IP chi2 rispetto al PV
        "L1_CHI2NDOF",       # qualita della traccia
        "L1_PROBNN_GHOST",   # probabilita che la traccia sia ghost
        "L1_PROBNN_MU",      # probabilita neurale di essere un muone
        "L1_PID_MU",         # identificazione come muone

        # Componenti del quadrimpulso, usate per ricalcolare m(mu mu)
        "L1_PX",
        "L1_PY",
        "L1_PZ",
        "L1_ENERGY",

        # ============================================================
        # SECONDO MUONE: L2
        # ============================================================
        "L2_PT",
        "L2_P",
        "L2_ETA",
        "L2_BPVIPCHI2",
        "L2_CHI2NDOF",
        "L2_PROBNN_GHOST",
        "L2_PROBNN_MU",
        "L2_PID_MU",

        # Componenti del quadrimpulso, usate per ricalcolare m(mu mu)
        "L2_PX",
        "L2_PY",
        "L2_PZ",
        "L2_ENERGY",
]

d = t.arrays(
    branches + hlt1 + hlt2,
    library="np",
    entry_stop=200000
) 

print("eventi:", t.num_entries)

# controllo quali variabili esistono 
voluti = ["Phi_MASS", "B_MVA", "B_QSQ", "TURBO_BLOCK", "H1_PROBNN_K", "L1_PT", "B_PT"]
for nome in voluti:
    print(nome, "OK" if nome in t.keys() else "MANCA")

# stampa tutti i branch con MASS nel nome
# print([k for k in t.keys() if "MASS" in k])


# aggiungo un cut alla volta
c1 = np.abs(d["Phi_MASS"] - 1019.46) < 12              # cut sul phi (i due K vengono dal phi solo se la loro massa inv è vicina alla massa del phi)
c2 = np.minimum(d["H1_PT"], d["H2_PT"]) > 250          # PT minimo dei due kaoni (cioè entrambi sopra 250)
c3 = np.minimum(d["L1_PT"], d["L2_PT"]) > 100          # PT minimo dei due muoni
c4 = np.minimum(d["H1_PROBNN_K"], d["H2_PROBNN_K"]) > 0.025   # pid
c5 = d["B_BPVDIRA"] > 0.9995                                   # richiede che il Bs punti bene verso il PV da cui proviene
c6 = d["B_CHI2NDOF"] < 20                                       # qualità del vertice ricostruito del Bs
c7 = np.minimum(d["H1_PID_K"], d["H2_PID_K"]) > -10.0              # PID dei kaoni
c8 = np.minimum(d["L1_PROBNN_MU"], d["L2_PROBNN_MU"]) > 0.05       # ProbNN dei muoni
c9 = np.minimum(d["L1_PID_MU"], d["L2_PID_MU"]) > -3.0             # PID dei muoni
pid_tutti = c4 & c7 & c8 & c9

c10 = np.minimum(d["H1_P"], d["H2_P"]) > 0.0                        # impulso totale dei due kaoni
c11 = np.minimum(np.abs(d["H1_ETA"]), np.abs(d["H2_ETA"])) > 1.0    # eta minima dei kaoni
c12 = np.maximum(np.abs(d["H1_ETA"]), np.abs(d["H2_ETA"])) < 6.0    # eta massima dei kaoni
c13 = np.minimum(d["H1_BPVIPCHI2"], d["H2_BPVIPCHI2"]) > 6.0        # IPCHI2 dei kaoni (misura quanto una traccia è incompatibile col venire dal PV)
c14 = np.maximum(d["H1_CHI2NDOF"], d["H2_CHI2NDOF"]) < 4.0          # qualità delle tracce dei kaoni
c15 = np.maximum(d["H1_PROBNN_GHOST"], d["H2_PROBNN_GHOST"]) < 0.5  # probabilità di traccia ghost

# tagli aggiuntivi sui muoni
c16 = np.minimum(d["L1_P"], d["L2_P"]) > 0.0                        # impulso totale dei due muoni
c17 = np.minimum(np.abs(d["L1_ETA"]), np.abs(d["L2_ETA"])) > 1.0    # eta minima dei muoni
c18 = np.maximum(np.abs(d["L1_ETA"]), np.abs(d["L2_ETA"])) < 6.0    # eta massima dei muoni
c19 = np.minimum(d["L1_BPVIPCHI2"], d["L2_BPVIPCHI2"]) > 2.0        # IPCHI2 dei muoni
c20 = np.maximum(d["L1_CHI2NDOF"], d["L2_CHI2NDOF"]) < 4.0          # qualità delle tracce dei muoni
c21 = np.maximum(d["L1_PROBNN_GHOST"], d["L2_PROBNN_GHOST"]) < 0.5  # probabilità di traccia ghost

# tagli aggiuntivi su B_s
c22 = d["B_PT"] > 0.0                    # impulso trasverso del B_s positivo
c23 = d["B_P"] > 0.0                     # impulso totale del B_s positivo
c24 = d["B_PT"] < 100000.0               # limite superiore sul PT del B_s
c25 = d["B_BPVFDCHI2"] > 10.0            # significatività della distanza di volo dal PV
c26 = d["B_BPVIPCHI2"] < 100000.0        # IPCHI2 del B_s rispetto al PV

# tagli sul sistema dimuonico Jps -> mu+ mu-
c27 = d["Jps_PT"] > 0.0               # impulso trasverso del sistema dimuonico
c28 = d["Jps_DOCACHI2"] < 36.0        # compatibilità geometrica tra le due tracce dei muoni
c29 = d["Jps_CHI2NDOF"] < 30.0        # qualità del vertice formato dai due muoni
c30 = d["Jps_BPVFDCHI2"] > 0.0        # distanza di volo del sistema dimuonico rispetto al PV

# tagli sul candidato Phi -> K+ K-
c31 = d["Phi_PT"] > 600.0             # impulso trasverso della Phi maggiore di 600 MeV
c32 = d["Phi_CHI2NDOF"] < 16.0        # qualità del vertice formato dai due kaoni
c33 = d["Phi_DOCACHI2"] < 10.0        # compatibilità geometrica tra le due tracce dei kaoni
c34 = d["Phi_DOCA"] < 0.5             # distanza minima tra le due tracce dei kaoni
c35 = d["Phi_BPVIPCHI2"] > 0.0        # IP chi2 della Phi rispetto al PV

# Il taglio TURBO_BLOCK == 2 non viene applicato:
# il branch TURBO_BLOCK non e presente nella tupla.                       
                                    
q2 = d["Jps_MASS"]**2 / 1e6 # q^2 senza DTF. 1e6 serve per trasformare da mev2 a gev2
q2_dtf = d["B_DTF_HEAD_Jps_MASS"]**2 / 1e6 # q^2 con DTF HEAD
cq2 = (q2 > 0) & (q2 < 19) # taglio 0 < q^2 < 19 GeV^2 (per entrambi)
cq2_dtf = (q2_dtf > 0) & (q2_dtf < 19)

n = len(d["B_MASS"])                                   # candidati di partenza
print("partenza:             ", n)
print("solo finestra phi:    ", c1.sum(), f"({c1.mean():.1%})")
print("solo PT kaoni:        ", c2.sum(), f"({c2.mean():.1%})")
print("solo PT muoni:        ", c3.sum(), f"({c3.mean():.1%})")
print("phi + PT kaoni:       ", (c1 & c2).sum(), f"({(c1 & c2).mean():.1%})") # (c1 & c2).sum() quanti passano entrambi, mean è la frazione 
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

tr1_or  = np.any([d[n].astype(bool) for n in hlt1], axis=0) # np.any fa l'or riga per riga tra gli array della lista
tr1_and = np.all([d[n].astype(bool) for n in hlt1], axis=0) # np.all fa l'and
tr2_or  = np.any([d[n].astype(bool) for n in hlt2], axis=0)
tr2_and = np.all([d[n].astype(bool) for n in hlt2], axis=0)

print("Hlt1 in OR: ", f"{tr1_or.mean():.1%}", "  Hlt1 in AND:", f"{tr1_and.mean():.1%}")
print("Hlt2 in OR: ", f"{tr2_or.mean():.1%}", "  Hlt2 in AND:", f"{tr2_and.mean():.1%}")
print("tutto (OR) + phi + PROBNN_K:", f"{(c1 & c4 & tr1_or & tr2_or).mean():.1%}")

tutti = c1 & c2 & c3 & c5 & c6 & pid_tutti & tr1_or & tr2_or # tutti i tagli messi insieme
k = tutti.sum()              # quanti passano
eff = k / n                  # efficienza
err = np.sqrt(eff * (1 - eff) / n)   # errore

print("passano:", k, "su", n)
print(f"efficienza = {eff:.4f} +- {err:.4f}")


# print([k for k in t.keys() if "DTF" in k and "MASS" in k])

# print(np.allclose(d["Jps_MASS"], d["B_DTF_PV_Jps_MASS"]))
# print(np.allclose(d["Jps_MASS"], d["B_DTF_HEAD_Jps_MASS"]))

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

"""fig, ax = plt.subplots(1, 2, figsize=(13, 5))

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
fig.savefig("confronto_ricalcolata.png")"""


fig, ax = plt.subplots(1, 2, figsize=(10, 4)) # fa due plot affiancati 
ax[0].hist(d["Jps_MASS"], bins=80, range=(2500, 3500), histtype="step", label="prima") # prima tutti i candidati
ax[0].hist(d["Jps_MASS"][c1], bins=80, range=(2500, 3500), histtype="step", label="dopo") # dopo solo quelli che passano c1
ax[0].set_xlabel("Jps_MASS [MeV]"); ax[0].legend()
ax[1].hist(d["B_MASS"], bins=80, range=(5100, 5700), histtype="step", label="prima")
ax[1].hist(d["B_MASS"][c1], bins=80, range=(5100, 5700), histtype="step", label="dopo")
ax[1].set_xlabel("B_MASS [MeV]"); ax[1].legend()
fig.savefig("mc_spruce.png")

