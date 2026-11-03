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
] # B_MVA e TURBO_BLOCK non sono presenti nella tupla. B_QSQ neanche ma lo costruisco io 

d = t.arrays(
    branches + hlt1 + hlt2,
    library="np",
    entry_stop=200000
) 

print("candidati totali nel tree:", t.num_entries)

""" # controllo quali variabili esistono 
voluti = ["Phi_MASS", "B_MVA", "B_QSQ", "TURBO_BLOCK", "H1_PROBNN_K", "L1_PT", "B_PT"]
for nome in voluti:
    print(nome, "OK" if nome in t.keys() else "MANCA") """

# stampa tutti i branch con MASS nel nome
# print([k for k in t.keys() if "MASS" in k])

# aggiungo un cut alla volta
# ============================================================
# TAGLI SUI DUE KAONI: H1 e H2
# ============================================================

cut_k_pt = np.minimum(d["H1_PT"], d["H2_PT"]) > 250.0 # entrambi i kaoni devono avere PT > 250 MeV
cut_k_p = np.minimum(d["H1_P"], d["H2_P"]) > 0.0 # entrambi i kaoni devono avere impulso totale positivo
cut_k_eta_min = np.minimum( np.abs(d["H1_ETA"]), np.abs(d["H2_ETA"])) > 1.0 # entrambi i k devono avere |eta| > 1
cut_k_eta_max = np.maximum( np.abs(d["H1_ETA"]), np.abs(d["H2_ETA"])) < 6.0 # entrambi i k devono avere |eta| < 6
cut_k_ipchi2 = np.minimum( d["H1_BPVIPCHI2"], d["H2_BPVIPCHI2"]) > 6.0 # entrambi i kaoni devono essere incompatibili con il provenire dal PV
cut_k_chi2 = np.maximum( d["H1_CHI2NDOF"], d["H2_CHI2NDOF"]) < 4.0 # entrambe le tracce dei kaoni devono avere buona qualità
cut_k_ghost = np.maximum( d["H1_PROBNN_GHOST"], d["H2_PROBNN_GHOST"]) < 0.5 # entrambe le tracce devono avere probabilità ghost < 0.5
cut_k_probnn = np.minimum( d["H1_PROBNN_K"], d["H2_PROBNN_K"]) > 0.025 # entrambi i k devono essere compatibili con l'ipotesi di kaone
cut_k_pid = np.minimum( d["H1_PID_K"], d["H2_PID_K"] ) > -10.0 # entrambi i k devono superare il requisito PID del kaone

# ============================================================
# TAGLI SUI DUE MUONI: L1 e L2
# ============================================================

cut_mu_pt = np.minimum(d["L1_PT"], d["L2_PT"]) > 100.0 # entrambi i muoni devono avere PT > 100 MeV
cut_mu_p = np.minimum(d["L1_P"], d["L2_P"]) > 0.0 # entrambi i muoni devono avere impulso totale positivo
cut_mu_eta_min = np.minimum(np.abs(d["L1_ETA"]), np.abs(d["L2_ETA"])) > 1.0# entrambi i mu devono avere |eta| > 1
cut_mu_eta_max = np.maximum(np.abs(d["L1_ETA"]), np.abs(d["L2_ETA"])) < 6.0 # entrambi i muo devono avere |eta| < 6
cut_mu_ipchi2 = np.minimum(d["L1_BPVIPCHI2"], d["L2_BPVIPCHI2"]) > 2.0 # entrambi i muoni devono essere incompatibili con il provenire dal PV
cut_mu_chi2 = np.maximum(d["L1_CHI2NDOF"], d["L2_CHI2NDOF"]) < 4.0# entrambe le tracce dei muoni devono avere buona qualità
cut_mu_ghost = np.maximum(d["L1_PROBNN_GHOST"], d["L2_PROBNN_GHOST"]) < 0.5 # entrambe le tracce devono avere probabilità ghost < 0.5
cut_mu_probnn = np.minimum(d["L1_PROBNN_MU"], d["L2_PROBNN_MU"]) > 0.05 # entrambi i mu devono essere compatibili con l'ipotesi di muone
cut_mu_pid = np.minimum(d["L1_PID_MU"], d["L2_PID_MU"]) > -3.0 # entrambi i mu devono superare il requisito PID del muone

# ============================================================
# TAGLI SUL CANDIDATO B_s
# ============================================================

cut_b_pt_min = d["B_PT"] > 0.0 # impulso trasverso del B_s positivo
cut_b_p = d["B_P"] > 0.0 # impulso totale del B_s positivo
cut_b_pt_max = d["B_PT"] < 100000.0 # limite superiore molto largo sul PT del B_s
cut_b_fdchi2 = d["B_BPVFDCHI2"] > 10.0 # decadimento del B_s significativamente separato dal PV
cut_b_chi2 = d["B_CHI2NDOF"] < 20.0 # buona qualità del vertice di decadimento del B_s
cut_b_dira = d["B_BPVDIRA"] > 0.9995 # impulso del B_s allineato con la direzione PV -> vertice di decadimento
cut_b_ipchi2 = d["B_BPVIPCHI2"] < 100000.0 # limite superiore molto largo sull'IP chi2 del B_s
# B_MVA > 0.9355 non applicato: il branch B_MVA non è presente nella tupla

# ============================================================
# TAGLI SUL SISTEMA DIMUONICO: Jps -> mu+ mu-
# ============================================================

cut_jps_pt = d["Jps_PT"] > 0.0 # impulso trasverso del sistema dimuonico positivo
cut_jps_docachi2 = d["Jps_DOCACHI2"] < 36.0 # compatibilità geometrica tra le due tracce dei muoni
cut_jps_chi2 = d["Jps_CHI2NDOF"] < 30.0 # qualità del vertice formato dai due muoni
cut_jps_fdchi2 = d["Jps_BPVFDCHI2"] > 0.0 # significatività positiva della distanza di volo del dimuone

# ============================================================
# TAGLI SUL CANDIDATO PHI -> K+ K-
# ============================================================

cut_phi_mass = np.abs(d["Phi_MASS"] - 1019.46) < 12.0 # massa dei due kaoni entro 12 MeV dalla massa nominale della phi
cut_phi_pt = d["Phi_PT"] > 600.0 # impulso trasverso della phi maggiore di 600 MeV
cut_phi_chi2 = d["Phi_CHI2NDOF"] < 16.0 # qualità del vertice formato dai due kaoni
cut_phi_docachi2 = d["Phi_DOCACHI2"] < 10.0 # compatibilità geometrica tra le due tracce dei kaoni
cut_phi_doca = d["Phi_DOCA"] < 0.5 # distanza minima tra le due tracce dei kaoni
cut_phi_ipchi2 = d["Phi_BPVIPCHI2"] > 0.0 # IP chi2 della phi rispetto al PV
# TURBO_BLOCK == 2 non applicato: il branch TURBO_BLOCK non è presente nella tupla
            
                                    
q2 = d["Jps_MASS"]**2 / 1e6 # q^2 senza DTF. 1e6 serve per trasformare da mev2 a gev2
q2_dtf = d["B_DTF_HEAD_Jps_MASS"]**2 / 1e6 # q^2 con DTF HEAD
cq2 = (q2 > 0) & (q2 < 19) # taglio 0 < q^2 < 19 GeV^2 (per entrambi)
cq2_dtf = (q2_dtf > 0) & (q2_dtf < 19)

# trigger: ogni decisione da sola, poi OR e AND dentro ciascun gruppo
#for nome in hlt1 + hlt2:
#    print(f"{nome:42s}", f"{d[nome].astype(bool).mean():.1%}")

tr1_or  = np.any([d[n].astype(bool) for n in hlt1], axis=0) # np.any fa l'or riga per riga tra gli array della lista
# tr1_and = np.all([d[n].astype(bool) for n in hlt1], axis=0) # np.all fa l'and
tr2_or  = np.any([d[n].astype(bool) for n in hlt2], axis=0)
# tr2_and = np.all([d[n].astype(bool) for n in hlt2], axis=0)

# print("Hlt1 in OR: ", f"{tr1_or.mean():.1%}", "  Hlt1 in AND:", f"{tr1_and.mean():.1%}")
# print("Hlt2 in OR: ", f"{tr2_or.mean():.1%}", "  Hlt2 in AND:", f"{tr2_and.mean():.1%}")
# print("tutto (OR) + phi + PROBNN_K:", f"{(c1 & c4 & tr1_or & tr2_or).mean():.1%}")

tagli = [
    # kaoni
    ("K PT", cut_k_pt),
    ("K P", cut_k_p),
    ("K eta min", cut_k_eta_min),
    ("K eta max", cut_k_eta_max),
    ("K IPCHI2", cut_k_ipchi2),
    ("K CHI2NDOF", cut_k_chi2),
    ("K ghost", cut_k_ghost),
    ("K ProbNN", cut_k_probnn),
    ("K PID", cut_k_pid),

    # muoni
    ("mu PT", cut_mu_pt),
    ("mu P", cut_mu_p),
    ("mu eta min", cut_mu_eta_min),
    ("mu eta max", cut_mu_eta_max),
    ("mu IPCHI2", cut_mu_ipchi2),
    ("mu CHI2NDOF", cut_mu_chi2),
    ("mu ghost", cut_mu_ghost),
    ("mu ProbNN", cut_mu_probnn),
    ("mu PID", cut_mu_pid),

    # candidato B_s
    ("B PT min", cut_b_pt_min),
    ("B P", cut_b_p),
    ("B PT max", cut_b_pt_max),
    ("B FDCHI2", cut_b_fdchi2),
    ("B CHI2NDOF", cut_b_chi2),
    ("B DIRA", cut_b_dira),
    ("B IPCHI2", cut_b_ipchi2),

    # sistema dimuonico
    ("Jps PT", cut_jps_pt),
    ("Jps DOCACHI2", cut_jps_docachi2),
    ("Jps CHI2NDOF", cut_jps_chi2),
    ("Jps FDCHI2", cut_jps_fdchi2),

    # candidato phi
    ("Phi mass", cut_phi_mass),
    ("Phi PT", cut_phi_pt),
    ("Phi CHI2NDOF", cut_phi_chi2),
    ("Phi DOCACHI2", cut_phi_docachi2),
    ("Phi DOCA", cut_phi_doca),
    ("Phi IPCHI2", cut_phi_ipchi2),
]

n = len(d["B_MASS"])                                   # candidati di partenza
print("candidati analizzati:             ", n)
"""print("solo finestra phi:    ", c1.sum(), f"({c1.mean():.1%})")
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
print("pid + phi:           ", (pid_tutti & c1).sum(), f"({(pid_tutti & c1).mean():.1%})") """

mask = np.ones(n, dtype=bool)

print("\nCUT CUMULATIVI")
print(f"{'partenza':10s}: {n:8d}   {100.0:6.2f}%") #allinea le colonne

for nome, cut in tagli:
    mask &= cut # fa un and tra chi è sopravvissuto fino a quel momento e chi passa il nuovo taglio 

    k = mask.sum() # conta i candidati rimasti 
    eff = k / n # efficienza cumulativa 

    print(f"{nome:10s}: {k:8d}   {100*eff:6.2f}%")

# aggiungo HLT1 ai cut cumulativi
mask &= tr1_or
k = mask.sum()
eff = k / n
print(f"{'HLT1':10s}: {k:8d}   {100*eff:6.2f}%")

# aggiungo HLT2 ai cut cumulativi
mask &= tr2_or
k = mask.sum()
eff = k / n
print(f"{'HLT2':10s}: {k:8d}   {100*eff:6.2f}%")

# cut cumulativo diviso in gruppi di particelle
mask = np.ones(n, dtype=bool)

nomi_gruppi = ["Partenza"]
n_gruppi = [n]

# tutti i tagli sui kaoni
for cut in [
    cut_k_pt, cut_k_p, cut_k_eta_min, cut_k_eta_max,
    cut_k_ipchi2, cut_k_chi2, cut_k_ghost,
    cut_k_probnn, cut_k_pid
]:
    mask &= cut

nomi_gruppi.append("Kaoni")
n_gruppi.append(mask.sum())

# tutti i tagli sui muoni
for cut in [
    cut_mu_pt, cut_mu_p, cut_mu_eta_min, cut_mu_eta_max,
    cut_mu_ipchi2, cut_mu_chi2, cut_mu_ghost,
    cut_mu_probnn, cut_mu_pid
]:
    mask &= cut

nomi_gruppi.append("Muoni")
n_gruppi.append(mask.sum())

# tutti i tagli sul Bs
for cut in [
    cut_b_pt_min, cut_b_p, cut_b_pt_max,
    cut_b_fdchi2, cut_b_chi2, cut_b_dira, cut_b_ipchi2
]:
    mask &= cut

nomi_gruppi.append("Bs")
n_gruppi.append(mask.sum())

# tutti i tagli sul sistema dimuonico
for cut in [
    cut_jps_pt, cut_jps_docachi2,
    cut_jps_chi2, cut_jps_fdchi2
]:
    mask &= cut

nomi_gruppi.append("Dimuone")
n_gruppi.append(mask.sum())

# tutti i tagli sulla phi
for cut in [
    cut_phi_mass, cut_phi_pt, cut_phi_chi2,
    cut_phi_docachi2, cut_phi_doca, cut_phi_ipchi2
]:
    mask &= cut

nomi_gruppi.append("Phi")
n_gruppi.append(mask.sum())

# HLT1
mask &= tr1_or
nomi_gruppi.append("HLT1")
n_gruppi.append(mask.sum())

# HLT2
mask &= tr2_or
nomi_gruppi.append("HLT2")
n_gruppi.append(mask.sum())

# tutti = c1 & c2 & c3 & c5 & c6 & pid_tutti & tr1_or & tr2_or # tutti i tagli messi insieme
tutti_comuni = (
    cut_k_pt & cut_k_p & cut_k_eta_min & cut_k_eta_max & cut_k_ipchi2 & cut_k_chi2 & cut_k_ghost & cut_k_probnn & cut_k_pid &
    cut_mu_pt & cut_mu_p & cut_mu_eta_min & cut_mu_eta_max & cut_mu_ipchi2 & cut_mu_chi2 & cut_mu_ghost & cut_mu_probnn & cut_mu_pid &
    cut_b_pt_min & cut_b_p & cut_b_pt_max & cut_b_fdchi2 & cut_b_chi2 & cut_b_dira & cut_b_ipchi2 &
    cut_jps_pt & cut_jps_docachi2 & cut_jps_chi2 & cut_jps_fdchi2 &
    cut_phi_mass & cut_phi_pt & cut_phi_chi2 & cut_phi_docachi2 & cut_phi_doca & cut_phi_ipchi2 &
    tr1_or & tr2_or )

k = tutti_comuni.sum()              # quanti passano
eff = k / n                  # efficienza
err = np.sqrt(eff * (1 - eff) / n)   # errore
eff_gruppi = np.array(n_gruppi) / n # eff di ogni gruppo 

print("passano con tutti i cut comuni:", k, "su", n)
print(f"efficienza = {eff:.4f} +- {err:.4f}")

# efficienza relativa dei singoli tagli 
mask = np.ones(n, dtype=bool)

print("\nEFFICIENZE RELATIVE")
print(f"{'taglio':16s} {'prima':>10s} {'dopo':>10s} {'eff.rel.':>10s}")

for nome, cut in tagli:

    n_prima = mask.sum()      # candidati prima di applicare questo taglio
    mask &= cut               # applico il taglio
    n_dopo = mask.sum()       # candidati rimasti

    eff_rel = n_dopo / n_prima
    print(
        f"{nome:16s} "
        f"{n_prima:10d} "
        f"{n_dopo:10d} "
        f"{100*eff_rel:9.2f}%"
    )

# aggiungo HLT1
n_prima = mask.sum()
mask &= tr1_or
n_dopo = mask.sum()
eff_rel = n_dopo / n_prima

print(
    f"{'HLT1':16s} "
    f"{n_prima:10d} "
    f"{n_dopo:10d} "
    f"{100*eff_rel:9.2f}%"
)

# aggiuno HLT2
n_prima = mask.sum()
mask &= tr2_or
n_dopo = mask.sum()
eff_rel = n_dopo / n_prima

print(
    f"{'HLT2':16s} "
    f"{n_prima:10d} "
    f"{n_dopo:10d} "
    f"{100*eff_rel:9.2f}%"
)

print("\nTAGLIO q2") # taglio q2 preso da solo sui 200000 candidati 

print(
    "senza DTF:",
    cq2.sum(),
    "su", n,
    f"({cq2.mean():.2%})"
)

print(
    "DTF HEAD:",
    cq2_dtf.sum(),
    "su", n,
    f"({cq2_dtf.mean():.2%})"
)


fig, ax = plt.subplots(1, 2, figsize=(10, 4)) # fa due plot affiancati 
ax[0].hist(d["Jps_MASS"], bins=80, range=(2500, 3500), histtype="step", label="prima") # prima tutti i candidati
ax[0].hist(d["Jps_MASS"][tutti_comuni], bins=80, range=(2500, 3500), histtype="step", label="dopo") # dopo solo quelli che passano tutti_comuni
ax[0].set_xlabel("Jps_MASS [MeV]"); ax[0].legend()
ax[0].set_ylabel("Candidati")
ax[1].hist(d["B_MASS"], bins=80, range=(5100, 5700), histtype="step", label="prima")
ax[1].hist(d["B_MASS"][tutti_comuni], bins=80, range=(5100, 5700), histtype="step", label="dopo")
ax[1].set_xlabel("B_MASS [MeV]"); ax[1].legend()
ax[1].set_ylabel("Candidati")
fig.savefig("mc_spruce_tutti_cut.png")


plt.figure(figsize=(9, 5))

plt.plot(
    nomi_gruppi,
    100 * eff_gruppi,
    marker="o"
)

plt.ylabel("Efficienza cumulativa [%]")
plt.xlabel("Selezione")
plt.ylim(0, 105)

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("cut_cumulativo_mc_spruce.png", dpi=150)

plt.figure(figsize=(8, 5))

plt.hist(
    q2,
    bins=100,
    range=(8, 11),
    histtype="step",
    label="no DTF"
)

plt.hist(
    q2_dtf,
    bins=100,
    range=(8, 11),
    histtype="step",
    label="DTF HEAD"
)

plt.xlabel(r"$q^2$ [GeV$^2$]")
plt.ylabel("Candidati")
plt.legend()

plt.tight_layout()
plt.savefig("q2_dtf_vs_nodtf.png", dpi=150)