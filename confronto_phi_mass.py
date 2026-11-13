"""Confronto tra Phi_MASS dell'ntuple e la massa del phi ricostruita a mano.

calcolo la massa invariante dagli impulsi (px, py, pz) dei due kaoni H1 e H2 e la confronto con il branch Phi_MASS.
Se coincidono, Phi_MASS e' proprio la massa """

import numpy as np
import uproot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PATH = "/eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/FTUPLE.ROOT/00412801/0000/00412801_00000001_1.ftuple.root"
TREE = "SpruceRD_B0ToHpHmMuMu/DecayTree"
N_EVENTI = 200000          
OUTPUT_PNG = "phi_mass_confronto.png"

M_K = 493.677      # massa del k carico [MeV]
M_PI = 139.570     # massa del pi carico [MeV]


def energia(px, py, pz, m):

    return np.sqrt(px**2 + py**2 + pz**2 + m**2)


def massa_invariante(p1, m1, p2, m2):

    E = energia(*p1, m1) + energia(*p2, m2)
    px = p1[0] + p2[0]
    py = p1[1] + p2[1]
    pz = p1[2] + p2[2]
    m_quadro = E**2 - px**2 - py**2 - pz**2
    return np.sqrt(np.maximum(m_quadro, 0.0))      # evita radici di numeri negativi per errori numerici


def main():
    t = uproot.open(PATH)[TREE]

    # controllo che i branch esistano
    richiesti = ["Phi_MASS"] + [f"{h}_{c}" for h in ("H1", "H2") for c in ("PX", "PY", "PZ")]
    mancanti = [k for k in richiesti if k not in t.keys()]
    if mancanti:
        print("Branch non trovati:", mancanti)
        print("Branch che iniziano per H1_P / H2_P:",
              [k for k in t.keys() if k.startswith(("H1_P", "H2_P"))])
        raise SystemExit("Adatta i nomi in 'richiesti' e riprova.")

    # lettura
    d = t.arrays(richiesti, library="np", entry_stop=N_EVENTI)
    p1 = (d["H1_PX"], d["H1_PY"], d["H1_PZ"])
    p2 = (d["H2_PX"], d["H2_PY"], d["H2_PZ"])
    phi_ntuple = d["Phi_MASS"]
    print(f"Candidati letti: {len(phi_ntuple):}\n")

    # provo diverse ipotesi di massa per i due adroni
    ipotesi = {"K  K ": (M_K, M_K), "pi pi": (M_PI, M_PI), "K  pi": (M_K, M_PI), "pi K ": (M_PI, M_K)}
    risultati = {}
    print("ipotesi   |diff| max   differenza media   rms")
    for nome, (a, b) in ipotesi.items():
        m = massa_invariante(p1, a, p2, b)
        diff = m - phi_ntuple
        risultati[nome] = (m, diff)
        print(f"{nome}   {np.max(np.abs(diff)):10.4f}   {np.mean(diff):14.4f}   {np.std(diff):8.4f}")

    migliore = min(risultati, key=lambda k: np.mean(np.abs(risultati[k][1])))
    print(f"\nIpotesi più vicina a Phi_MASS: {migliore.strip()}")

    # grafici per l'ipotesi K K
    m_kk, diff_kk = risultati["K  K "]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].hist(phi_ntuple, bins=60, range=(990, 1050), histtype="step", label="Phi_MASS (ntuple)")
    ax[0].hist(m_kk, bins=60, range=(990, 1050), histtype="step", linestyle="--", label="ricostruita (K K)")
    ax[0].set_xlabel("massa [MeV]")
    ax[0].set_ylabel("candidati")
    ax[0].legend()

    lo, hi = np.percentile(diff_kk, [0.5, 99.5])
    if lo == hi:                      # differenza identica per tutti: allargo un po' l'intervallo
        lo, hi = lo - 1e-3, hi + 1e-3
    ax[1].hist(diff_kk, bins=80, range=(lo, hi))
    ax[1].set_xlabel("massa ricostruita - Phi_MASS [MeV]")
    ax[1].set_ylabel("candidati")
    ax[1].set_yscale("log")

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150)
    # print(f"Grafico salvato in {OUTPUT_PNG}")


if __name__ == "__main__":
    main()