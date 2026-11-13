import ROOT
ROOT.gROOT.SetBatch(True)      

f = ROOT.TFile.Open("/eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/FTUPLE.ROOT/00412801/0000/00412801_00000001_1.ftuple.root")
tree = f.Get("SpruceRD_B0ToHpHmMuMu/DecayTree")
print(tree.GetEntries("abs(Phi_MASS - 1019.46) < 12") / tree.GetEntries())

h = ROOT.TH1D("h", ";Phi_MASS [MeV];candidati", 60, 990, 1050)
tree.Draw("Phi_MASS>>h")

c = ROOT.TCanvas("c", "c", 800, 600)
h.Draw()
c.SaveAs("phi_mass.png")