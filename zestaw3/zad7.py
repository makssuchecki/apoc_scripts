import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

G = 256
ALFA = -1 / 3
SPRAWDZ = (30, 35, 40)


def wczytaj_8bit(sciezka):
    """Obraz w skali szarości 0..255. RGB -> (R+G+B)/3 jak w ImageJ (Image > Type > 8-bit)."""
    a = np.array(Image.open(sciezka))
    if a.ndim == 3:
        a = np.floor(a[..., :3].astype(float).mean(axis=2) + 0.5)
    return a.astype(np.uint8)


def zapisz_z_histogramem(img, nazwa, tytul, krzywa=None):
    Image.fromarray(img).save(nazwa + ".png")
    h = np.bincount(img.ravel(), minlength=G)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.2, 1]})
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=255); ax[0].set_title(tytul); ax[0].axis("off")
    ax[1].bar(range(G), h, width=1, color="black"); ax[1].set_xlim(0, 255)
    ax[1].set_title(f"Mean={img.mean():.2f}  StdDev={img.std():.2f}  Min={img.min()}  Max={img.max()}", fontsize=9)
    if krzywa is not None:
        ax2 = ax[1].twinx(); ax2.plot(range(G), krzywa, "r"); ax2.set_ylim(0, 256)
    fig.tight_layout(); fig.savefig(nazwa + "_hist.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    img = wczytaj_8bit(sys.argv[1] if len(sys.argv) > 1 else "Metropolis.png")
    MN = img.size
    print(f"Metropolis 8-bit: {img.shape[1]}x{img.shape[0]} = {MN} pikseli, "
          f"min={img.min()}, max={img.max()}, mean={img.mean():.2f}")
    zapisz_z_histogramem(img, "z7_wejscie", "obraz wejściowy (8-bit)")

    # ---------------- (a) histogram i histogram skumulowany ----------------
    h = np.bincount(img.ravel(), minlength=G)    # h(g)  = liczba pikseli o wartości g
    hs = np.cumsum(h)                            # hs(g) = h(0) + h(1) + ... + h(g)
    Hs = hs / MN                                 # Hs(g) w [0, 1]

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].bar(range(G), h, width=1, color="black"); ax[0].set_title("histogram h(g)"); ax[0].set_xlim(0, 255)
    ax[1].plot(range(G), hs, "k"); ax[1].set_title("histogram skumulowany hs(g)"); ax[1].set_xlim(0, 255)
    ax[1].set_xlabel("g")
    ax2 = ax[1].twinx(); ax2.plot(range(G), Hs, "r--", lw=0.8); ax2.set_ylabel("Hs(g)", color="r"); ax2.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig("z7a_histogram_skumulowany.png", dpi=110); plt.close(fig)

    # ---------------- (b) wyrównanie ----------------
    H_eq = np.round((G - 1) * Hs).astype(np.uint8)          # tablica LUT: g -> H_eq(g)
    wynik_b = H_eq[img]                                     # każdy piksel przez LUT
    zapisz_z_histogramem(wynik_b, "z7b_wyrownanie", "(b) wyrównanie histogramu", krzywa=H_eq)

    # ---------------- (c) hiperbolizacja ----------------
    wykladnik = 1 / (ALFA + 1)                              # alfa = -1/3 -> 3/2
    H_hyper = np.round((G - 1) * Hs ** wykladnik).astype(np.uint8)
    wynik_c = H_hyper[img]
    zapisz_z_histogramem(wynik_c, "z7c_hiperbolizacja", "(c) hiperbolizacja α = -1/3", krzywa=H_hyper)

    # ---------------- tabela i odpowiedzi ----------------
    np.savetxt("z7_tabela.csv", np.c_[np.arange(G), h, hs, Hs, H_eq, H_hyper],
               fmt=["%d", "%d", "%d", "%.6f", "%d", "%d"], delimiter=";",
               header="g;h(g);hs(g);Hs(g);H_eq(g);H_hyper(g)", comments="")

    print(f"\nwykladnik hiperbolizacji 1/(alfa+1) = {wykladnik:.4f}")
    print("\n  g |  h(g) |  hs(g) |  Hs(g)  | 255*Hs  | H_eq | 255*Hs^1.5 | H_hyper")
    for g in SPRAWDZ:
        print(f" {g:2d} | {h[g]:5d} | {hs[g]:6d} | {Hs[g]:.5f} | {255 * Hs[g]:7.3f} | {H_eq[g]:4d} |"
              f" {255 * Hs[g] ** wykladnik:10.3f} | {H_hyper[g]:4d}")
    print(f"\nwyrownanie:     mean {img.mean():.2f} -> {wynik_b.mean():.2f}")
    print(f"hiperbolizacja: mean {img.mean():.2f} -> {wynik_c.mean():.2f}")
    print("\nZapisano: z7_wejscie*, z7a_histogram_skumulowany.png, z7b_*, z7c_*, z7_tabela.csv")