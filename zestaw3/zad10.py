import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

G = 256
g = np.arange(G, dtype=float)
PROCENT_ODCIECIA = 0.5      # ile % pikseli odcinamy z każdej strony w kroku 1
ALFA = -1 / 3


def statystyki(img):
    h = np.bincount(img.ravel(), minlength=G)
    return (f"Mean={img.mean():.2f}  StdDev={img.std():.2f}  Min={img.min()}  "
            f"Max={img.max()}  Mode={h.argmax()}")


def zapisz_krok(img, nazwa, tytul, T=None, wejscie=None):
    """Zapisuje obraz + rysunek: obraz | histogram wyniku | (opcjonalnie) hist. wejścia z krzywą T."""
    Image.fromarray(img).save(nazwa + ".png")
    n = 3 if T is not None else 2
    fig, ax = plt.subplots(1, n, figsize=(4 * n, 3.6))
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=255); ax[0].set_title(tytul, fontsize=9); ax[0].axis("off")
    ax[1].bar(range(G), np.bincount(img.ravel(), minlength=G), width=1, color="black")
    ax[1].set_xlim(0, 255); ax[1].set_yticks([]); ax[1].set_title("histogram wyniku", fontsize=9)
    ax[1].set_xlabel(statystyki(img), fontsize=7)
    if T is not None:
        ax[2].bar(range(G), np.bincount(wejscie.ravel(), minlength=G), width=1, color="0.7")
        ax[2].set_yticks([]); ax[2].set_xlim(0, 255)
        a2 = ax[2].twinx(); a2.plot(range(G), T, "r"); a2.set_ylim(0, 256)
        ax[2].set_title("histogram wejścia + krzywa T(g)", fontsize=9)
    fig.tight_layout(); fig.savefig(nazwa + "_hist.png", dpi=110); plt.close(fig)
    print(f"{nazwa}: {statystyki(img)}")


if __name__ == "__main__":
    img0 = np.array(Image.open(sys.argv[1] if len(sys.argv) > 1 else "CalunTurynski.png").convert("L"))

    # ---- krok 0: analiza oryginału ----
    zapisz_krok(img0, "z10_1_oryginal", "1. oryginał")

    # ---- krok 1: clipping + rozproszenie ----
    gmin, gmax = np.percentile(img0, [PROCENT_ODCIECIA, 100 - PROCENT_ODCIECIA]).round().astype(int)
    print(f"\nkrok 1: gmin={gmin}, gmax={gmax}  "
          f"(odcięte: {np.mean(img0 < gmin) * 100:.2f}% ciemnych, {np.mean(img0 > gmax) * 100:.2f}% jasnych)")
    T1 = np.clip(np.round(255 * (g - gmin) / (gmax - gmin)), 0, 255).astype(np.uint8)
    img1 = T1[img0]
    zapisz_krok(img1, "z10_2_clipping", f"2. clipping {gmin}..{gmax} + rozproszenie", T1, img0)

    # ---- krok 2: inwersja ----
    T2 = (255 - g).astype(np.uint8)
    img2 = T2[img1]
    zapisz_krok(img2, "z10_3_inwersja", "3. inwersja histogramu", T2, img1)

    # ---- krok 3: hiperbolizacja ----
    Hs = np.cumsum(np.bincount(img2.ravel(), minlength=G)) / img2.size
    T3 = np.round(255 * Hs ** (1 / (ALFA + 1))).astype(np.uint8)
    img3 = T3[img2]
    zapisz_krok(img3, "z10_4_hiperbolizacja", "4. hiperbolizacja α = -1/3", T3, img2)

    # (porównanie) zwykłe wyrównanie zamiast hiperbolizacji
    Teq = np.round(255 * Hs).astype(np.uint8)
    zapisz_krok(Teq[img2], "z10_4b_wyrownanie_porownanie", "4'. wyrównanie (porównanie)", Teq, img2)

    # ---- zestawienie wszystkich kroków ----
    fig, ax = plt.subplots(1, 4, figsize=(14, 4.6))
    for a, im, t in zip(ax, [img0, img1, img2, img3],
                        ["1. oryginał", "2. clipping", "3. inwersja", "4. hiperbolizacja"]):
        a.imshow(im, cmap="gray", vmin=0, vmax=255); a.set_title(t); a.axis("off")
    fig.tight_layout(); fig.savefig("z10_zestawienie.png", dpi=110); plt.close(fig)
    print("\nZapisano: z10_0..3*.png, *_hist.png, z10_zestawienie.png")