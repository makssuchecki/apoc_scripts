"""Zadanie 17 – okienkowanie obrazu ptaki.png.

(a) okno sinusoidalne:  w(m,n) = sin(pi*m/M) * sin(pi*n/N),  g* = w * g
(b) filtr uśredniający 3x3 na wyniku (a)
(c) korekta gamma wyniku (b), gamma dobrana tak, by średnia ≈ średnia ptaki.png
(d) filtr uśredniający 3x3 bezpośrednio na ptaki.png + porównanie z (c)

Uruchomienie:  python zad17.py ptaki.png
Wymaga: numpy, pillow, matplotlib, scipy
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

G = 256


def do_8bit(a):
    return np.clip(np.round(a), 0, 255).astype(np.uint8)


def srednia_3x3(img):
    # 1/9 * suma 3x3; na brzegach powielany skrajny piksel (jak Process > Smooth w ImageJ)
    return do_8bit(uniform_filter(img.astype(float), size=3, mode="nearest"))


def gamma(img, gm):
    return do_8bit(255 * (img / 255.0) ** gm)


def zapisz(img, nazwa, tytul):
    Image.fromarray(img).save(nazwa + ".png")
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.6))
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=255); ax[0].set_title(tytul, fontsize=9); ax[0].axis("off")
    ax[1].bar(range(G), np.bincount(img.ravel(), minlength=G), width=1, color="black")
    ax[1].set_xlim(0, 255); ax[1].set_yticks([])
    ax[1].set_title(f"Mean={img.mean():.2f}  StdDev={img.std():.2f}  Min={img.min()}  Max={img.max()}", fontsize=8)
    fig.tight_layout(); fig.savefig(nazwa + "_hist.png", dpi=110); plt.close(fig)
    print(f"{nazwa:26s} mean={img.mean():7.2f}  std={img.std():6.2f}")


if __name__ == "__main__":
    g = np.array(Image.open(sys.argv[1] if len(sys.argv) > 1 else "ptaki.png").convert("L"))
    M, N = g.shape                       # M = liczba wierszy, N = liczba kolumn
    zapisz(g, "z17_0_ptaki", "ptaki.png (oryginał)")

    # ---- (a) okno sinusoidalne ----
    m = np.arange(M)[:, None]            # indeks wiersza 0..M-1
    n = np.arange(N)[None, :]            # indeks kolumny 0..N-1
    w = np.sin(np.pi * m / M) * np.sin(np.pi * n / N)     # 0 na brzegach, 1 w środku
    Image.fromarray(do_8bit(255 * w)).save("z17_okno_w.png")
    a = do_8bit(w * g)
    zapisz(a, "z17a_okno", "(a) okno sinusoidalne")

    # ---- (b) uśrednianie 3x3 ----
    b = srednia_3x3(a)
    zapisz(b, "z17b_srednia", "(b) (a) + filtr uśredniający 3x3")

    # ---- (c) gamma: szukamy gamma, przy której średnia(c) ≈ średnia(ptaki) ----
    cel = g.mean()
    kandydaci = np.arange(0.05, 1.5, 0.001)
    srednie = np.array([gamma(b, x).mean() for x in kandydaci])
    gm = kandydaci[np.argmin(np.abs(srednie - cel))]
    c = gamma(b, gm)
    print(f"\n(c) cel: mean = {cel:.2f}  ->  gamma = {gm:.3f}, mean(c) = {c.mean():.2f}\n")
    zapisz(c, "z17c_gamma", f"(c) korekta gamma, γ = {gm:.3f}")

    # ---- (d) uśrednianie bezpośrednio na oryginale ----
    d = srednia_3x3(g)
    zapisz(d, "z17d_srednia_oryginal", "(d) filtr 3x3 na oryginale")

    # ---- porównanie (c) vs (d): jasność i szum w fragmentach 40x40 ----
    fragmenty = {"róg lewy-górny": (slice(5, 45), slice(5, 45)),
                 "środek, jasna ściana": (slice(230, 270), slice(60, 100)),
                 "środek, ciemna ściana": (slice(230, 270), slice(400, 440))}
    print("\nfragment 40x40          |   oryginał    |     (c)       |     (d)")
    print("                        |  mean   std   |  mean   std   |  mean   std")
    for nazwa, (ys, xs) in fragmenty.items():
        wiersz = f"{nazwa:24s}|"
        for im in (g, c, d):
            f = im[ys, xs].astype(float)
            wiersz += f" {f.mean():6.1f} {f.std():5.1f}  |"
        print(wiersz)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4.2))
    for x, im, t in zip(ax, [g, c, d], ["oryginał", f"(c) okno + średnia + γ={gm:.3f}", "(d) średnia na oryginale"]):
        x.imshow(im, cmap="gray", vmin=0, vmax=255); x.set_title(t); x.axis("off")
    fig.tight_layout(); fig.savefig("z17_porownanie_c_d.png", dpi=110); plt.close(fig)
    print("\nZapisano: z17_*.png, *_hist.png, z17_porownanie_c_d.png")