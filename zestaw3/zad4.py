"""Zadanie 4 – binaryzacja obrazu roze.png metodą Otsu.

(a) próg globalny Otsu
(b) iteracyjne progowanie trójklasowe (warunek |Delta| < 2)
(c) lokalny próg Otsu w oknie 11x11, symetryczne odbicie na brzegach

Uruchomienie:  python zad4.py roze.png
Wymaga: numpy, pillow, matplotlib  (pip install numpy pillow matplotlib)
"""
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Otsu wg skryptu: dla każdego progu l liczymy wariancję wewnątrzklasową
#   sigma_w^2(l) = w0*sigma0^2 + w1*sigma1^2
# klasa 0: g <= l, klasa 1: g > l. Wybieramy l z NAJMNIEJSZĄ sigma_w^2.
# Liczymy tylko na przedziale wartości [lo, hi] (potrzebne w części (b)).
# ---------------------------------------------------------------------------
def otsu(h, lo=0, hi=255):
    g = np.arange(lo, hi + 1, dtype=float)
    hh = h[lo:hi + 1].astype(float)

    # sumy skumulowane dla klasy 0 (wszystko <= l)
    w0 = np.cumsum(hh)              # liczba pikseli
    s0 = np.cumsum(hh * g)          # suma wartości
    q0 = np.cumsum(hh * g * g)      # suma kwadratów
    # klasa 1 = reszta przedziału
    w1, s1, q1 = w0[-1] - w0, s0[-1] - s0, q0[-1] - q0

    # w_i * sigma_i^2 = sum(g^2 h) - (sum(g h))^2 / sum(h)
    with np.errstate(divide="ignore", invalid="ignore"):
        var0 = np.where(w0 > 0, q0 - s0 ** 2 / w0, 0.0)
        var1 = np.where(w1 > 0, q1 - s1 ** 2 / w1, 0.0)
    sigma_w = var0 + var1

    k = int(np.argmin(sigma_w))     # pierwszy próg z minimum
    T = lo + k
    mu0 = s0[k] / w0[k] if w0[k] > 0 else float("nan")
    mu1 = s1[k] / w1[k] if w1[k] > 0 else float("nan")
    return T, mu0, mu1


def binaryzuj(img, T):
    """g > T -> 255 (biały), g <= T -> 0 (czarny)."""
    return np.where(img > T, 255, 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# (b) Iteracyjne progowanie trójklasowe
#   krok 0: Otsu na całym histogramie -> T, mu0, mu1
#   krok k: Otsu tylko na regionie TBD [mu0, mu1] z poprzedniego kroku
#   stop, gdy |T_nowe - T_stare| < eps
# ---------------------------------------------------------------------------
def otsu_iteracyjny(h, eps=2):
    T, mu0, mu1 = otsu(h)
    kroki = [(0, 0, 255, T, mu0, mu1, None)]
    k = 0
    while True:
        lo, hi = math.ceil(mu0), math.floor(mu1)    # region TBD
        T_nowe, mu0, mu1 = otsu(h, lo, hi)
        k += 1
        delta = abs(T_nowe - T)
        kroki.append((k, lo, hi, T_nowe, mu0, mu1, delta))
        T = T_nowe
        if delta < eps:
            return T, kroki


# ---------------------------------------------------------------------------
# (c) Lokalny Otsu: dla każdego piksela histogram okna 11x11 -> własny próg
# ---------------------------------------------------------------------------
def otsu_lokalny(img, rozmiar=11):
    r = rozmiar // 2
    # symetryczne odbicie: indeks -1 -> 0, -2 -> 1, ..., W -> W-1, ...
    pad = np.pad(img, r, mode="symmetric")
    H, W = img.shape
    progi = np.zeros((H, W), dtype=np.int32)
    for y in range(H):
        for x in range(W):
            okno = pad[y:y + rozmiar, x:x + rozmiar]      # 11x11 wokół (x, y)
            h = np.bincount(okno.ravel(), minlength=256)
            progi[y, x] = otsu(h)[0]
        if y % 50 == 0:
            print(f"   wiersz {y}/{H}")
    return progi
    # Uwaga: w jednolitym oknie (np. białe tło) każdy próg daje sigma_w = 0,
    # więc wybierany jest pierwszy, T = 0 -> piksel zostaje biały.


# ---------------------------------------------------------------------------
def zapisz_z_histogramem(img, nazwa, tytul):
    Image.fromarray(img).save(nazwa + ".png")
    h = np.bincount(img.ravel(), minlength=256)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5), gridspec_kw={"width_ratios": [1, 1.3]})
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    ax[0].set_title(tytul); ax[0].axis("off")
    ax[1].bar(range(256), h, width=1, color="black"); ax[1].set_xlim(0, 255)
    ax[1].set_title(f"histogram: białe = {np.mean(img == 255) * 100:.1f}%")
    fig.tight_layout(); fig.savefig(nazwa + "_hist.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    sciezka = sys.argv[1] if len(sys.argv) > 1 else "roze.png"
    img = np.array(Image.open(sciezka).convert("L"))
    h = np.bincount(img.ravel(), minlength=256)
    print(f"roze: {img.shape[1]}x{img.shape[0]}, min={img.min()}, max={img.max()}, mean={img.mean():.2f}")

    # (a)
    T, mu0, mu1 = otsu(h)
    print(f"\n(a) Otsu globalny: T = {T}, mu0 = {mu0:.2f}, mu1 = {mu1:.2f}")
    zapisz_z_histogramem(binaryzuj(img, T), "z4a_otsu_globalny", f"(a) Otsu globalny, T = {T}")

    # (b)
    T3, kroki = otsu_iteracyjny(h, eps=2)
    print("\n(b) Iteracyjny trójklasowy Otsu:")
    print(" k | region TBD  |  T  |   mu0  |   mu1  | Delta")
    for k, lo, hi, t, m0, m1, d in kroki:
        print(f" {k} | [{lo:3d}, {hi:3d}] | {t:3d} | {m0:6.2f} | {m1:6.2f} | {'-' if d is None else d}")
    print(f"    koncowy prog T = {T3}")
    zapisz_z_histogramem(binaryzuj(img, T3), "z4b_otsu_iteracyjny", f"(b) iteracyjny 3-klasowy, T = {T3}")

    # (c)
    print("\n(c) Lokalny Otsu 11x11 (chwilę trwa)...")
    progi = otsu_lokalny(img, 11)
    wynik_c = np.where(img > progi, 255, 0).astype(np.uint8)
    zapisz_z_histogramem(wynik_c, "z4c_otsu_lokalny", "(c) lokalny Otsu 11x11")
    Image.fromarray(progi.astype(np.uint8)).save("z4c_mapa_progow.png")
    print(f"    progi lokalne od {progi.min()} do {progi.max()}")
    print("\nZapisano: z4a_*, z4b_*, z4c_* (.png + _hist.png)")