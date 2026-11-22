"""Zadanie 18 – flat-field correction obrazu torus.png.

(a) g_C = (g - g_D) * g_Mean / (g_F - g_D),   g_Mean = średnia(g_F - g_D)
(b) rozproszenie histogramu wyniku do pełnego zakresu 0..255 -> kontrast globalny = 1

Uruchomienie:  python zad18.py torus.png Dark_Frame.png Flat_Frame.png
Wymaga: numpy, pillow, matplotlib
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

G = 256
pliki = sys.argv[1:4] if len(sys.argv) >= 4 else ["torus.png", "Dark_Frame.png", "Flat_Frame.png"]
g, gD, gF = (np.array(Image.open(p).convert("L")).astype(float) for p in pliki)   # liczymy na float (32-bit)


def zapisz(img, nazwa, tytul):
    Image.fromarray(img).save(nazwa + ".png")
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.6))
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=255); ax[0].set_title(tytul, fontsize=9); ax[0].axis("off")
    ax[1].bar(range(G), np.bincount(img.ravel(), minlength=G), width=1, color="black")
    ax[1].set_xlim(0, 255); ax[1].set_yticks([])
    ax[1].set_title(f"Mean={img.mean():.2f} StdDev={img.std():.2f} Min={img.min()} Max={img.max()}", fontsize=8)
    fig.tight_layout(); fig.savefig(nazwa + "_hist.png", dpi=110); plt.close(fig)


for im, n in [(g, "torus"), (gD, "dark"), (gF, "flat")]:
    zapisz(im.astype(np.uint8), f"z18_0_{n}", n)
    print(f"{n:6s}: min={im.min():.0f} max={im.max():.0f} mean={im.mean():.2f}")

# ---------------- (a) flat-field correction ----------------
mianownik = gF - gD                         # samo nierównomierne oświetlenie (bez szumu sensora)
gMean = mianownik.mean()
print(f"\ng_Mean = średnia(g_F - g_D) = {gMean:.3f}")

zle = mianownik <= 0                        # dzielenie przez 0 (lub ujemne)
print(f"pikseli z g_F - g_D <= 0: {zle.sum()} (kolumny {sorted(set(np.nonzero(zle)[1].tolist()))})")
gC = np.where(zle, np.nan, (g - gD) * gMean / np.where(zle, 1, mianownik))
# brakujące piksele uzupełniamy wartością z najbliższej poprawnej kolumny (tu: kolumna 2, to tło)
pierwsza_ok = int(np.where(~zle.any(axis=0))[0][0])
gC[zle] = np.broadcast_to(gC[:, pierwsza_ok:pierwsza_ok + 1], gC.shape)[zle]

print(f"g_C (float): min={gC.min():.2f}, max={gC.max():.2f}, mean={gC.mean():.2f}")
print(f"  pikseli > 255: {np.mean(gC > 255) * 100:.2f}%  (jasne miejsca torusa, w oryginale nasycone na 255)")
a = np.clip(np.round(gC), 0, 255).astype(np.uint8)        # do 8-bit, wartości > 255 obcięte
zapisz(a, "z18a_flatfield", "(a) flat-field correction")

# ---------------- (b) kontrast globalny = 1 ----------------
gmin, gmax = int(a.min()), int(a.max())
b = np.round(255 * (a.astype(float) - gmin) / (gmax - gmin)).astype(np.uint8)
zapisz(b, "z18b_kontrast1", f"(b) rozproszenie {gmin}..{gmax} -> 0..255")
print(f"\n(b) rozproszenie {gmin}..{gmax} -> {b.min()}..{b.max()}")
print(f"    (max-min)/(max+min) = {(int(b.max()) - int(b.min())) / (int(b.max()) + int(b.min())):.3f}")
print(f"    (max-min)/(G-1)     = {(int(b.max()) - int(b.min())) / 255:.3f}")

# ---------------- profil liniowy: przed i po ----------------
w = g.shape[0] // 2
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(g[w], label="g (torus)"); ax.plot(gF[w], label="g_F (flat)"); ax.plot(gD[w], label="g_D (dark)")
ax.plot(a[w], label="g_C (a)", lw=2)
ax.set_title(f"profil liniowy – wiersz {w}"); ax.set_xlabel("kolumna"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig("z18_profil.png", dpi=110); plt.close(fig)
print("\nZapisano: z18_0_*, z18a_flatfield*, z18b_kontrast1*, z18_profil.png")
