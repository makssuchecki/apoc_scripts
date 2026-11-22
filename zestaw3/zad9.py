"""Zadanie 9 – zwiększenie kontrastu przez kolor (pseudokolor wg diagramu).

Trzy funkcje mapowania R(g), G(g), B(g) – odcinkami liniowe, punkty węzłowe z diagramu:
    g :   0   31   95  159  223  255
    R :   0    0  255  255    0    0
    G :   0    0    0  255  255  128
    B : 128  255  255    0    0    0

Uruchomienie:  python zad9.py z7b_wyrownanie.png
(obraz wyjściowy z zadania 7; skrypt zad7.py zapisuje go pod tą nazwą)
Wymaga: numpy, pillow, matplotlib
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

g = np.arange(256)

# ---- 1) funkcje mapowania: interpolacja liniowa między punktami z diagramu ----
R = np.interp(g, [0, 31, 95, 159, 223, 255], [0, 0, 255, 255, 0, 0])
G = np.interp(g, [0, 95, 159, 223, 255], [0, 0, 255, 255, 128])
B = np.interp(g, [0, 31, 95, 159, 255], [128, 255, 255, 0, 0])
lut = np.round(np.c_[R, G, B]).astype(np.uint8)          # tablica 256 x 3: g -> (R, G, B)

# ---- 2) obraz wejściowy (8-bit) ----
sciezka = sys.argv[1] if len(sys.argv) > 1 else "z7b_wyrownanie.png"
img = np.array(Image.open(sciezka).convert("L"))
print(f"wejscie: {sciezka}, {img.shape[1]}x{img.shape[0]}, min={img.min()}, max={img.max()}")

# ---- 3) każdy piksel g zamieniamy na kolor (R(g), G(g), B(g)) ----
rgb = lut[img]                                           # kształt H x W x 3
Image.fromarray(rgb).save("z9_pseudokolor.png")

# ---- 4) wykres funkcji mapowania (jak diagram w zadaniu) ----
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(g, R, "r", label="R(g)"); ax.plot(g, G, "g", label="G(g)"); ax.plot(g, B, "b", label="B(g)")
ax.imshow(lut[None, :, :], extent=(0, 255, -25, -5), aspect="auto")   # pasek kolorów pod osią
ax.set_xlim(0, 255); ax.set_ylim(-25, 260)
ax.set_xticks([0, 31, 95, 159, 223, 255]); ax.set_yticks([0, 128, 255]); ax.legend(fontsize=8)
ax.set_title("funkcje mapowania wartości szarości")
fig.tight_layout(); fig.savefig("z9_funkcje.png", dpi=110); plt.close(fig)

# ---- 5) histogramy kanałów R, G, B wyniku ----
fig, ax = plt.subplots(1, 3, figsize=(11, 2.8))
for i, (kolor, nazwa) in enumerate(zip("rgb", ["Red", "Green", "Blue"])):
    kanal = rgb[..., i]
    ax[i].bar(g, np.bincount(kanal.ravel(), minlength=256), width=1, color=kolor)
    ax[i].set_xlim(0, 255); ax[i].set_yticks([])
    ax[i].set_title(f"{nazwa}: mean={kanal.mean():.1f}, min={kanal.min()}, max={kanal.max()}", fontsize=9)
fig.tight_layout(); fig.savefig("z9_histogramy_rgb.png", dpi=110); plt.close(fig)

# ---- 6) LUT do ImageJ (File > Import > LUT...) ----
np.savetxt("z9_lut.txt", lut, fmt="%d", delimiter="\t")

print("\nprzykladowe kolory:")
for v in (0, 31, 63, 95, 127, 159, 191, 223, 255):
    print(f"  g={v:3d} -> (R,G,B) = {tuple(int(c) for c in lut[v])}")
print("\nZapisano: z9_pseudokolor.png, z9_funkcje.png, z9_histogramy_rgb.png, z9_lut.txt")