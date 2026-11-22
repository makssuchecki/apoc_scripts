"""Zadanie 14 – podkreślenie krawędzi kolorem.

(a) operacje ARYTMETYCZNE: krawędzie na czerwono na obrazie szarym bakterie.png
        R = f + E,   G = f - E,   B = f - E        (8-bit, wynik obcinany do 0..255)
(b) operacje LOGICZNE: krawędzie na cyjan (0,255,255) na obrazie bakterieRGB.png
        wynik = (RGB AND NOT E)  OR  (E AND (0,255,255))

E = bakterie_krawedzie.png: krawędź = 255 (1111 1111), tło = 0 (0000 0000)

Uruchomienie:  python zad14.py bakterie.png bakterieRGB.png bakterie_krawedzie.png
Wymaga: numpy, pillow
"""
import sys

import numpy as np
from PIL import Image

pliki = sys.argv[1:4] if len(sys.argv) >= 4 else ["bakterie.png", "bakterieRGB.png", "bakterie_krawedzie.png"]
f = np.array(Image.open(pliki[0]).convert("L")).astype(int)          # szary, 0..255
rgb = np.array(Image.open(pliki[1]).convert("RGB")).astype(np.uint8)  # kolorowy (bez kanału alfa)
E = np.array(Image.open(pliki[2]).convert("L"))
E = np.where(E > 127, 255, 0).astype(np.uint8)                       # pewność, że jest 0/255
print(f"krawędzie: {np.mean(E == 255) * 100:.2f}% pikseli")

# ---------------- (a) arytmetyka ----------------
# na krawędzi (E=255): R = f+255 -> 255,  G = B = f-255 -> 0   => (255, 0, 0) czerwony
# poza krawędzią (E=0): R = G = B = f                           => szary jak w oryginale
R = np.clip(f + E, 0, 255)
G = np.clip(f - E, 0, 255)
B = np.clip(f - E, 0, 255)
wynik_a = np.dstack([R, G, B]).astype(np.uint8)
Image.fromarray(wynik_a).save("z14a_czerwone_krawedzie.png")
print("(a) kolor na krawędziach:", np.unique(wynik_a[E == 255].reshape(-1, 3), axis=0))

# ---------------- (b) logika ----------------
E3 = np.dstack([E, E, E])                       # maska krawędzi w 3 kanałach
cyjan = np.array([0, 255, 255], dtype=np.uint8)

wyciete = rgb & ~E3          # krok 1: AND z negacją maski -> krawędzie = (0,0,0), reszta bez zmian
kolor = E3 & cyjan           # krok 2: AND maski z cyjanem -> krawędzie = (0,255,255), reszta = 0
wynik_b = wyciete | kolor    # krok 3: OR -> wklejenie cyjanu w wycięte miejsca

Image.fromarray(wyciete).save("z14b_krok1_wyciete.png")
Image.fromarray(kolor).save("z14b_krok2_kolor.png")
Image.fromarray(wynik_b).save("z14b_cyjan_krawedzie.png")
print("(b) kolor na krawędziach:", np.unique(wynik_b[E == 255].reshape(-1, 3), axis=0))
print("(b) reszta obrazu bez zmian:", bool((wynik_b[E == 0] == rgb[E == 0]).all()))

# przykład bitowy dla jednego piksela krawędzi
ys, xs = np.nonzero(E == 255)
i = int(np.argmax(rgb[ys, xs].astype(int).sum(axis=1)))   # najjaśniejszy piksel krawędzi
y, x = ys[i], xs[i]
p = rgb[y, x]
print(f"\nprzykład piksela krawędzi ({x},{y}): RGB = {tuple(int(v) for v in p)}")
for nazwa, v, c in zip("RGB", p, cyjan):
    print(f"  {nazwa}: ({v:08b} AND 00000000) OR (11111111 AND {c:08b}) = {int((v & 0) | (255 & c)):08b} = {int(c)}")
print("\nZapisano: z14a_czerwone_krawedzie.png, z14b_krok1_wyciete.png, z14b_krok2_kolor.png, z14b_cyjan_krawedzie.png")