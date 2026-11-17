
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

path = sys.argv[1] if len(sys.argv) > 1 else "PlytkaFresnela.png"
step = int(sys.argv[2]) if len(sys.argv) > 2 else 50

img = np.array(Image.open(path).convert("L"), dtype=float)
n = img.shape[0]
c = (n - 1) / 2
prof = img[n // 2]                      # profil wzdluz srodka (jak linia w ImageJ)
x = np.arange(n)
xs = np.arange(0, n, step)              # punkty probkowania

K = 2 * math.pi * 2560                  # f(r) = r / K [cykli/px]
r_crit = K / (2 * step)                 # f(r) > fs/2 -> aliasing
fmin = 2 * (n / 2) / K

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(x, prof, lw=0.8, label="profil")
ax.plot(xs, prof[xs], "ro", label=f"probki co {step} px")
ax.plot(xs, prof[xs], "r--", lw=0.8, label="rekonstrukcja z probek")
fig.tight_layout(); fig.savefig("zad1a_profil_probki.png", dpi=150)

