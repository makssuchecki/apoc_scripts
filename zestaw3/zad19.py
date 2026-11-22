
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

obraz = np.array(Image.open(sys.argv[1] if len(sys.argv) > 1 else "ZnalezienieSali_modified.png").convert("RGB"))
H, W = obraz.shape[:2]
KANALY = "RGB"

# ---------------- (a) przegląd płaszczyzn bitowych ----------------
# płaszczyzna n = (kanał AND 2^n) > 0 ; pokazujemy ją jako 0/255
fig, ax = plt.subplots(3, 8, figsize=(20, 6.5))
print("udział jedynek w płaszczyznach bitowych (zwykły obraz: ok. 0.5, ukryty obraz: wyraźnie inaczej)")
for c in range(3):
    udzialy = []
    for n in range(8):
        plaszczyzna = (obraz[..., c] & (1 << n)) > 0
        udzialy.append(plaszczyzna.mean())
        ax[c, n].imshow(plaszczyzna, cmap="gray"); ax[c, n].axis("off")
        ax[c, n].set_title(f"{KANALY[c]}  bit {n}  (AND {1 << n:08b})", fontsize=8)
    print(f"  {KANALY[c]}: " + "  ".join(f"b{n}={u:.3f}" for n, u in enumerate(udzialy)))
fig.tight_layout(); fig.savefig("z19a_wszystkie_plaszczyzny.png", dpi=80); plt.close(fig)

# kanał i bit z ukrytym obrazem: płaszczyzna, która najbardziej odbiega od "losowej" (0.5)
odchylenie = [[abs(((obraz[..., c] >> n) & 1).mean() - 0.5) for n in range(3)] for c in range(3)]
c_ukr, n_ukr = np.unravel_index(np.argmax(odchylenie), (3, 3))
ukryty = ((obraz[..., c_ukr] >> n_ukr) & 1) * 255
Image.fromarray(ukryty.astype(np.uint8)).save("z19a_ukryty_obraz.png")
print(f"\n(a) ukryty obraz: kanał {KANALY[c_ukr]}, bit {n_ukr}  -> z19a_ukryty_obraz.png")

# ---------------- (b) ukrycie własnego obrazu ----------------
if len(sys.argv) > 2:
    wiad = Image.open(sys.argv[2]).convert("L").resize((W, H))
else:
    wiad = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(wiad)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 52)
    except OSError:
        try:
            font = ImageFont.truetype("arial.ttf", 52)
        except OSError:
            font = ImageFont.load_default()
    d.text((W // 2, H // 2 - 50), "Zestaw 3 - PO", font=font, fill=255, anchor="mm")
    d.text((W // 2, H // 2 + 40), "ukryta wiadomość: π ≈ 3.14159", font=font, fill=255, anchor="mm")
    d.rectangle([20, 20, W - 21, H - 21], outline=255, width=8)
bity = (np.array(wiad) > 127).astype(np.uint8)        # 0/1

maska_zerujaca = np.uint8(0xFF ^ (1 << n_ukr))          # np. 1111 1110 dla bitu 0
nowy = obraz.copy()
nowy[..., c_ukr] = (nowy[..., c_ukr] & maska_zerujaca) | (bity << n_ukr)   # wyzeruj bit, wstaw nowy
Image.fromarray(nowy).save("z19b_ukryty_wlasny.png")   # PNG = bezstratny (JPEG zniszczyłby bit!)

# kontrola: odczyt z zapisanego pliku
odczyt = (np.array(Image.open("z19b_ukryty_wlasny.png"))[..., c_ukr] >> n_ukr) & 1
Image.fromarray((odczyt * 255).astype(np.uint8)).save("z19b_kontrola_odczytu.png")
print(f"(b) maska zerująca = {maska_zerujaca:08b}, maks. zmiana piksela = "
      f"{np.abs(nowy.astype(int) - obraz.astype(int)).max()}, odczyt poprawny: {np.array_equal(odczyt, bity)}")
print("\nZapisano: z19a_wszystkie_plaszczyzny.png, z19a_ukryty_obraz.png, "
      "z19b_ukryty_wlasny.png (do oddania), z19b_kontrola_odczytu.png")