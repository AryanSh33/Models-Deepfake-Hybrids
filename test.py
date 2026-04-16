print(train_ds.samples[:5])
print(val_ds.samples[:5])
from pathlib import Path

p = Path(r"C:\Users\Dell\.cache\kagglehub\datasets\pranabr0y\celebdf-v2image-dataset\versions\1\celebdf-v2image-dataset\Celeb_V2")

print("Exists:", p.exists())
print("Images:", len(list(p.rglob("*.jpg"))))