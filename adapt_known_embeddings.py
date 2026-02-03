import numpy as np
from pathlib import Path
from face_embedder import FaceEmbedder
import shutil

# ================= CONFIG =================
INPUT_EMB_DIR  = "embeddings"          # existing embeddings
OUTPUT_EMB_DIR = "embeddings_adapted"  # new adapted embeddings
ADAPTER_PATH   = "final_embedding_adapter.pt"
DEVICE         = "cuda"
# ========================================


def main():
    input_dir = Path(INPUT_EMB_DIR)
    output_dir = Path(OUTPUT_EMB_DIR)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    embedder = FaceEmbedder(adapter_path=ADAPTER_PATH,device=DEVICE)

    npy_files = list(input_dir.glob("*.npy"))
    print(f"Found {len(npy_files)} embeddings to adapt")

    for npy_path in npy_files:
        name = npy_path.stem

        emb = np.load(npy_path).astype("float32")

        # safety normalize (old embeddings may or may not be normalized)
        emb = emb / (np.linalg.norm(emb) + 1e-8)

        # APPLY ADAPTER
        emb_adapted = embedder.adapt(emb)

        out_path = output_dir / f"{name}.npy"
        np.save(out_path, emb_adapted)

        print(f"Adapted: {name}")

    print("\nAll embeddings adapted successfully")
    print(f"Saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

