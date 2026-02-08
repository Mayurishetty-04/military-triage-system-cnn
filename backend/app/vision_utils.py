from PIL import Image, ImageFilter
import io

def is_blurry(image_bytes: bytes, threshold: float = 25.0) -> bool:
    print("🔥 BLUR CHECK FUNCTION CALLED")

    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    edges = img.filter(ImageFilter.FIND_EDGES)

    min_val, max_val = edges.getextrema()
    contrast = max_val - min_val

    print(f"[BLUR] min={min_val} max={max_val} contrast={contrast}")

    return contrast < threshold
