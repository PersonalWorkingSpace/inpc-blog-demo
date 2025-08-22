from typing import Tuple, Dict
import os
import argparse
import numpy as np
from PIL import Image
from glob import glob

def compute_red_mask(arr: np.ndarray) -> np.ndarray:
    """
    Calculate red area (boolean array, shape HxW).

    Param:
        arr: uint8 RGBA ndarray (H,W,4)
    Return:
        red_mask: boolean ndarray (H,W)
    """

    f = arr.astype(np.float32) / 255.0
    r = f[..., 0]
    g = f[..., 1]
    b = f[..., 2]
    a = f[..., 3]

    red_mask = (r == 1.) & (g == 0.) & (b == 0.) & (a == 1.)
    return red_mask


def remove_red_frame(arr: np.ndarray,
                     red_fraction_threshold: float = 0.6) -> np.ndarray:
    """
    Cover the red border with transparent pixels

    Param:
        arr: uint8 RGBA ndarray (H,W,4)
        red_fraction_threshold: float
    Return:
        arr: uint8 RGBA ndarray (H,W,4)
    """

    if arr.dtype != np.uint8:
        raise ValueError("arr must be uint8 RGBA")
    if arr.shape[2] != 4:
        raise ValueError("arr must have 4 channels (RGBA)")

    H, W, _ = arr.shape
    red_mask = compute_red_mask(arr)
    red_col_count = red_mask.sum(axis=0)
    red_row_count = red_mask.sum(axis=1)

    nontransparent_mask = arr[:, :, 3] > 0
    nontransparent_col_count = nontransparent_mask.sum(axis=0)
    nontransparent_row_count = nontransparent_mask.sum(axis=1)
    
    def scan_row(y, direction):
        while (y >= 0 and direction == "up" or y < H and direction == "down"):
            if nontransparent_row_count[y] == 0:
                if direction == "down":
                    y += 1
                else:
                    y -= 1
                continue
            
            frac = red_row_count[y] / nontransparent_row_count[y]
            if frac >= red_fraction_threshold:
                cols = np.nonzero(red_mask[y, :])[0]
                if cols.size > 0:
                    arr[y, cols, 0:4] = 0
                    red_mask[y, cols] = 0
            break
    
    scan_row(0, "down") # scan top -> bottom
    scan_row(H-1, "up") # scan bottom -> up
    
    def scan_col(x, direction):
        while (x >= 0 and direction == "left" or x < W and direction == "right"):
            if nontransparent_col_count[x] == 0:
                if direction == "right":
                    x += 1
                else:
                    x -= 1
                continue
            
            frac = red_col_count[x] / nontransparent_col_count[x]
            if frac >= red_fraction_threshold:
                rows = np.nonzero(red_mask[:, x])[0]
                if rows.size > 0:
                    arr[rows, x, 0:4] = 0
                    red_mask[rows, x] = 0
            break

    scan_col(0,  "right") # scan left -> right
    scan_col(W-1, "left") # scan right -> left

    return arr


def remove_red_border_from_image(img: Image.Image,
                                 red_fraction_threshold: float = 0.6) -> Image.Image:
    """
    Remove red border from image
    
    Param:
        img: PIL image
        red_fraction_threshold: float
    Return:
        out_img: PIL image
    """

    rgba = img.convert("RGBA")
    arr = np.array(rgba, dtype=np.uint8)
    out_arr = remove_red_frame(arr, red_fraction_threshold=red_fraction_threshold)
    out_img = Image.fromarray(out_arr, mode="RGBA")
    return out_img


def main():
    parser = argparse.ArgumentParser(description="Remove red border from RGBA PNG by scanning edges.")
    parser.add_argument("--in", dest="indir", required=True, help="input image path (PNG with alpha)")
    parser.add_argument("--out", dest="outdir", required=True, help="output image path")
    parser.add_argument("--frac", dest="frac", type=float, default=0.6, help="red fraction threshold per line")
    args = parser.parse_args()

    images = glob(os.path.join(args.indir, "*.png"), recursive=True)
    os.makedirs(args.outdir, exist_ok=True)
    
    for img_file in images:
        print(f"Process image: {img_file}")
        img = Image.open(img_file)
        out_img = remove_red_border_from_image(img, red_fraction_threshold=args.frac)
        img_name = os.path.basename(img_file)
        out_file = os.path.join(args.outdir, img_name)
        out_img.save(out_file)
        print(f"Done. Save file to {out_file}")


if __name__ == '__main__':
    main()