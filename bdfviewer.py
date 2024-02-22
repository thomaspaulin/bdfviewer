import math
from dataclasses import dataclass
from pathlib import Path

import click
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
from PIL import Image

# https://stackoverflow.com/questions/65115921/2d-boolean-array-to-image
@dataclass
class Glyph:
    char: str
    pixels: np.ndarray

# https://stackoverflow.com/a/31064279/4427782
# First off, lets convert a byte represented by a string consisting of two hex digits
# into an array of 8 bits (represented by ints).
def hex_byte_to_bits(hex_byte):
    # todo something isn't right here
    binary_byte = bin(int(hex_byte, base=16))
    # Use zfill to pad the string with zeroes as we want all 8 digits of the byte.
    bits_string = binary_byte[2:].zfill(8)
    return [int(bit) for bit in bits_string]

def simplify_name(input_string: str) -> str:
    number_mapping = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        "space": "_space", "underscore": "_", "period": "dot", "hyphen": "-", "parenleft": "(", "parenright": "(",
        "semicolon": ";"
    }

    return str(number_mapping.get(input_string.lower(), input_string))
def extract_chars(font_file: Path) -> list[Glyph]:
    char_height = 12

    with open(font_file, "r", encoding="UTF-8") as f:
        lines = f.readlines()

        in_char_context = False
        in_bitmap_context = False
        bitmap_line_no = 0

        glyphs: list[Glyph] = []
        g: Glyph | None = None

        for line in lines:
            l = line.strip()
            tokens = l.split(" ")

            # todo need bbx and bwidth, swidth
            if tokens[0] == "FONTBOUNDINGBOX":
                # char_dims = (int(tokens[1].strip()), 8)
                char_dims = (16, 8)
            elif l.startswith("STARTCHAR"):
                in_char_context = True
                hex_str = tokens[1].lstrip("0")
                if len(hex_str) == len(tokens[1].strip()) and len(hex_str) == 1:
                    # probably not hex, a single character
                    char = simplify_name(hex_str)
                else:
                    try:
                        char = chr(int(hex_str, base=16))
                    except ValueError as e:
                        print(e)
                        char = simplify_name(hex_str)
                g = Glyph(char=char, pixels=np.zeros(shape=char_dims, dtype=np.uint8))
            elif l == "ENDCHAR" and in_char_context:
                in_char_context = False
                in_bitmap_context = False
                bitmap_line_no = 0
                glyphs.append(g)
                g = None
            elif l == "BITMAP":
                in_bitmap_context = True
                bitmap_line_no = 0
            elif in_bitmap_context:
                # lines represent the glyphs now
                g.pixels[bitmap_line_no] = hex_byte_to_bits(l.strip())
                bitmap_line_no += 1
            else:
                print("Skipping line " + l)
        return glyphs


def convert_bdf(font_file: str) -> Path:
    folder: Path = Path("output")
    folder.mkdir(parents=True, exist_ok=True)
    arr: list[Glyph] = extract_chars(Path(font_file))
    for glyph in arr:
        mask_array = np.array(glyph.pixels, dtype=np.uint8) * 255
        img = Image.fromarray(mask_array, mode="L")
        img.save(f"{folder.name}/{glyph.char}.jpg")
    return folder

def display_all(folder: Path):
    images = []
    for i in folder.glob("*.jpg"):
        images.append({"img": mpimg.imread(str(i)), "filename": i.name})

    images.sort(key=lambda x: x["filename"])

    cols = 8
    rows = math.ceil(len(images) / 8)

    _, axs = plt.subplots(cols, rows, figsize=(10, 10))
    axs = axs.flatten()
    for img, ax in zip(images, axs):
        ax.set_title(img["filename"])
        ax.imshow(img["img"])
        ax.axis("off")

    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


@click.command()
@click.option("--fontfile", help='BDF font file to view')
def view_bdf(fontfile: str):
    converted_folder = convert_bdf(fontfile)
    display_all(converted_folder)


if __name__ == "__main__":
    view_bdf()
