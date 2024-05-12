import logging
import math
import os
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

import click
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk

logger = logging.getLogger(__name__)


# https://stackoverflow.com/questions/65115921/2d-boolean-array-to-image
@dataclass
class Glyph:
    char: str
    pixels: np.ndarray


@dataclass
class GlyphContext:
    char: str
    shape: tuple[int, int]
    pixels: np.ndarray


@dataclass
class GlyphFile:
    char: str
    img_file: Path
    dims: tuple[int, int]


@dataclass
class RunMetadata:
    directory: Path
    glyphs: list[GlyphFile]


# https://stackoverflow.com/a/31064279/4427782
# First off, lets convert a byte represented by a string consisting of two hex digits
# into an array of 8 bits (represented by ints).
def hex_byte_to_bits(hex_byte, num_bits):
    bin_str = ""
    # hex-encoded bitmap, padded on the right with zeroes to the nearest byte (that is, multiple of 8).
    # Hex data can be turned into binary by taking two bytes at a time, each of which represents 4 bits of the 8-bit
    # value. For example, the byte 01101101 is two hex digits: 6 (0110 in hex) and D (1101 in hex)
    for hex_str in hex_byte:
        bin_str += bin(int(hex_str, 16))[2:].zfill(4)
    # Use zfill to pad the string with zeroes as we want all 8 digits of the byte.

    pixels = [int(bit) for bit in bin_str[:num_bits]]  # num_bits truncates if the BDF wants width to be below a multiple of 8
    return pixels


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

        in_bitmap_context = False
        bitmap_line_no = 0

        glyphs: list[Glyph] = []
        g: Glyph | None = None

        overall_dims: tuple[int, int]

        context: GlyphContext | None = None
        for line in lines:
            try:
                l = line.strip()
                tokens = l.split(" ")
                if tokens[0] == "FONTBOUNDINGBOX":
                    w = int(tokens[1])
                    h = int(tokens[2])
                    overall_dims = (h, w)  # rows by columns
                elif l.startswith("STARTCHAR"):
                    char = simplify_name(tokens[1].strip())
                    context = GlyphContext(char=char, shape=overall_dims, pixels=np.zeros(shape=(1, 1), dtype=np.uint8))
                elif l.startswith("BBX") and context is not None:
                    w = int(tokens[1])
                    h = int(tokens[2])
                    context.shape = (h, w)  # rows by columns
                elif l == "ENDCHAR" and context is not None:
                    in_bitmap_context = False
                    bitmap_line_no = 0
                    glyphs.append(Glyph(char=context.char, pixels=context.pixels))
                    context = None
                elif l == "BITMAP":
                    in_bitmap_context = True
                    bitmap_line_no = 0
                    context.pixels = np.zeros(shape=context.shape, dtype=np.uint8)
                elif in_bitmap_context:
                    # lines represent the glyphs now
                    bits = hex_byte_to_bits(l.strip(), context.shape[1])
                    context.pixels[bitmap_line_no] = bits
                    bitmap_line_no += 1
                else:
                    logger.debug("Skipping line " + l)
            except ValueError as e:
                logger.exception(e)
                logger.error(f"Failed for line {line} with context {context}")
                raise e
        return glyphs


def convert_bdf(font_file: str) -> RunMetadata:
    folder: Path = Path("output")
    folder.mkdir(parents=True, exist_ok=True)
    arr: list[Glyph] = extract_chars(Path(font_file))
    success_count = 0
    failed_glyphs = []
    empty_glyphs = []
    glyphs: list[GlyphFile] = []
    for glyph in arr:
        try:
            if not np.any(glyph.pixels):
                empty_glyphs.append(glyph)
                continue
            mask_array = np.array(glyph.pixels, dtype=np.uint8) * 255
            img = Image.fromarray(mask_array, mode="L")
            img_path = Path(folder.name, f"{glyph.char}.jpg")
            img.save(str(img_path.resolve()))
            glyphs.append(GlyphFile(char=glyph.char, img_file=img_path, dims=img.size))
            success_count += 1
        except Exception as e:
            failed_glyphs.append(glyph)
            logger.warning(f"Error while converting {glyph} - {e}")
    logger.info(f"Conversion results: successes={success_count}, failures={len(failed_glyphs)}")
    logger.info(f"Failed glyphs: {[g.char for g in failed_glyphs]}")
    logger.info(f"Empty glyphs: {[g.char for g in empty_glyphs]}")
    metadata = RunMetadata(directory=folder, glyphs=glyphs)
    if success_count == 0:
        raise ValueError("Failed to convert any glyphs")
    return metadata


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


class BdfViewerGUI:
    def __init__(self, root, metadata: RunMetadata):
        self.root = root
        self.root.title("Image Selector App")

        self.metadata = metadata

        # Initialize variables
        self.image_dir = metadata.directory
        self.glyphs = {g.char: g for g in metadata.glyphs}

        # Create dropdown selection
        self.dropdown_label = ttk.Label(root, text="Select a glyph:")
        self.dropdown_label.pack()

        self.dropdown = ttk.Combobox(root, values=["Show All"] + [g for g in self.glyphs.keys()])
        self.dropdown.current(0)
        self.dropdown.bind("<<ComboboxSelected>>", self.select_image)
        self.dropdown.pack()

        # Create image display area
        self.frame = ttk.Frame(root)
        self.frame.pack(pady=10)

        # Display thumbnails initially
        self.show_all_thumbnails(self.frame)

    def load_images(self):
        # Load image files from the specified directory
        image_files = []
        for file in os.listdir(self.image_dir):
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                image_files.append(file)
        return image_files

    def select_image(self, event=None):
        selected_option = self.dropdown.get()
        if selected_option == "Show All":
            self.show_all_thumbnails(self.frame)
        else:
            self.display_selected_image(selected_option, self.frame)

    def display_selected_image(self, selected_image, frame):
        for widget in frame.winfo_children():
            widget.destroy()
        # Load and display the selected image
        g = self.glyphs.get(selected_image)
        selected_image_path = os.path.join(self.image_dir, str(g.img_file.resolve()))
        image = Image.open(selected_image_path)
        image = image.resize((50, 50))  # Resize image for display
        photo = ImageTk.PhotoImage(image)
        image_label = ttk.Label(frame, image=photo)
        image_label.image = photo
        image_label.pack()

    def show_all_thumbnails(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()

        num_columns = 16

        # Display thumbnails and captions in a grid layout
        for i, glyph in enumerate(self.glyphs.values()):
            image_path = str(glyph.img_file.resolve())
            image = Image.open(image_path).resize(np.multiply(glyph.dims, 3))
            photo = ImageTk.PhotoImage(image)
            # Calculate row and column index for the current thumbnail
            row = i // num_columns
            column = i % num_columns

            # Create a label for the thumbnail image
            thumbnail_label = ttk.Label(frame, image=photo)
            thumbnail_label.image = photo
            thumbnail_label.grid(row=row, column=column, padx=5, pady=5)


@click.command()
@click.option("--fontfile", help='BDF font file to view')
def view_bdf(fontfile: str):
    converted_folder = convert_bdf(fontfile)

    root = tk.Tk()
    BdfViewerGUI(root, converted_folder)
    root.mainloop()


if __name__ == "__main__":
    view_bdf()
