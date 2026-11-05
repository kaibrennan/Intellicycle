import json
import os
import re
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps
import serial 

#-------------------- macOS zbar --------------------

import ctypes.util
import platform
import subprocess
from pathlib import Path

def configure_macos_zbar():
    """Help pyzbar locate Homebrew's ZBar library on macOS."""
    if platform.system() != "Darwin":
        return

    try:
        prefix = subprocess.check_output(
            ["brew", "--prefix", "zbar"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return

    library_path = Path(prefix) / "lib" / "libzbar.dylib"

    if not library_path.exists():
        return

    original_find_library = ctypes.util.find_library

    def find_library(name):
        if name == "zbar":
            return str(library_path)
        return original_find_library(name)

    ctypes.util.find_library = find_library


configure_macos_zbar()

from pyzbar.pyzbar import ZBarSymbol, decode
# -------------------- Configuration --------------------

CAMERA_INDEX = 0
ENABLE_LOOKUP = True

CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
SCAN_INTERVAL = 0.15

PREVIEW_WIDTH = 1000
PREVIEW_HEIGHT = 450
PANEL_HEIGHT = 470
WINDOW_NAME = "Packaging Barcode Scanner"

CONTACT_EMAIL = os.getenv(
    "SCANNER_CONTACT_EMAIL",
    "your-email@example.com",
)
USER_AGENT = f"PackagingScannerPrototype/1.0 ({CONTACT_EMAIL})"
API_URL = "https://world.openfoodfacts.org/api/v2/product/{}.json"

LOCAL_DATABASE_FILE = Path("local_products.json")

# Project acceptance rules, NOT verified local recycling eligibility.
ACCEPTED_TYPES = {
    ("bottle", "plastic"): True,
    ("can", "aluminum"): True,
    ("bottle", "glass"): True,
}

# Optional built-in records. Enter only physically verified products.
# A local JSON file can also provide records in the same format.
LOCAL_PRODUCTS = {
    # "YOUR_BARCODE": {
    #     "name": "Verified demonstration product",
    #     "components": [
    #         {"shape": "bottle", "material": "plastic"},
    #         {"shape": "cap", "material": "plastic"}
    #     ]
    # }
}

RED = (0, 0, 255)
GREEN = (0, 220, 0)
WHITE = (255, 255, 255)
GRAY = (175, 175, 175)
YELLOW = (0, 220, 255)

FORMATS = [
    ZBarSymbol.EAN13,
    ZBarSymbol.EAN8,
    ZBarSymbol.UPCA,
    ZBarSymbol.UPCE,
    ZBarSymbol.CODE128,
    ZBarSymbol.CODE39,
    ZBarSymbol.I25,
]

# Exact aliases: extend these after inspecting actual API responses.
SHAPE_ALIASES = {
    "bottle": "bottle",
    "bottles": "bottle",
    "can": "can",
    "cans": "can",
    "beverage-can": "can",
    "jar": "jar",
    "jars": "jar",
    "cap": "cap",
    "caps": "cap",
    "lid": "lid",
    "lids": "lid",
    "label": "label",
    "labels": "label",
    "film": "film",
    "wrapper": "wrapper",
    "box": "box",
    "carton": "carton",
    "tray": "tray",
    "bag": "bag",
}

MATERIAL_ALIASES = {
    "plastic": "plastic",
    "pet": "plastic",
    "pete": "plastic",
    "pet-polyethylene-terephthalate": "plastic",
    "polyethylene-terephthalate": "plastic",
    "hdpe": "plastic",
    "hdpe-high-density-polyethylene": "plastic",
    "ldpe": "plastic",
    "ldpe-low-density-polyethylene": "plastic",
    "pp": "plastic",
    "pp-polypropylene": "plastic",
    "polypropylene": "plastic",
    "glass": "glass",
    "clear-glass": "glass",
    "green-glass": "glass",
    "brown-glass": "glass",
    "aluminum": "aluminum",
    "aluminium": "aluminum",
    "steel": "steel",
    "paper": "paper",
    "cardboard": "cardboard",
}

MAIN_CONTAINER_SHAPES = {
    "bottle", "can", "jar", "box", "carton", "tray", "bag"
}


# -------------------- Normalization --------------------

def taxonomy_slug(value):
    if not isinstance(value, str):
        return ""
    value = value.strip().lower()
    return re.sub(r"^[a-z]{2}:", "", value)


def normalize_shape(value):
    return SHAPE_ALIASES.get(taxonomy_slug(value), "unknown")


def normalize_material(value):
    return MATERIAL_ALIASES.get(taxonomy_slug(value), "unknown")


def readable(value):
    if not isinstance(value, str) or not value.strip():
        return "Unavailable"
    return taxonomy_slug(value).replace("-", " ")


def barcode_key(barcode):
    """Treat equivalent UPC/EAN/GTIN leading-zero forms as one local key."""
    value = str(barcode).strip()
    return value.zfill(14) if value.isdigit() else value


def load_local_database():
    records = dict(LOCAL_PRODUCTS)

    if LOCAL_DATABASE_FILE.exists():
        try:
            loaded = json.loads(
                LOCAL_DATABASE_FILE.read_text(encoding="utf-8")
            )
            if not isinstance(loaded, dict):
                raise ValueError("The JSON root must be an object.")
            records.update(loaded)
        except (OSError, ValueError) as error:
            print(f"Local database warning: {error}")

    return {
        barcode_key(code): record
        for code, record in records.items()
        if isinstance(record, dict)
    }


# -------------------- Packaging classification --------------------

def normalize_components(entries):
    components = []

    if not isinstance(entries, list):
        return components

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        raw_shape = entry.get("shape")
        raw_material = entry.get("material")

        components.append({
            "shape": normalize_shape(raw_shape),
            "material": normalize_material(raw_material),
            "shape_label": readable(raw_shape),
            "material_label": readable(raw_material),
        })

    return components


def evaluate_components(components, source):
    """Evaluate main containers only, never caps or labels."""
    containers = [
        component
        for component in components
        if component["shape"] in MAIN_CONTAINER_SHAPES
    ]

    result = {
        "source": source,
        "components": components,
        "material": "Unknown",
        "accepted": None,
        "reason": "Main container shape/material is not established.",
    }

    if len(containers) > 1:
        result["reason"] = (
            "Multiple main container components; manual check required."
        )
        return result

    if not containers:
        return result

    container = containers[0]
    shape = container["shape"]
    material = container["material"]

    result["material"] = (
        f"{material.title()} {shape}"
        if material != "unknown"
        else f"Unknown-material {shape}"
    )

    if material == "unknown":
        result["reason"] = "Main container material is missing or unmapped."
        return result

    decision = ACCEPTED_TYPES.get((shape, material))
    result["accepted"] = decision

    if decision is True:
        result["reason"] = "Main container matches project acceptance rules."
    elif decision is False:
        result["reason"] = "Main container is not accepted by project rules."
    else:
        result["reason"] = "No project rule exists for this container type."

    return result


def text_components(packaging):
    """
    Conservative English-text fallback.

    First associate shape/material within individual descriptions.
    If only one shape and material occur in all the text, allow a
    whole-text match such as "Plastic, PET, Bottle".

    Never combine "plastic cap, glass bottle" into a plastic bottle.
    """
    if not isinstance(packaging, str) or not packaging.strip():
        return []

    def extract(text):
        tokens = set(re.findall(r"[a-z]+", text.lower()))
        shapes = {
            SHAPE_ALIASES[token]
            for token in tokens
            if token in SHAPE_ALIASES
        }
        materials = {
            MATERIAL_ALIASES[token]
            for token in tokens
            if token in MATERIAL_ALIASES
        }
        return shapes, materials

    def make_component(shape, material):
        return {
            "shape": shape,
            "material": material,
            "shape_label": shape,
            "material_label": material,
        }

    components = []

    for description in re.split(r"[,;\n]+", packaging):
        shapes, materials = extract(description)

        if len(shapes) == 1:
            shape = next(iter(shapes))
            material = (
                next(iter(materials))
                if len(materials) == 1
                else "unknown"
            )
            components.append(make_component(shape, material))

    shapes, materials = extract(packaging)

    # Safe only within the limits of this simple text heuristic.
    if len(shapes) == 1 and len(materials) == 1:
        return [
            make_component(
                next(iter(shapes)),
                next(iter(materials)),
            )
        ]

    return components


def resolve_material(barcode, product, local_database):
    local = local_database.get(barcode_key(barcode))

    if local is not None:
        return evaluate_components(
            normalize_components(local.get("components")),
            "Local verified database",
        )

    structured = normalize_components(product.get("packagings"))

    if structured:
        result = evaluate_components(
            structured,
            "Open Food Facts structured packaging",
        )

        # Preserve ambiguous structured records rather than overriding
        # them with a weaker text inference.
        if any(
            component["shape"] in MAIN_CONTAINER_SHAPES
            for component in structured
        ):
            return result

    fallback = text_components(product.get("packaging"))

    if fallback:
        return evaluate_components(
            fallback,
            "Packaging text fallback (heuristic)",
        )

    return evaluate_components(
        structured,
        "Unknown / manual review",
    )


# -------------------- Product lookup --------------------

def lookup_product(barcode, local_database):
    local = local_database.get(barcode_key(barcode))
    product = {}
    status = "Online lookup disabled."

    if ENABLE_LOOKUP:
        try:
            response = requests.get(
                API_URL.format(barcode),
                params={
                    "fields": (
                        "product_name,product_name_en,generic_name,"
                        "brands,categories,packaging,packagings"
                    )
                },
                headers={"User-Agent": USER_AGENT},
                timeout=(3, 8),
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") == 1:
                candidate = data.get("product")
                if not isinstance(candidate, dict):
                    raise ValueError("Invalid product record.")
                product = candidate
                status = "Product record found."
            else:
                status = "Product not found online."

        except (requests.RequestException, ValueError, AttributeError) as error:
            status = f"Online lookup failed: {error}"

    name = (
        (local or {}).get("name")
        or product.get("product_name_en")
        or product.get("product_name")
        or product.get("generic_name")
        or "Unknown product"
    )

    return {
        "status": status,
        "name": name,
        "brand": product.get("brands") or "Unavailable",
        "category": product.get("categories") or "Unavailable",
        "packaging": product.get("packaging") or "Unavailable",
        "recycling": resolve_material(
            barcode, product, local_database
        ),
    }


# -------------------- Barcode and image scanning --------------------

def decode_first(image):
    for item in decode(image, symbols=FORMATS):
        barcode = item.data.decode("utf-8", errors="replace").strip()
        if barcode:
            return {"barcode": barcode, "format": item.type}
    return None


def scan_image(filename):
    image = None

    try:
        with Image.open(filename) as original:
            rgb = np.array(
                ImageOps.exif_transpose(original).convert("RGB")
            )

        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        result = decode_first(gray)
        if result:
            return image, result, "Barcode read at original resolution."

        for scale in (0.5, 0.75, 1.5):
            if scale > 1 and max(gray.shape) > 3000:
                continue
            resized = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=(
                    cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
                ),
            )
            result = decode_first(resized)
            if result:
                return image, result, "Barcode read from rescaled image."

        enhanced = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
        result = decode_first(enhanced)

        if not result:
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            result = decode_first(binary)

        if not result:
            result = decode_first(
                cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
            )

        return (
            image,
            result,
            "Barcode read." if result else "No barcode found.",
        )

    except Exception as error:
        return image, None, f"Image processing failed: {error}"


def choose_image():
    """
    Select a local image.

    macOS:
        Use the native file chooser through AppleScript, without Tkinter.

    Windows / Linux:
        Use Tkinter in a separate Python process.

    Call this through the existing picker_worker, not directly from
    OpenCV's main loop.
    """
    import json
    import subprocess
    import sys

    if sys.platform == "darwin":
        script = '''
        try
            activate
            set selectedFile to choose file with prompt ¬
                "Select a screenshot or barcode image"
            return POSIX path of selectedFile
        on error number -128
            return ""
        end try
        '''

        try:
            result = subprocess.run(
                ["/usr/bin/osascript", "-e", script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=180,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                "File selection timed out. Press U to try again."
            ) from error

        if result.returncode != 0:
            details = result.stderr.strip()
            raise RuntimeError(
                "macOS file picker failed: "
                + (details or "No error details available.")
            )

        # Remove the newline printed by osascript.
        return result.stdout.rstrip("\r\n")

    # Retain a separate-process Tkinter picker for Windows / Linux.
    picker_script = r'''
import json
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

try:
    root.attributes("-topmost", True)
    filename = filedialog.askopenfilename(
        parent=root,
        title="Select a screenshot or barcode image",
        filetypes=[
            (
                "Image files",
                ("*.png", "*.jpg", "*.jpeg", "*.bmp",
                 "*.tif", "*.tiff", "*.webp")
            ),
            ("All files", "*"),
        ],
    )
    print(json.dumps(filename), flush=True)
finally:
    root.destroy()
'''

    try:
        result = subprocess.run(
            [sys.executable, "-c", picker_script],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "File selection timed out. Press U to try again."
        ) from error

    if result.returncode != 0:
        details = result.stderr.strip()
        raise RuntimeError(
            "File picker failed: "
            + (details[-1500:] if details else "No error details available.")
        )

    output = result.stdout.strip()
    if not output:
        return ""

    try:
        filename = json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "The file picker returned an unexpected response."
        ) from error

    if not isinstance(filename, str):
        raise RuntimeError("The file picker did not return a file path.")

    return filename
# -------------------- Display --------------------

def draw_text(image, text, position, color=WHITE):
    safe = str(text).encode("ascii", errors="replace").decode("ascii")
    cv2.putText(
        image, safe, position,
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA,
    )


def preview_image(image):
    canvas = np.zeros(
        (PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), dtype=np.uint8
    )

    if image is None:
        draw_text(
            canvas,
            "No image available. Press U to select a file.",
            (30, 100),
            YELLOW,
        )
        return canvas

    height, width = image.shape[:2]
    scale = min(PREVIEW_WIDTH / width, PREVIEW_HEIGHT / height)
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
    )

    h, w = resized.shape[:2]
    x, y = (PREVIEW_WIDTH - w) // 2, (PREVIEW_HEIGHT - h) // 2
    canvas[y:y + h, x:x + w] = resized
    return canvas


def compose_display(preview, rows):
    panel = np.full(
        (PANEL_HEIGHT, PREVIEW_WIDTH, 3), 25, dtype=np.uint8
    )
    y = 24

    for text, color in rows:
        lines = textwrap.wrap(str(text), width=108) or [""]
        if len(lines) > 2:
            lines = lines[:2]
            lines[-1] = lines[-1][:102] + "..."

        for line in lines:
            if y > PANEL_HEIGHT - 10:
                return np.vstack((preview, panel))
            draw_text(panel, line, (14, y), color)
            y += 22

    return np.vstack((preview, panel))


def request_autofocus(camera):
    if not camera.isOpened():
        return "No webcam; image scanning remains available."
    if camera.set(cv2.CAP_PROP_AUTOFOCUS, 1):
        return "Autofocus requested; hardware support required."
    return "Camera/backend did not accept autofocus."


# -------------------- Update public facing output for raspi ---------------_#
def setoutput(newval):
    with open("results.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    data["Output"] = newval

    with open("results.json", "w", encoding="utf-8") as file:
        json.dump(data, file, indent=1, ensure_ascii=False)


# ----  Arduino Section ----#
arduino = serial.Serial(port='COM4', baudrate=9600, timeout=.1) 

def arduinowrite(x): 
    arduino.write(bytes(x, 'utf-8')) 
    time.sleep(0.05) 
    data = arduino.readline() 
    return data


# -------------------- Main application --------------------

def main():
    local_database = load_local_database()
    camera = cv2.VideoCapture(CAMERA_INDEX)

    if camera.isOpened():
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    autofocus_status = request_autofocus(camera)
    image_worker = ThreadPoolExecutor(max_workers=1)
    lookup_worker = ThreadPoolExecutor(max_workers=2)
    picker_worker = ThreadPoolExecutor(max_workers=1)

    picker_future = None

    image_future = None
    lookup_future = None
    uploaded_image = None
    image_path = None
    scan = None
    product = None
    mode = "webcam"
    message = "Place the entire barcode inside the red box."
    last_scan = 0.0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        WINDOW_NAME, PREVIEW_WIDTH, PREVIEW_HEIGHT + PANEL_HEIGHT
    )

    def reset_scan():
        nonlocal scan, product, lookup_future
        if lookup_future is not None:
            lookup_future.cancel()
        lookup_future = None
        scan = None
        product = None

    def accept_scan(result):
        nonlocal scan, lookup_future
        scan = result
        print(f"\nBarcode: {scan['barcode']} ({scan['format']})")
        lookup_future = lookup_worker.submit(
            lookup_product, scan["barcode"], local_database
        )

    try:
        while True:

                        # Check the picker without blocking OpenCV's event loop.
            if picker_future is not None and picker_future.done():
                completed_picker = picker_future
                picker_future = None

                try:
                    selected = completed_picker.result()
                except Exception as error:
                    message = f"File picker failed: {error}"
                else:
                    if selected:
                        reset_scan()
                        mode = "image"
                        image_path = selected
                        uploaded_image = None
                        message = "Scanning original-resolution image..."

                        image_future = image_worker.submit(
                            scan_image,
                            selected,
                        )
                    else:
                        message = "Image selection cancelled."
            if image_future is not None and image_future.done():
                uploaded_image, result, message = image_future.result()
                image_future = None
                if result:
                    accept_scan(result)

            if lookup_future is not None and lookup_future.done():
                product = lookup_future.result()
                lookup_future = None
                print(json.dumps(product, indent=2, ensure_ascii=False))

            if mode == "webcam":
                ok, frame = (
                    camera.read() if camera.isOpened() else (False, None)
                )

                if ok:
                    h, w = frame.shape[:2]
                    x1, x2 = int(w * 0.11), int(w * 0.89)
                    y1, y2 = int(h * 0.325), int(h * 0.675)

                    if (
                        scan is None
                        and time.monotonic() - last_scan >= SCAN_INTERVAL
                    ):
                        last_scan = time.monotonic()
                        gray = cv2.cvtColor(
                            frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY
                        )
                        result = decode_first(gray)
                        if result:
                            accept_scan(result)
                            message = "Barcode read. Press R for next item."

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2),
                        GREEN if scan else RED, 4,
                    )
                    preview = preview_image(frame)
                else:
                    preview = preview_image(None)

            else:
                preview = preview_image(uploaded_image)
                cv2.rectangle(
                    preview,
                    (3, 3),
                    (PREVIEW_WIDTH - 4, PREVIEW_HEIGHT - 4),
                    GREEN if scan else RED,
                    3,
                )

            rows = [
                (
                    "U: image | W: webcam | R: reset/rescan | "
                    "F: autofocus | Q/Esc: quit",
                    GRAY,
                ),
                (
                    f"Source: {mode}"
                    + (
                        f" - {Path(image_path).name}"
                        if mode == "image" and image_path else ""
                    ),
                    WHITE,
                ),
                (message, YELLOW),
            ]

            if mode == "webcam":
                rows.append((autofocus_status, GRAY))

            if scan:
                rows.append(
                    (
                        f"Barcode: {scan['barcode']} | "
                        f"Format: {scan['format']}",
                        GREEN,
                    )
                )

                if product is None:
                    rows.append(("Resolving product and packaging...", YELLOW))
                else:
                    recycling = product["recycling"]
                    accepted = recycling["accepted"]
                    color = (
                        GREEN if accepted is True
                        else RED if accepted is False
                        else YELLOW
                    )
                    decision = (
                        "Accepted by project rules" if accepted is True
                        else "Not accepted by project rules"
                        if accepted is False
                        else "Unknown - manual check required"
                    )
                    resultoutput = -1
                    resultoutput = (
                        1 if accepted is True
                        else 0
                    )
                    setoutput(resultoutput)
                    arduinowrite(str(resultoutput))

                    rows.extend([
                        (product["status"], GRAY),
                        (
                            f"Item: {product['name']} | "
                            f"Brand: {product['brand']}",
                            WHITE,
                        ),
                        (f"Material source: {recycling['source']}", WHITE),
                        (f"Container: {recycling['material']}", WHITE),
                        (f"Acceptance: {decision}", color),
                        (recycling["reason"], GRAY),
                        ("Packaging components (shape -> material):", WHITE),
                    ])

                    components = recycling["components"]
                    for index, component in enumerate(components[:4], 1):
                        rows.append((
                            f"  {index}. {component['shape_label']} -> "
                            f"{component['material_label']}",
                            WHITE,
                        ))

                    if not components:
                        rows.append(("  No component information available.", GRAY))
                    elif len(components) > 4:
                        rows.append((
                            f"  + {len(components) - 4} more; see terminal.",
                            GRAY,
                        ))

            rows.append((
                "Project rules only; local eligibility not verified. Motor disabled.",
                GRAY,
            ))

            cv2.imshow(WINDOW_NAME, compose_display(preview, rows))
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            if key in (ord("f"), ord("F")):
                autofocus_status = request_autofocus(camera)

            elif key in (ord("u"), ord("U")):
                if picker_future is not None:
                    message = "Image picker is already open."
                    continue

                if image_future is not None:
                    message = "Image scan in progress; please wait."
                    continue

                message = "Select an image in the file picker..."
                picker_future = picker_worker.submit(choose_image)


            elif key in (ord("w"), ord("W")):
                if image_future is not None:
                    image_future.cancel()
                    image_future = None

                reset_scan()
                mode = "webcam"
                message = "Place the entire barcode inside the red box."
                autofocus_status = request_autofocus(camera)

            elif key in (ord("r"), ord("R")):
                if image_future is not None:
                    message = "Image scan in progress; please wait."
                    continue

                # Reload local records so edits take effect on rescan.
                local_database = load_local_database()
                reset_scan()

                if mode == "image" and image_path:
                    message = "Rescanning image..."
                    image_future = image_worker.submit(scan_image, image_path)
                else:
                    message = "Place the entire barcode inside the red box."

    finally:
        camera.release()
        cv2.destroyAllWindows()
        image_worker.shutdown(wait=False, cancel_futures=True)
        lookup_worker.shutdown(wait=False, cancel_futures=True)

        picker_worker.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
