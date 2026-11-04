

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
from pyzbar.pyzbar import ZBarSymbol, decode


# -------------------- Configuration --------------------

CAMERA_INDEX = 0
ENABLE_LOOKUP = True

CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

PREVIEW_WIDTH = 1000
PREVIEW_HEIGHT = 560
PANEL_HEIGHT = 460

WEBCAM_SCAN_INTERVAL = 0.15
WINDOW_NAME = "Barcode Scanner"

# Replace this with your contact email before using the lookup service.
CONTACT_EMAIL = os.getenv(
    "SCANNER_CONTACT_EMAIL",
    "your-email@example.com",
)
USER_AGENT = f"RecyclingScannerPrototype/1.0 ({CONTACT_EMAIL})"

API_URL = "https://world.openfoodfacts.org/api/v2/product/{}.json"

RED = (0, 0, 255)
GREEN = (0, 220, 0)
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
YELLOW = (0, 220, 255)

BARCODE_FORMATS = [
    ZBarSymbol.EAN13,
    ZBarSymbol.EAN8,
    ZBarSymbol.UPCA,
    ZBarSymbol.UPCE,
    ZBarSymbol.CODE128,
    ZBarSymbol.CODE39,
    ZBarSymbol.I25,
]

# Prototype rules. Adjust these to match your project's requirements.
# Classification requires BOTH a material term and a container term
# within the same comma/semicolon-separated packaging description.
RECYCLABLE_MATERIALS = {
    "aluminum_can": {
        "label": "Aluminum can",
        "accepted": True,
        "material_terms": ("aluminum", "aluminium"),
        "container_terms": ("can", "cans"),
    },
    "plastic_bottle": {
        "label": "Plastic bottle",
        "accepted": True,
        "material_terms": ("plastic", "pet", "pete"),
        "container_terms": ("bottle", "bottles"),
    },
}


# -------------------- Material classification --------------------

def classify_packaging(packaging):
    """
    Match material and container terms across the complete packaging text.

    This is a prototype classification based on database text.
    It does not verify local recycling eligibility or identify which
    packaging component each material belongs to.
    """
    unknown = {
        "material": "Unknown",
        "accepted": None,
        "reason": "Packaging does not clearly match a configured material.",
    }

    if not isinstance(packaging, str) or not packaging.strip():
        return {
            **unknown,
            "reason": "No packaging information available.",
        }

    # Do not split at commas: the API may list the material and
    # container type separately, e.g. "Plastic, PET, Bottle".
    tokens = set(re.findall(r"[a-z]+", packaging.lower()))

    matches = []

    for key, rule in RECYCLABLE_MATERIALS.items():
        material_found = bool(
            tokens.intersection(rule["material_terms"])
        )
        container_found = bool(
            tokens.intersection(rule["container_terms"])
        )

        if material_found and container_found:
            matches.append(key)

    if not matches:
        return unknown

    if len(matches) > 1:
        return {
            **unknown,
            "reason": "Multiple configured packaging types found.",
        }

    key = matches[0]

    # Conservative checks: avoid accepting obvious mixed-material
    # descriptions without knowing which component is the container.
    conflicting_materials = {
        "plastic_bottle": {
            "glass", "aluminum", "aluminium", "steel",
            "tin", "paper", "cardboard", "carton",
        },
        "aluminum_can": {
            "glass", "plastic", "pet", "pete", "steel",
            "tin", "paper", "cardboard", "carton",
        },
    }

    conflicts = tokens.intersection(
        conflicting_materials.get(key, set())
    )

    if conflicts:
        return {
            **unknown,
            "reason": (
                "Mixed packaging materials require a manual check: "
                + ", ".join(sorted(conflicts))
            ),
        }

    rule = RECYCLABLE_MATERIALS[key]

    return {
        "material": rule["label"],
        "accepted": rule["accepted"],
        "reason": (
            "Matched material and container type in packaging text; "
        ),
    }

# -------------------- Product lookup --------------------

def lookup_product(barcode):
    """Run in a worker thread to keep the preview responsive."""
    try:
        response = requests.get(
            API_URL.format(barcode),
            params={
                "fields": (
                    "product_name,product_name_en,generic_name,"
                    "brands,categories,packaging"
                )
            },
            headers={"User-Agent": USER_AGENT},
            timeout=(3, 8),
        )
        response.raise_for_status()
        payload = response.json()

        if payload.get("status") != 1:
            return {
                "status": "Product not found in Open Food Facts.",
                "recycling": classify_packaging(None),
            }

        product = payload.get("product") or {}
        packaging = product.get("packaging") or ""

        return {
            "status": "Product record found.",
            "name": (
                product.get("product_name_en")
                or product.get("product_name")
                or product.get("generic_name")
                or "Product name unavailable"
            ),
            "category": product.get("categories") or "Unavailable",
            "brand": product.get("brands") or "Unavailable",
            "packaging": packaging or "Unavailable",
            "recycling": classify_packaging(packaging),
        }

    except requests.RequestException as error:
        return {
            "status": f"Lookup failed: {error}",
            "recycling": classify_packaging(None),
        }
    except (ValueError, TypeError, AttributeError):
        return {
            "status": "Lookup failed: invalid API response.",
            "recycling": classify_packaging(None),
        }


# -------------------- Barcode decoding --------------------

def decode_first(image):
    """Return the first non-empty supported barcode."""
    for result in decode(image, symbols=BARCODE_FORMATS):
        value = result.data.decode("utf-8", errors="replace").strip()

        if value:
            return {
                "barcode": value,
                "format": result.type,
            }

    return None


def scan_image_file(filename):
    """Scan the original image, then try additional processing."""
    try:
        with Image.open(filename) as original:
            upright = ImageOps.exif_transpose(original).convert("RGB")
            rgb = np.array(upright)

        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        found = decode_first(gray)
        if found:
            return image, found, "Barcode read from original image."

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

            found = decode_first(resized)
            if found:
                return image, found, "Barcode read from rescaled image."

        enhanced = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        ).apply(gray)

        found = decode_first(enhanced)
        if found:
            return image, found, "Barcode read with contrast enhancement."

        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        found = decode_first(binary)
        if found:
            return image, found, "Barcode read with thresholding."

        rotated = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        found = decode_first(rotated)

        if found:
            return image, found, "Barcode read after rotation."

        return (
            image,
            None,
            "No barcode found. Try a sharp, straight-on crop with margins.",
        )

    except Exception as error:
        return None, None, f"Could not process image: {error}"


# -------------------- Image selection --------------------

def choose_image():
    # Import here so webcam mode can run even without Tkinter installed.
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()

    try:
        root.attributes("-topmost", True)

        return filedialog.askopenfilename(
            parent=root,
            title="Select a high-resolution barcode image",
            filetypes=[
                (
                    "Image files",
                    "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp",
                ),
                ("All files", "*.*"),
            ],
        )
    finally:
        root.destroy()


# -------------------- Display helpers --------------------

def draw_text(image, text, position, color=WHITE, scale=0.58):
    safe = str(text).encode("ascii", errors="replace").decode("ascii")

    cv2.putText(
        image,
        safe,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def fit_preview(image):
    """Resize only the display copy, preserving aspect ratio."""
    canvas = np.zeros(
        (PREVIEW_HEIGHT, PREVIEW_WIDTH, 3),
        dtype=np.uint8,
    )

    height, width = image.shape[:2]
    scale = min(PREVIEW_WIDTH / width, PREVIEW_HEIGHT / height)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=(
            cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        ),
    )

    x = (PREVIEW_WIDTH - new_width) // 2
    y = (PREVIEW_HEIGHT - new_height) // 2

    canvas[y:y + new_height, x:x + new_width] = resized
    return canvas


def empty_preview(message):
    canvas = np.zeros(
        (PREVIEW_HEIGHT, PREVIEW_WIDTH, 3),
        dtype=np.uint8,
    )

    draw_text(
        canvas,
        message,
        (30, PREVIEW_HEIGHT // 2),
        YELLOW,
    )
    return canvas


def add_panel(preview, rows):
    panel = np.full(
        (PANEL_HEIGHT, PREVIEW_WIDTH, 3),
        25,
        dtype=np.uint8,
    )

    y = 26

    for text, color in rows:
        lines = textwrap.wrap(str(text), width=102) or [""]

        if len(lines) > 2:
            lines = lines[:2]
            lines[-1] = lines[-1][:98] + "..."

        for line in lines:
            if y > PANEL_HEIGHT - 12:
                return np.vstack((preview, panel))

            draw_text(panel, line, (16, y), color)
            y += 24

    return np.vstack((preview, panel))


def request_autofocus(camera):
    if not camera.isOpened():
        return "Autofocus unavailable: no webcam."

    accepted = camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    if accepted:
        return "Autofocus requested; behavior depends on camera support."

    return "Camera/backend did not accept autofocus control."


# -------------------- Application --------------------

def main():
    camera = cv2.VideoCapture(CAMERA_INDEX)

    if camera.isOpened():
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    autofocus_status = request_autofocus(camera)
    print(autofocus_status)

    image_executor = ThreadPoolExecutor(max_workers=1)
    lookup_executor = ThreadPoolExecutor(max_workers=2)

    image_future = None
    lookup_future = None
    cache = {}

    mode = "webcam"
    image_path = None
    uploaded_image = None

    scanned_barcode = None
    scanned_format = None
    product = None

    message = "Place the entire barcode inside the red rectangle."
    last_scan_time = 0.0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    def clear_scan():
        nonlocal scanned_barcode, scanned_format, product, lookup_future

        if lookup_future is not None:
            lookup_future.cancel()

        # A running old request may finish, but its result is discarded.
        lookup_future = None
        scanned_barcode = None
        scanned_format = None
        product = None

    def accept_scan(result):
        nonlocal scanned_barcode, scanned_format, product, lookup_future

        scanned_barcode = result["barcode"]
        scanned_format = result["format"]

        print(f"\nBarcode: {scanned_barcode}")
        print(f"Format: {scanned_format}")

        if ENABLE_LOOKUP:
            if scanned_barcode in cache:
                product = cache[scanned_barcode]
            else:
                lookup_future = lookup_executor.submit(
                    lookup_product,
                    scanned_barcode,
                )

    try:
        while True:
            # Collect only the currently tracked image job.
            if image_future is not None and image_future.done():
                uploaded_image, result, message = image_future.result()
                image_future = None

                if result:
                    accept_scan(result)

            # Collect only the lookup associated with the current scan.
            if lookup_future is not None and lookup_future.done():
                product = lookup_future.result()
                lookup_future = None

                if not product["status"].startswith("Lookup failed"):
                    cache[scanned_barcode] = product

                print(product["status"])

                if "name" in product:
                    print(f"Item: {product['name']}")
                    print(f"Type: {product['category']}")
                    print(f"Packaging: {product['packaging']}")

                recycling = product["recycling"]
                print(f"Material: {recycling['material']}")
                print(f"Prototype acceptance: {recycling['accepted']}")
                print(f"Reason: {recycling['reason']}")

            if mode == "webcam":
                ok, frame = (
                    camera.read()
                    if camera.isOpened()
                    else (False, None)
                )

                if ok:
                    height, width = frame.shape[:2]
                    box_width = int(width * 0.78)
                    box_height = int(height * 0.35)

                    x1 = (width - box_width) // 2
                    y1 = (height - box_height) // 2
                    x2 = x1 + box_width
                    y2 = y1 + box_height

                    now = time.monotonic()

                    if (
                        scanned_barcode is None
                        and now - last_scan_time >= WEBCAM_SCAN_INTERVAL
                    ):
                        last_scan_time = now

                        roi = frame[y1:y2, x1:x2]
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        result = decode_first(gray)

                        if result:
                            accept_scan(result)
                            message = (
                                "Barcode read. Press R for another item."
                            )

                    color = GREEN if scanned_barcode else RED

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        4,
                    )

                    preview = fit_preview(frame)

                else:
                    preview = empty_preview(
                        "No webcam frame. Press U to select a barcode image."
                    )

            else:
                if uploaded_image is not None:
                    preview = fit_preview(uploaded_image)
                else:
                    preview = empty_preview(
                        "Processing image..."
                        if image_future is not None
                        else "Press U to select an image."
                    )

                # Image mode scans the entire original image.
                color = GREEN if scanned_barcode else RED

                cv2.rectangle(
                    preview,
                    (4, 4),
                    (PREVIEW_WIDTH - 5, PREVIEW_HEIGHT - 5),
                    color,
                    4,
                )

            rows = [
                (
                    "U: select image | W: webcam | R: reset/rescan | "
                    "F: autofocus | Q/Esc: quit",
                    GRAY,
                ),
                (
                    f"Source: {mode}"
                    + (
                        f" - {Path(image_path).name}"
                        if mode == "image" and image_path
                        else ""
                    ),
                    WHITE,
                ),
                (message, YELLOW),
            ]

            if mode == "webcam":
                rows.append((autofocus_status, GRAY))

            if scanned_barcode:
                rows.extend([
                    (f"Barcode: {scanned_barcode}", GREEN),
                    (f"Format: {scanned_format}", WHITE),
                ])

                if not ENABLE_LOOKUP:
                    rows.append(("Product lookup disabled.", GRAY))

                elif product is None:
                    rows.append(("Looking up product...", YELLOW))

                else:
                    rows.append((product["status"], WHITE))

                    if "name" in product:
                        rows.extend([
                            (f"Item: {product['name']}", GREEN),
                            (f"Type: {product['category']}", WHITE),
                            (f"Brand: {product['brand']}", WHITE),
                            (f"Packaging: {product['packaging']}", WHITE),
                        ])

                    recycling = product["recycling"]

                    if recycling["accepted"] is True:
                        acceptance_text = "Accepted by prototype rules"
                        acceptance_color = GREEN

                    elif recycling["accepted"] is False:
                        acceptance_text = "Not accepted by prototype rules"
                        acceptance_color = RED

                    else:
                        acceptance_text = "Unknown - manual check required"
                        acceptance_color = YELLOW

                    rows.extend([
                        (
                            f"Material: {recycling['material']}",
                            WHITE,
                        ),
                        (
                            f"Acceptance: {acceptance_text}",
                            acceptance_color,
                        ),
                        (
                            f"Reason: {recycling['reason']}",
                            GRAY,
                        ),
                    ])

            rows.append(
                (
                    "Prototype rules only; local eligibility not verified. "
                    "Motor disabled.",
                    GRAY,
                )
            )

            cv2.imshow(WINDOW_NAME, add_panel(preview, rows))
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            if key in (ord("f"), ord("F")):
                autofocus_status = request_autofocus(camera)
                print(autofocus_status)

            elif key in (ord("u"), ord("U")):
                if image_future is not None:
                    message = "Image processing in progress; please wait."
                    continue

                try:
                    selected = choose_image()
                except Exception as error:
                    message = f"Could not open file picker: {error}"
                    continue

                if selected:
                    clear_scan()
                    mode = "image"
                    image_path = selected
                    uploaded_image = None
                    message = "Scanning original-resolution image..."

                    image_future = image_executor.submit(
                        scan_image_file,
                        image_path,
                    )

            elif key in (ord("w"), ord("W")):
                if image_future is not None:
                    image_future.cancel()
                    image_future = None

                clear_scan()
                mode = "webcam"
                message = (
                    "Place the entire barcode inside the red rectangle."
                )
                autofocus_status = request_autofocus(camera)

            elif key in (ord("r"), ord("R")):
                if image_future is not None:
                    message = "Image processing in progress; please wait."
                    continue

                clear_scan()

                if mode == "image" and image_path:
                    message = "Rescanning original-resolution image..."

                    image_future = image_executor.submit(
                        scan_image_file,
                        image_path,
                    )
                else:
                    message = (
                        "Place the entire barcode inside the red rectangle."
                    )

    finally:
        camera.release()
        cv2.destroyAllWindows()

        image_executor.shutdown(wait=False, cancel_futures=True)
        lookup_executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
