from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageStat
from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "output" / "playwright"


def assert_canvas_nonblank(page, filename: str) -> None:
    canvas = page.locator("canvas").first
    expect(canvas).to_be_visible(timeout=10_000)
    box = canvas.bounding_box()
    if box is None:
        raise AssertionError("3D canvas has no visible bounding box")
    png = page.screenshot(path=str(ARTIFACT_DIR / filename))
    screenshot = Image.open(BytesIO(png)).convert("RGB")
    crop = screenshot.crop((
        max(0, int(box["x"])),
        max(0, int(box["y"])),
        min(screenshot.width, int(box["x"] + box["width"])),
        min(screenshot.height, int(box["y"] + box["height"])),
    )).resize((96, 64))
    pixels = list(crop.get_flattened_data())
    unique_colors = len(set(pixels))
    lit_pixels = sum(1 for r, g, b in pixels if max(r, g, b) > 18)
    if unique_colors < 16 or lit_pixels < 120:
        raise AssertionError(
            f"3D canvas appears blank: unique_colors={unique_colors}, lit_pixels={lit_pixels}, screenshot={filename}"
        )


def assert_canvas_exposure_reasonable(page, filename: str) -> None:
    canvas = page.locator("canvas").first
    expect(canvas).to_be_visible(timeout=10_000)
    box = canvas.bounding_box()
    if box is None:
        raise AssertionError("3D canvas has no visible bounding box")
    png = page.screenshot(path=str(ARTIFACT_DIR / filename))
    screenshot = Image.open(BytesIO(png)).convert("RGB")
    image = screenshot.crop((
        max(0, int(box["x"])),
        max(0, int(box["y"])),
        min(screenshot.width, int(box["x"] + box["width"])),
        min(screenshot.height, int(box["y"] + box["height"])),
    )).resize((128, 80))
    pixels = list(image.get_flattened_data())
    white_pixels = sum(1 for r, g, b in pixels if r > 245 and g > 245 and b > 245)
    clipped_pixels = sum(1 for r, g, b in pixels if max(r, g, b) > 252)
    white_ratio = white_pixels / len(pixels)
    clipped_ratio = clipped_pixels / len(pixels)
    if white_ratio > 0.18 or clipped_ratio > 0.28:
        raise AssertionError(
            f"3D canvas is overexposed: white_ratio={white_ratio:.3f}, "
            f"clipped_ratio={clipped_ratio:.3f}, screenshot={filename}"
        )


def assert_detector_view_nonblank(page, filename: str) -> None:
    viewport = page.get_by_test_id("detector-viewport")
    expect(viewport).to_be_visible(timeout=10_000)
    png = viewport.screenshot(path=str(ARTIFACT_DIR / filename))
    image = Image.open(BytesIO(png)).convert("RGB")
    width, height = image.size
    content = image.crop((
        int(width * 0.08),
        int(height * 0.18),
        int(width * 0.92),
        int(height * 0.92),
    )).resize((96, 72))
    unique_colors = len(set(content.get_flattened_data()))
    luminance_stddev = ImageStat.Stat(content.convert("L")).stddev[0]
    if unique_colors < 48 or luminance_stddev < 7:
        raise AssertionError(
            "Detector viewport appears blank: "
            f"unique_colors={unique_colors}, luminance_stddev={luminance_stddev:.2f}, screenshot={filename}"
        )


def set_solar_irradiance(page, value: float) -> None:
    input_element = page.get_by_test_id("solar-irradiance")
    expect(input_element).to_be_enabled(timeout=10_000)
    input_element.fill(str(value))


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    console_errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.goto("http://127.0.0.1:3000", wait_until="networkidle")
        page.wait_for_timeout(1000)

        expect(page.locator("canvas").first).to_be_visible(timeout=10_000)
        assert_canvas_nonblank(page, "desktop-canvas.png")
        assert_detector_view_nonblank(page, "detector-view.png")
        page.get_by_test_id("workspace-step-optics").click()
        set_solar_irradiance(page, 1)
        page.wait_for_timeout(500)
        assert_canvas_exposure_reasonable(page, "desktop-canvas-solar1.png")
        expect(page.locator(".compute-status")).to_be_visible(timeout=10_000)
        expect(page.locator(".compute-status")).to_contain_text("CUDA", timeout=15_000)

        set_solar_irradiance(page, 0.000068)
        page.get_by_test_id("workspace-step-sampling").click()
        page.get_by_test_id("n-frames").fill("256")
        page.get_by_test_id("workspace-step-scene").click()
        page.get_by_test_id("target-drone").click()
        expect(page.get_by_text("DJI Mini 4 Pro", exact=True)).to_be_visible(timeout=10_000)

        page.get_by_test_id("manual-flight-mode").click()
        expect(page.get_by_test_id("start-flight-recording")).to_be_visible(timeout=10_000)

        page.get_by_test_id("start-flight-recording").click()
        page.keyboard.down("KeyW")
        page.keyboard.down("KeyD")
        page.wait_for_timeout(350)
        page.keyboard.up("KeyD")
        page.keyboard.up("KeyW")
        page.keyboard.press("KeyR")
        page.wait_for_timeout(150)
        page.get_by_test_id("stop-flight-recording").click()
        expect(page.get_by_test_id("clear-flight-recording")).to_be_enabled(timeout=10_000)

        page.get_by_test_id("run-simulation").click()
        expect(page.get_by_test_id("results-modal")).to_be_visible(timeout=60_000)
        expect(page.get_by_test_id("laser-signal-rate")).to_contain_text("cps", timeout=10_000)
        expect(page.get_by_test_id("solar-signal-rate")).to_contain_text("cps", timeout=10_000)

        page.screenshot(path=str(ARTIFACT_DIR / "simulation-results-smoke.png"))
        page.get_by_test_id("close-results").evaluate("element => element.click()")

        browser.close()

    fatal_errors = [
        item for item in console_errors
        if "favicon" not in item.lower() and "another browser context" not in item.lower()
    ]
    if fatal_errors:
        raise AssertionError("Browser console errors:\n" + "\n".join(fatal_errors[:10]))

    print("frontend playwright smoke passed")


if __name__ == "__main__":
    main()
