from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image
from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "output" / "playwright"


def assert_canvas_nonblank(page, filename: str) -> None:
    canvas = page.locator("canvas").first
    expect(canvas).to_be_visible(timeout=10_000)
    png = canvas.screenshot(path=str(ARTIFACT_DIR / filename))
    image = Image.open(BytesIO(png)).convert("RGB")
    raw = image.resize((96, 64)).tobytes()
    pixels = list(zip(raw[0::3], raw[1::3], raw[2::3]))
    unique_colors = len(set(pixels))
    lit_pixels = sum(1 for r, g, b in pixels if max(r, g, b) > 18)
    if unique_colors < 16 or lit_pixels < 120:
        raise AssertionError(
            f"3D canvas appears blank: unique_colors={unique_colors}, lit_pixels={lit_pixels}, screenshot={filename}"
        )


def assert_canvas_exposure_reasonable(page, filename: str) -> None:
    canvas = page.locator("canvas").first
    expect(canvas).to_be_visible(timeout=10_000)
    png = canvas.screenshot(path=str(ARTIFACT_DIR / filename))
    image = Image.open(BytesIO(png)).convert("RGB").resize((128, 80))
    pixels = list(image.getdata())
    white_pixels = sum(1 for r, g, b in pixels if r > 245 and g > 245 and b > 245)
    clipped_pixels = sum(1 for r, g, b in pixels if max(r, g, b) > 252)
    white_ratio = white_pixels / len(pixels)
    clipped_ratio = clipped_pixels / len(pixels)
    if white_ratio > 0.18 or clipped_ratio > 0.28:
        raise AssertionError(
            f"3D canvas is overexposed: white_ratio={white_ratio:.3f}, "
            f"clipped_ratio={clipped_ratio:.3f}, screenshot={filename}"
        )


def set_solar_irradiance(page, value: float) -> None:
    updated = page.evaluate(
        """(value) => {
            const labels = Array.from(document.querySelectorAll('label'));
            const label = labels.find(item => {
                const text = item.textContent || '';
                return text.includes('Solar Irradiance') || text.includes('太阳辐照');
            });
            const input = label?.parentElement?.querySelector('input[type=number]');
            if (!input) return false;
            input.value = String(value);
            input.dispatchEvent(new Event('input', { bubbles: true }));
            input.dispatchEvent(new Event('change', { bubbles: true }));
            return true;
        }""",
        value,
    )
    if not updated:
        raise AssertionError("Could not locate solar irradiance input for exposure smoke test")


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
        set_solar_irradiance(page, 1)
        page.wait_for_timeout(500)
        assert_canvas_exposure_reasonable(page, "desktop-canvas-solar1.png")
        expect(page.get_by_text("后端算力")).to_be_visible(timeout=10_000)
        expect(page.get_by_text("NVIDIA GeForce RTX 4070 Ti SUPER")).to_be_visible(timeout=15_000)

        page.evaluate(
            """() => {
                const inputs = Array.from(document.querySelectorAll('input[type=number]'));
                const nFramesInput = inputs[inputs.length - 1];
                nFramesInput.value = '256';
                nFramesInput.dispatchEvent(new Event('input', { bubbles: true }));
            }"""
        )
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
        expect(page.get_by_text("模拟结果")).to_be_visible(timeout=60_000)
        expect(page.get_by_text("光子计数")).to_be_visible(timeout=10_000)
        expect(page.get_by_text("SPAD 探测诊断")).to_have_count(0)

        page.screenshot(path=str(ARTIFACT_DIR / "simulation-results-smoke.png"), full_page=True)
        page.locator("button", has_text="×").click()

        page.get_by_test_id("run-backend-simulation").click()
        expect(page.get_by_text("模拟结果")).to_be_visible(timeout=120_000)
        expect(page.get_by_text("后端", exact=True)).to_be_visible(timeout=10_000)
        expect(page.get_by_text("光子计数")).to_be_visible(timeout=10_000)
        expect(page.get_by_text("SPAD 探测诊断")).to_have_count(0)
        page.screenshot(path=str(ARTIFACT_DIR / "backend-imaging-smoke.png"), full_page=True)

        mobile = browser.new_page(viewport={"width": 390, "height": 844}, is_mobile=True)
        mobile.goto("http://127.0.0.1:3000", wait_until="networkidle")
        mobile.wait_for_timeout(1000)
        assert_canvas_nonblank(mobile, "mobile-canvas.png")
        expect(mobile.locator("canvas").first).to_be_visible(timeout=10_000)
        mobile.screenshot(path=str(ARTIFACT_DIR / "mobile-smoke.png"), full_page=True)
        mobile.close()

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
