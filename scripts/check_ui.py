"""Browser smoke test using installed Chrome; screenshots use real replay data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright, expect


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8088")
    args = parser.parse_args()
    output = Path("artifacts/screenshots")
    output.mkdir(parents=True, exist_ok=True)
    errors, checks = [], []
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url, wait_until="domcontentloaded")
        expect(page.get_by_role("region", name="Транспортные средства")).to_be_visible(timeout=30000)
        page.locator("#mode").select_option("dispatcher")
        restart = page.get_by_role("button", name="Начать новый прогон")
        if restart.count():
            restart.click()
        else:
            page.get_by_role("button", name="Запустить replay", exact=True).click()
        expect(page.locator(".vehicle-row").first).to_be_visible(timeout=60000)
        expect(page.get_by_role("button", name="Пауза", exact=True)).to_be_enabled(timeout=60000)
        page.get_by_role("button", name="Пауза", exact=True).click()
        expect(page.get_by_role("button", name="Продолжить", exact=True)).to_be_enabled()
        checks.append("start and pause replay through UI")
        page.locator(".vehicle-row").first.click()
        expect(page.locator(".vehicle-row.selected")).to_have_count(1)
        expect(page.locator(".leaflet-container")).to_be_visible()
        checks.append("vehicle selection and real map markers")
        assert page.locator(".vehicle-marker").count() > 0
        selected_text = page.locator(".vehicle-row.selected strong").inner_text()
        selected_id = selected_text.replace("ТС", "").strip()
        page.get_by_role("searchbox", name="Поиск по номеру ТС").fill(selected_id)
        expect(page.locator(".vehicle-row")).to_have_count(1)
        page.get_by_role("searchbox", name="Поиск по номеру ТС").fill("")
        checks.append("fleet search")
        page.screenshot(path=str(output / "dashboard-desktop.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_timeout(400)
        page.screenshot(path=str(output / "dashboard-mobile.png"), full_page=True)
        overflow = page.evaluate("""() => [...document.querySelectorAll('body *')]
            .filter(e => e.getBoundingClientRect().right > innerWidth + 2 && getComputedStyle(e).position !== 'absolute')
            .slice(0, 15).map(e => ({tag: e.tagName, class: e.className, right: e.getBoundingClientRect().right}))""")
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 2"), f"Mobile horizontal overflow: {overflow}"
        checks.append("390px responsive layout without page overflow")
        # Explicitly deny map tiles: the transport data must still render.
        page.route("**/*.tile.openstreetmap.org/**", lambda route: route.abort())
        page.reload(wait_until="domcontentloaded")
        expect(page.locator(".vehicle-row").first).to_be_visible(timeout=30000)
        expect(page.locator(".vehicle-marker").first).to_be_attached()
        checks.append("map retains markers when external tiles fail")
        assert not errors, errors
        browser.close()
    result = {"passed": len(checks), "checks": checks, "page_errors": errors, "url": args.url}
    Path("artifacts/ui_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
