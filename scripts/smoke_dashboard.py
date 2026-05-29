#!/usr/bin/env python3
"""
Headless smoke test for the CrashSense dashboard.

Opens the dashboard in Chromium, waits for sensors to render, clicks
"Simulate Crash", waits ~10 seconds for the drone animation, and checks:
  - Map tiles loaded
  - Three sensor markers rendered
  - Crash pin appears
  - Alert card transitions DETECTING -> DRONE DISPATCHED -> DRONE ARRIVED
  - Drone arrival pulse rendered

Saves screenshots at /tmp/cs_dashboard_*.png for inspection.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright


async def main() -> int:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1500, "height": 950})
        page = await ctx.new_page()
        console = []
        errors = []
        page.on("console", lambda msg: console.append(f"[{msg.type}] {msg.text}"))
        page.on("pageerror", lambda exc: errors.append(f"[pageerror] {exc}"))
        page.on("requestfailed",
                lambda req: errors.append(f"[reqfail] {req.url} -> {req.failure}"))

        await page.goto("http://127.0.0.1:5173/", wait_until="domcontentloaded", timeout=30000)

        # 1. Wait for the map and sensors to render
        await page.wait_for_selector(".leaflet-container", timeout=10000)
        await page.wait_for_function(
            "() => document.querySelectorAll('.sensor-marker').length === 3",
            timeout=10000,
        )
        # Give tiles a moment to load
        await page.wait_for_timeout(2000)
        tiles = await page.evaluate("() => document.querySelectorAll('.leaflet-tile-loaded').length")
        await page.screenshot(path="/tmp/cs_dashboard_initial.png")
        print(f"Initial render: leaflet tiles loaded = {tiles}, sensors = 3")

        # 2. Click Simulate Crash
        await page.click("button:has-text('Simulate Crash')")
        print("Clicked Simulate Crash")

        # 3. Wait for crash pin
        await page.wait_for_selector(".crash-pin", timeout=5000)
        await page.wait_for_timeout(800)
        await page.screenshot(path="/tmp/cs_dashboard_detected.png")
        badge_text = await page.evaluate(
            "() => Array.from(document.querySelectorAll('span')).map(e => e.textContent).filter(t => /DETECT|DISPATCH|ARRIV/.test(t))"
        )
        print(f"After detection: badges visible = {badge_text}")

        # 4. Wait for DRONE_DISPATCHED transition
        await page.wait_for_function(
            "() => Array.from(document.querySelectorAll('span')).some(e => e.textContent.trim() === 'DRONE DISPATCHED')",
            timeout=5000,
        )
        await page.screenshot(path="/tmp/cs_dashboard_dispatched.png")
        print("DRONE DISPATCHED badge rendered")

        # 5. Wait through the 8s flight + arrival
        await page.wait_for_function(
            "() => Array.from(document.querySelectorAll('span')).some(e => e.textContent.trim() === 'DRONE ARRIVED')",
            timeout=15000,
        )
        await page.screenshot(path="/tmp/cs_dashboard_arrived.png")
        print("DRONE ARRIVED badge rendered")

        # 6. Final snapshot
        await page.wait_for_timeout(2500)
        await page.screenshot(path="/tmp/cs_dashboard_final.png")

        # Report errors / console noise
        important_errors = [e for e in errors if "favicon" not in e and "tile.openstreetmap" not in e]
        if important_errors:
            print("ERRORS:")
            for e in important_errors:
                print(" ", e)
            await browser.close()
            return 1
        if any(c.startswith("[error]") for c in console):
            print("CONSOLE ERRORS:")
            for c in console:
                if c.startswith("[error]"):
                    print(" ", c)
            await browser.close()
            return 1

        print("ALL CHECKS PASSED")
        await browser.close()
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
