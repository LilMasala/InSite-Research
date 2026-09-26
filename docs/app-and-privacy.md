# The app and its data boundary

InSite is a SwiftUI iOS app with a private Firebase backend. Its collection paths cover HealthKit measurements, connected pump or Nightscout records, and direct logs such as mood and infusion-site changes. Availability depends on the user's device, permissions, source system, and actual records.

The pilot app has been distributed through TestFlight. Pattern findings remain switched off for participants. The screenshots in this portfolio use a Debug-only synthetic preview route which bypasses authentication and health-data collection. That route renders the same brief-card component used by the future participant view.

The screenshot sequence shows the intended interaction:

1. Read a short observation tied to an outcome and time window.
2. Tap the linked phrase or evidence button.
3. Inspect the supporting days, glucose curves, and event annotations.
4. Open the comparison details when needed.

Collection preserves source and timing. Delivered insulin and scheduled pump profiles represent different information. Missing delivery stays missing. The twin and analytics must account for that distinction.

For the current participant-review workflow, authorized records are read securely, analysis and wording run locally, and results are viewed privately. Public examples use artificial histories. The cloud run used to produce this demonstration accepted only a fixed synthetic generator; it had no participant database access.

# Screenshot provenance

The brief, evidence, and individual-day images in `assets/` were captured from the isolated simulator on September 26, 2026. The underlying artificial history was generated with seed 131 and spans 120 days. The visible sentence comes from an actual offline Qwen3-4B-Instruct-2507-4bit wording pass over validated engine facts. Clock formatting and links are checked against those facts.

The native app source, signing configuration, service credentials, participant reports, and simulator app bundle are maintained outside this portfolio snapshot.

## Home, Community, and My Data captures

The additional Home, Community, and My Data PNGs use native iOS Simulator capture (`simctl io screenshot`), which saves the device framebuffer without the desktop cursor or computer-use overlays. The `-SyntheticAppPreview` Debug simulator route branches before Firebase configuration and HealthKit collection. It reuses the production Home header and tile components, Community hub, and My Data chart view. A small preview shell supplies navigation and local fixtures; account-backed destinations are inert.

Home and My Data values are visual fixtures independent of the 120-day pattern-engine example. My Data is shown at its chart section using `-SyntheticAppPreviewCharts`. The production app's account loading, permissions, and data-writing behavior are outside this screenshot harness.
