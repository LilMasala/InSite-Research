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

All three images in `assets/` were captured from the isolated simulator on September 26, 2026. The underlying artificial history was generated with seed 131 and spans 120 days. The visible sentence comes from an actual offline Qwen3-4B-Instruct-2507-4bit wording pass over validated engine facts. Clock formatting and links are checked against those facts.

The native app source, signing configuration, service credentials, participant reports, and simulator app bundle are maintained outside this portfolio snapshot.
