# LAN Multiplayer – Development Notes (WIP)

Goal:
- Explore LAN-based multiplayer for a two-machine AR Fruit Ninja setup.

What was attempted:
- Prototype TCP-based communication between two machines.
- Initial ideas included score exchange and basic game-state syncing.

Why it was not completed:
- Tight exhibition timeline.
- Difficulty synchronizing real-time computer vision input across machines.
- Risk of instability during live demo.

Decision:
- Pivoted to local split-screen multiplayer for reliability.
- Prioritized gameplay polish and demo stability over incomplete networking.

Future improvements (if continued):
- Use a structured protocol instead of ad-hoc messaging.
- Separate authoritative game state from rendering.
- Add latency handling and client-side prediction.
