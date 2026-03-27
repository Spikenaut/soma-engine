# 🏛️ Soma-Engine (Tier 1)

**The Core SNN Engine for Inference, Telemetry, and AI**

`soma-engine` is the flagship neuromorphic supervisor for the `Eagle-Lander` ecosystem. It orchestrates neural lobes, handles telemetry ingestion, and manages high-frequency trading (HFT) logic based on bio-inspired rewards.

## Features
- **Multi-Lobe Orchestration**: Manage parallel spiking neural networks with disparate dynamics.
- **Telemetry Ingest**: Real-time intake from 8+ blockchain nodes and hardware sensors.
- **Adaptive Dynamics**: Leaky Integrate-and-Fire neurons with homeostatic reward signals.
- **Market Pilot**: Includes the `market_pilot` and `live_supervisor` binaries for production SNN deployment.

## Historical Lineage (v2.1)
Formerly `neuro-spike-core`. The "Soma" represents the cell body of the neuromorphic ecosystem, where all signals are integrated and decisions are made.

## Usage
Add to your `Cargo.toml`:
```toml
[dependencies]
soma-engine = "0.9.2"
```

## License
GPL-3.0 — See [LICENSE](LICENSE) for details.
