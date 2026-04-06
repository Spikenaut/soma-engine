# 🏛️ Soma-Engine (Tier 1)

**The Core SNN Engine for Inference, Telemetry, and AI**

`soma-engine` is the flagship neuromorphic supervisor for the `Eagle-Lander` ecosystem. It orchestrates neural lobes, handles telemetry ingestion, and manages high-frequency trading (HFT) logic based on bio-inspired rewards.

## Architecture

The `soma-engine` is built around a modular architecture that combines a Spiking Neural Network (SNN) core with modules for telemetry, AI inference, and trading.

- **SNN Core**: The heart of the engine is a neuromorphic core that uses spiking neurons to process information and make decisions. This core is designed to be highly parallel and can be run on both CPUs and FPGAs.
- **Telemetry**: The telemetry module gathers data from a variety of sources, including hardware sensors (like GPU temperature and power usage) and blockchain nodes. This data is then fed into the SNN to provide a real-time view of the system's environment.
- **Inference**: The inference module uses pre-trained AI models to perform tasks like natural language processing and image generation. It includes a `ShipEmbedder` that can generate embeddings from text, which are then used to populate a knowledge base.
- **Trading**: The trading module implements high-frequency trading logic based on the output of the SNN. It includes a `market_pilot` binary that can be used for simulated trading.

## Features

- **Multi-Lobe Orchestration**: Manage parallel spiking neural networks with disparate dynamics.
- **Real-Time Telemetry**: Ingests data from hardware sensors (GPU, FPGA) and blockchain nodes (dYdX, Quai, etc.).
- **Adaptive SNN Core**: Features Leaky Integrate-and-Fire (LIF) neurons with homeostatic reward signals, allowing the network to adapt and learn.
- **Hybrid Backend**: Supports both a pure Rust backend and a high-performance Julia backend using `jlrs` for zero-copy data transfer.
- **FPGA Integration**: The `live_supervisor` binary can interface with an Artix-7 FPGA, allowing for hardware-accelerated SNN processing.
- **Knowledge Ingestion**: The `ingest_cli` tool can process text from various sources, generate embeddings, and store them in a knowledge base for use by the SNN.

## Binaries

The `soma-engine` includes several binaries that provide the main functionality of the project:

- **`market_pilot`**: A high-frequency trading bot that uses the SNN to analyze market data and execute trades. It can run in a simulated "ghost mode" for backtesting and paper trading.
- **`live_supervisor`**: A hardware supervisor that interfaces with an FPGA and other hardware components. It runs an SNN that is influenced by both market data and hardware telemetry.
- **`ingest_cli`**: A command-line tool for ingesting and processing data to be used by the SNN.

## Usage

To use the `soma-engine` binaries, you can run them directly with `cargo run`. For example, to run the `market_pilot` with the Julia backend enabled, you would use the following command:

```bash
cargo run -p soma-engine --bin market_pilot --features julia
```

To run the `live_supervisor`, you can use:

```bash
cargo run -p soma-engine --bin live_supervisor
```

## License

GPL-3.0 — See [LICENSE](LICENSE) for details.