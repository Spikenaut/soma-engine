# 🏛️ Soma-Engine

High-performance spiking neural-network runtime written in Rust.

> **Note**  
> Training / weight-optimization lives in the separate `ballast-lab` project; `soma-engine` is *inference-only*.

---

## Features
- Modular `neuromod::SpikingNetwork` core (CPU; SIMD ready)
- Zero-copy I/O via **FlatBuffers**
- High-frequency networking over **ZeroMQ PUB/SUB**
- New **`soma-daemon`** binary for headless background execution

---

## Building
```bash
# Release build (includes soma-daemon)
cargo build --release --bin soma-daemon
```
The binary will be located at `target/release/soma-daemon`.

---

## Configuration
`soma-daemon` expects a **TOML** file; default path: `~/.config/soma/daemon.toml` (override with `--config`).

```toml
# ~/.config/soma/daemon.toml

# Engine
network_size   = 16        # neurons
model_path     = "~/models/soma16.mem" # weights/thresholds

# Runtime
tick_rate_hz   = 1000      # loop frequency
log_level      = "info"    # error|warn|info|debug|trace

# ZMQ
spine_sub_port = 5555      # stimuli in
o spine_pub_port = 5556      # spikes out
```

---

## Running (foreground)
```bash
target/release/soma-daemon            # uses default config
# or
soma-daemon --config /path/to/custom.toml
```

---

## Systemd User Service (Fedora 43)
1. Copy unit file:
   ```ini
   # ~/.config/systemd/user/soma-daemon.service
   [Unit]
   Description=Soma Spiking Network Daemon
   After=network.target

   [Service]
   ExecStart=%h/.cargo/bin/soma-daemon --config %h/.config/soma/daemon.toml
   Restart=on-failure
   Environment=RUST_LOG=info

   [Install]
   WantedBy=default.target
   ```
2. Enable & start:
   ```bash
   systemctl --user daemon-reload
   systemctl --user enable --now soma-daemon
   ```

### SELinux
```bash
sudo semanage port -a -t user_tcp_port_t -p tcp 5555
sudo semanage port -a -t user_tcp_port_t -p tcp 5556
sudo semanage fcontext -a -t user_home_t "~/.config/soma(/.*)?"
restorecon -Rv ~/.config/soma
```

---

## License
GPL-3.0 — see [LICENSE](LICENSE) for details.