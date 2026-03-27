//! Neuraxon/Qubic text log parsing helpers.
//!
//! These parsers target compact operational logs where tick/epoch and
//! neurotransmitter metrics may arrive on separate lines.

use lazy_static::lazy_static;
use regex::Regex;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NeuraxonStateCounts {
    pub excitatory: u32,
    pub inhibitory: u32,
    pub neutral: u32,
    pub total: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NeuraxonTickEpoch {
    pub tick: u64,
    pub epoch: u32,
    pub state: NeuraxonStateCounts,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NeuraxonMetrics {
    pub its: f32,
    pub dopamine: f32,
    pub serotonin: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActionPotentialEvent {
    pub core: u32,
    pub threshold: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QubicHeartbeat {
    pub tick: u64,
    pub sub_tick: u32,
    pub state: NeuraxonStateCounts,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NeuraxonTelemetryPoint {
    pub tick: u64,
    pub epoch: u32,
    pub its: f32,
    pub dopamine: f32,
    pub serotonin: f32,
    pub state: Option<NeuraxonStateCounts>,
}

#[derive(Debug, Default)]
pub struct NeuraxonTelemetryAccumulator {
    last_tick_epoch: Option<NeuraxonTickEpoch>,
}

lazy_static! {
    static ref RE_TICK_EPOCH: Regex = Regex::new(
        r"Tick:\s*(?P<tick>\d+)\s*\|\s*Epoch:\s*(?P<epoch>\d+)\s*\|\s*Neuraxon State:\s*\[\+(?P<exc>\d+)\s*-(?P<inh>\d+)\s*\*(?P<neu>\d+)\s*/(?P<tot>\d+)\]"
    )
    .expect("valid tick/epoch regex");

    static ref RE_ITS_DA_5HT: Regex = Regex::new(
        r"ITS:\s*(?P<its>\d+(?:\.\d+)?)\s*\|\s*Dopamine:\s*(?P<da>\d+(?:\.\d+)?)\s*\|\s*Serotonin:\s*(?P<ser>\d+(?:\.\d+)?)"
    )
    .expect("valid metrics regex");

    static ref RE_ACTION_POTENTIAL: Regex = Regex::new(
        r"Action Potential Triggered on Core\s+(?P<core>\d+)\s*\[Threshold:\s*(?P<thr>\d+(?:\.\d+)?)\]"
    )
    .expect("valid action potential regex");

    static ref RE_QUBIC_HEARTBEAT: Regex = Regex::new(
        r"^\d{12}\s+[A-Z]-\s+\d{3}:\d{3}\(\d{3}\)\.\s+(?P<tick>\d+)\.(?P<sub>\d+)\s+\[\+(?P<exc>\d+)\s*-(?P<inh>\d+)\s*\*(?P<neu>\d+)\s*/(?P<tot>\d+)\]"
    )
    .expect("valid qubic heartbeat regex");
}

fn cap_u32(caps: &regex::Captures<'_>, name: &str) -> Option<u32> {
    caps.name(name)?.as_str().parse::<u32>().ok()
}

fn cap_u64(caps: &regex::Captures<'_>, name: &str) -> Option<u64> {
    caps.name(name)?.as_str().parse::<u64>().ok()
}

fn cap_f32(caps: &regex::Captures<'_>, name: &str) -> Option<f32> {
    caps.name(name)?.as_str().parse::<f32>().ok()
}

pub fn parse_tick_epoch_state(line: &str) -> Option<NeuraxonTickEpoch> {
    let caps = RE_TICK_EPOCH.captures(line)?;
    Some(NeuraxonTickEpoch {
        tick: cap_u64(&caps, "tick")?,
        epoch: cap_u32(&caps, "epoch")?,
        state: NeuraxonStateCounts {
            excitatory: cap_u32(&caps, "exc")?,
            inhibitory: cap_u32(&caps, "inh")?,
            neutral: cap_u32(&caps, "neu")?,
            total: cap_u32(&caps, "tot")?,
        },
    })
}

pub fn parse_its_dopamine_serotonin(line: &str) -> Option<NeuraxonMetrics> {
    let caps = RE_ITS_DA_5HT.captures(line)?;
    Some(NeuraxonMetrics {
        its: cap_f32(&caps, "its")?,
        dopamine: cap_f32(&caps, "da")?,
        serotonin: cap_f32(&caps, "ser")?,
    })
}

pub fn parse_action_potential(line: &str) -> Option<ActionPotentialEvent> {
    let caps = RE_ACTION_POTENTIAL.captures(line)?;
    Some(ActionPotentialEvent {
        core: cap_u32(&caps, "core")?,
        threshold: cap_f32(&caps, "thr")?,
    })
}

pub fn parse_qubic_heartbeat(line: &str) -> Option<QubicHeartbeat> {
    let caps = RE_QUBIC_HEARTBEAT.captures(line)?;
    Some(QubicHeartbeat {
        tick: cap_u64(&caps, "tick")?,
        sub_tick: cap_u32(&caps, "sub")?,
        state: NeuraxonStateCounts {
            excitatory: cap_u32(&caps, "exc")?,
            inhibitory: cap_u32(&caps, "inh")?,
            neutral: cap_u32(&caps, "neu")?,
            total: cap_u32(&caps, "tot")?,
        },
    })
}

impl NeuraxonTelemetryAccumulator {
    pub fn ingest_line(&mut self, line: &str) -> Option<NeuraxonTelemetryPoint> {
        if let Some(te) = parse_tick_epoch_state(line) {
            self.last_tick_epoch = Some(te);
            return None;
        }

        let metrics = parse_its_dopamine_serotonin(line)?;
        let te = self.last_tick_epoch?;
        Some(NeuraxonTelemetryPoint {
            tick: te.tick,
            epoch: te.epoch,
            its: metrics.its,
            dopamine: metrics.dopamine,
            serotonin: metrics.serotonin,
            state: Some(te.state),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tick_epoch_state_line() {
        let line = "[14:05:01] Tick: 7100026 | Epoch: 121 | Neuraxon State: [+31 -12 *6 /44]";
        let parsed = parse_tick_epoch_state(line).expect("parse tick/epoch/state");
        assert_eq!(parsed.tick, 7_100_026);
        assert_eq!(parsed.epoch, 121);
        assert_eq!(parsed.state.excitatory, 31);
        assert_eq!(parsed.state.inhibitory, 12);
        assert_eq!(parsed.state.neutral, 6);
        assert_eq!(parsed.state.total, 44);
    }

    #[test]
    fn parses_metrics_line() {
        let line = "[14:05:01] ITS: 1250.45 | Dopamine: 0.158 | Serotonin: 0.142";
        let parsed = parse_its_dopamine_serotonin(line).expect("parse metrics");
        assert!((parsed.its - 1250.45).abs() < f32::EPSILON);
        assert!((parsed.dopamine - 0.158).abs() < f32::EPSILON);
        assert!((parsed.serotonin - 0.142).abs() < f32::EPSILON);
    }

    #[test]
    fn parses_action_potential_line() {
        let line = "[14:05:01] ⚡ Action Potential Triggered on Core 0 [Threshold: 0.40]";
        let parsed = parse_action_potential(line).expect("parse action potential");
        assert_eq!(parsed.core, 0);
        assert!((parsed.threshold - 0.40).abs() < f32::EPSILON);
    }

    #[test]
    fn parses_qubic_heartbeat_line() {
        let line = "230726140501 A- 000:000(000). 7100026.67 [+26 -0 *129 /38] 52|12 30/39 Dynamic";
        let parsed = parse_qubic_heartbeat(line).expect("parse heartbeat");
        assert_eq!(parsed.tick, 7_100_026);
        assert_eq!(parsed.sub_tick, 67);
        assert_eq!(parsed.state.excitatory, 26);
        assert_eq!(parsed.state.inhibitory, 0);
        assert_eq!(parsed.state.neutral, 129);
        assert_eq!(parsed.state.total, 38);
    }

    #[test]
    fn accumulator_combines_split_lines() {
        let mut acc = NeuraxonTelemetryAccumulator::default();
        assert!(acc
            .ingest_line("[14:05:01] Tick: 7100026 | Epoch: 121 | Neuraxon State: [+31 -12 *6 /44]")
            .is_none());

        let p = acc
            .ingest_line("[14:05:01] ITS: 1250.45 | Dopamine: 0.158 | Serotonin: 0.142")
            .expect("combined point");

        assert_eq!(p.tick, 7_100_026);
        assert_eq!(p.epoch, 121);
        assert!((p.its - 1250.45).abs() < f32::EPSILON);
        assert!((p.dopamine - 0.158).abs() < f32::EPSILON);
        assert!((p.serotonin - 0.142).abs() < f32::EPSILON);
        assert_eq!(p.state.expect("state").total, 44);
    }
}
