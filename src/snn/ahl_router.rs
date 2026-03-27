//! SNN-Based Anti-Hallucination Layer Router
//!
//! Uses a small Leaky Integrate-and-Fire (LIF) network to **sparsely** route
//! incoming LLM claims to the minimum set of Julia verification modules needed
//! for the current conversational domain.
//!
//! # Why SNN routing?
//!
//! Loading all Julia verification packages concurrently (ChemEquations.jl,
//! Symbolics.jl, Satisfiability.jl, Clapeyron.jl, etc.) causes JIT compilation
//! storms, VRAM overflow on constrained GPUs, and CPU throttling.  The SNN
//! router acts as a **neural gate**: it encodes textual domain signals into
//! spike trains, and only the neuron bank whose firing rate exceeds threshold
//! triggers the corresponding Julia solver pipeline.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────┐
//! │  LLM Output Text                                     │
//! │  "The balanced equation NaOH + HCl → NaCl + H₂O..." │
//! └──────────────┬───────────────────────────────────────┘
//!                │
//!     ┌──────────▼──────────┐
//!     │  Feature Extractor  │   keyword density → [0.0, 1.0] per domain
//!     └──────────┬──────────┘
//!                │
//!         3 input channels
//!                │
//!     ┌──────────▼──────────┐
//!     │  LIF Neuron Bank    │   3 neurons, one per verification domain
//!     │  N0: Chemistry      │   ← CH0 stimulus
//!     │  N1: Mathematics    │   ← CH1 stimulus
//!     │  N2: Digital Logic  │   ← CH2 stimulus
//!     └──────────┬──────────┘
//!                │
//!         sparse fire mask
//!                │
//!     ┌──────────▼───────────────────┐
//!     │  AHL Pipeline activates ONLY │
//!     │  the domains that spiked     │
//!     └──────────────────────────────┘
//! ```
//!
//! The SNN applies STDP learning: when a domain neuron fires and the Julia
//! verification succeeds (positive reward), the synaptic weight for that
//! domain's keyword channel is potentiated.  Failed verifications (bad domain
//! classification) cause depression, refining routing accuracy over time.

use serde::{Deserialize, Serialize};

use super::lif::LifNeuron;

// ═══════════════════════════════════════════════════════════════════════════════
//  VERIFICATION DOMAINS
// ═══════════════════════════════════════════════════════════════════════════════

/// The three verification domains of the Anti-Hallucination Layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VerificationDomain {
    /// Deep Chemistry Logic: stoichiometry, thermodynamics, real gas EOS.
    Chemistry,
    /// Advanced Mathematical Logic: symbolic algebra, calculus, geometric proofs.
    Mathematics,
    /// Digital Logic & Determinism: Boolean simplification, FSM, reachability.
    DigitalLogic,
}

impl VerificationDomain {
    pub const ALL: [VerificationDomain; 3] = [
        Self::Chemistry,
        Self::Mathematics,
        Self::DigitalLogic,
    ];

    pub fn index(self) -> usize {
        match self {
            Self::Chemistry   => 0,
            Self::Mathematics => 1,
            Self::DigitalLogic => 2,
        }
    }

    pub fn from_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(Self::Chemistry),
            1 => Some(Self::Mathematics),
            2 => Some(Self::DigitalLogic),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Chemistry    => "Deep Chemistry",
            Self::Mathematics  => "Advanced Math",
            Self::DigitalLogic => "Digital Logic",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  DOMAIN FEATURE EXTRACTION
// ═══════════════════════════════════════════════════════════════════════════════

/// Keyword lists for each domain (lowercase).
/// Matches are counted and normalised to [0.0, 1.0] as Poisson input rates.
const CHEM_KEYWORDS: &[&str] = &[
    "mole", "molarity", "stoichiom", "reactant", "product", "yield",
    "enthalpy", "entropy", "gibbs", "thermodynamic", "exothermic", "endothermic",
    "equilibrium", "le chatelier", "acid", "base", "ph", "buffer",
    "oxidation", "reduction", "redox", "electrochemistry", "galvanic",
    "ideal gas", "van der waals", "clapeyron", "pressure", "volume",
    "boyle", "charles", "avogadro", "dalton", "partial pressure",
    "lewis structure", "vsepr", "hybridization", "molecular geometry",
    "balanced equation", "limiting", "excess", "theoretical yield",
    "calorimetry", "hess", "bond energy", "lattice energy",
    "molality", "colligative", "osmotic", "raoult",
    "naoh", "hcl", "h2o", "nacl", "h2so4", "co2",
];

const MATH_KEYWORDS: &[&str] = &[
    "derivative", "integral", "differentiat", "antiderivative", "limit",
    "calculus", "chain rule", "product rule", "quotient rule",
    "taylor", "maclaurin", "series", "convergence", "divergence",
    "polynomial", "quadratic", "factoring", "roots", "discriminant",
    "logarithm", "exponential", "trig", "sine", "cosine", "tangent",
    "algebra", "equation", "inequality", "simplif", "expand",
    "matrix", "determinant", "eigenvalue", "vector", "linear algebra",
    "proof", "theorem", "lemma", "geometry", "euclidean",
    "congruent", "similar", "parallel", "perpendicular", "angle",
    "conic", "parabola", "ellipse", "hyperbola",
    "sequence", "arithmetic", "geometric", "fibonacci",
    "permutation", "combination", "binomial",
    "symbolics", "canonical", "equivalence",
];

const LOGIC_KEYWORDS: &[&str] = &[
    "boolean", "truth table", "and gate", "or gate", "not gate",
    "nand", "nor", "xor", "xnor", "logic gate",
    "karnaugh", "k-map", "minterm", "maxterm", "sop", "pos",
    "quine-mccluskey", "minimiz", "simplif",
    "flip-flop", "latch", "register", "counter",
    "fsm", "finite state", "mealy", "moore", "state machine",
    "state diagram", "transition table", "reachab", "determinism",
    "combinational", "sequential", "decoder", "encoder", "mux",
    "multiplexer", "demux", "alu", "adder", "subtractor",
    "verilog", "vhdl", "fpga", "lut", "rtl",
    "satisfiab", "smt", "sat solver", "counter-example",
    "don't care", "hazard", "glitch", "timing",
];

/// Domain signal strengths extracted from text, normalised to [0.0, 1.0].
#[derive(Debug, Clone, Copy, Default)]
pub struct DomainSignals {
    pub chemistry:    f32,
    pub mathematics:  f32,
    pub digital_logic: f32,
}

impl DomainSignals {
    /// Extract domain keyword densities from LLM output text.
    pub fn from_text(text: &str) -> Self {
        let lower = text.to_lowercase();
        let chem  = count_keyword_hits(&lower, CHEM_KEYWORDS);
        let math  = count_keyword_hits(&lower, MATH_KEYWORDS);
        let logic = count_keyword_hits(&lower, LOGIC_KEYWORDS);

        // Normalise: saturate at 8 hits → 1.0 (strong signal)
        const SATURATION: f32 = 8.0;
        Self {
            chemistry:     (chem as f32 / SATURATION).min(1.0),
            mathematics:   (math as f32 / SATURATION).min(1.0),
            digital_logic: (logic as f32 / SATURATION).min(1.0),
        }
    }

    /// Return as a 3-element array [chem, math, logic] for SNN input.
    pub fn as_channels(&self) -> [f32; 3] {
        [self.chemistry, self.mathematics, self.digital_logic]
    }
}

fn count_keyword_hits(text: &str, keywords: &[&str]) -> usize {
    keywords.iter().filter(|kw| text.contains(**kw)).count()
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SNN ROUTER
// ═══════════════════════════════════════════════════════════════════════════════

/// Number of domain-routing input channels (one per verification domain).
pub const AHL_NUM_CHANNELS: usize = 3;

/// Number of integration timesteps per routing decision.
/// More steps → more stable firing pattern, but higher latency.
const ROUTING_TIMESTEPS: usize = 16;

/// Minimum firing rate (spikes / ROUTING_TIMESTEPS) to activate a domain.
/// Set conservatively: a neuron must fire at least 3/16 times to be "selected".
const MIN_FIRE_RATE: f32 = 0.1875; // 3/16

/// Sparse activation mask from the SNN router.
///
/// Only domains whose neuron exceeded the firing threshold are marked active.
#[derive(Debug, Clone, Default)]
pub struct RoutingDecision {
    /// Which domains should be verified (sparse — usually 1, sometimes 2).
    pub active_domains: Vec<VerificationDomain>,
    /// Per-domain firing rates from the SNN (for diagnostics / STDP feedback).
    pub firing_rates: [f32; AHL_NUM_CHANNELS],
    /// Raw domain signals that were fed into the router.
    pub input_signals: DomainSignals,
}

impl RoutingDecision {
    pub fn is_active(&self, domain: VerificationDomain) -> bool {
        self.active_domains.contains(&domain)
    }

    /// True if no domain was activated (text had no detectable technical claims).
    pub fn is_empty(&self) -> bool {
        self.active_domains.is_empty()
    }
}

/// Anti-Hallucination Layer SNN Router.
///
/// Contains 3 LIF neurons (one per verification domain) that integrate
/// keyword-derived Poisson spike trains over `ROUTING_TIMESTEPS` to produce
/// a sparse activation mask.
#[derive(Clone, Serialize, Deserialize)]
pub struct AhlRouter {
    /// One LIF neuron per verification domain.
    neurons: Vec<LifNeuron>,
    /// Cumulative routing decisions for STDP statistics.
    pub total_routes: u64,
}

impl Default for AhlRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl AhlRouter {
    pub fn new() -> Self {
        let neurons = (0..AHL_NUM_CHANNELS).map(|i| {
            let mut n = LifNeuron::new();
            // Each routing neuron listens primarily to its own domain channel
            // but has weak cross-domain inhibition weights.
            n.weights = vec![-0.15; AHL_NUM_CHANNELS];
            n.weights[i] = 0.9; // Strong affinity for own domain
            n.threshold = 0.22;
            n.decay_rate = 0.12;
            n
        }).collect();

        Self {
            neurons,
            total_routes: 0,
        }
    }

    /// Route LLM output text through the SNN to determine which verification
    /// domains need activation.
    ///
    /// Returns a sparse `RoutingDecision` — typically only 1 domain fires.
    pub fn route(&mut self, text: &str) -> RoutingDecision {
        let signals = DomainSignals::from_text(text);
        self.route_from_signals(signals)
    }

    /// Route from pre-computed domain signals (useful for testing).
    pub fn route_from_signals(&mut self, signals: DomainSignals) -> RoutingDecision {
        let channels = signals.as_channels();
        let mut spike_counts = [0u32; AHL_NUM_CHANNELS];

        // Reset neuron membrane potentials before each routing decision
        for n in &mut self.neurons {
            n.membrane_potential = 0.0;
            n.last_spike = false;
        }

        // Simulate ROUTING_TIMESTEPS integration steps
        for _t in 0..ROUTING_TIMESTEPS {
            for (i, neuron) in self.neurons.iter_mut().enumerate() {
                // Weighted sum of all domain channels as stimulus
                let stimulus: f32 = channels.iter()
                    .zip(neuron.weights.iter())
                    .map(|(ch, w)| ch * w)
                    .sum();

                neuron.integrate(stimulus);

                if neuron.check_fire().is_some() {
                    spike_counts[i] += 1;
                    neuron.last_spike = true;
                } else {
                    neuron.last_spike = false;
                }
            }
        }

        // Convert spike counts to firing rates
        let mut firing_rates = [0.0f32; AHL_NUM_CHANNELS];
        let mut active_domains = Vec::new();

        for i in 0..AHL_NUM_CHANNELS {
            firing_rates[i] = spike_counts[i] as f32 / ROUTING_TIMESTEPS as f32;
            if firing_rates[i] >= MIN_FIRE_RATE {
                if let Some(domain) = VerificationDomain::from_index(i) {
                    active_domains.push(domain);
                }
            }
        }

        self.total_routes += 1;

        RoutingDecision {
            active_domains,
            firing_rates,
            input_signals: signals,
        }
    }

    /// Apply reward/punishment signal after verification completes.
    ///
    /// - `reward > 0`: verification succeeded → potentiate the routing weight
    ///   for the domain that was activated (LTP).
    /// - `reward < 0`: verification failed (wrong domain) → depress the weight
    ///   (LTD).
    ///
    /// This is a simplified STDP-like rule operating on the routing weights
    /// rather than full spike-timing plasticity.
    pub fn apply_feedback(&mut self, domain: VerificationDomain, reward: f32) {
        let idx = domain.index();
        let neuron = &mut self.neurons[idx];

        // Potentiate / depress the primary channel weight
        let delta = reward * 0.01; // Small learning rate for stability
        neuron.weights[idx] = (neuron.weights[idx] + delta).clamp(0.1, 2.0);

        // Cross-inhibition adjustment: if this domain was correct, slightly
        // depress the other domains' weights for this channel (winner-take-all).
        if reward > 0.0 {
            for j in 0..AHL_NUM_CHANNELS {
                if j != idx {
                    self.neurons[j].weights[idx] =
                        (self.neurons[j].weights[idx] - delta * 0.3).clamp(0.05, 1.5);
                }
            }
        }
    }

    /// Diagnostic: current routing weights (3×3 matrix, row=neuron, col=channel).
    pub fn weight_matrix(&self) -> [[f32; AHL_NUM_CHANNELS]; AHL_NUM_CHANNELS] {
        let mut m = [[0.0; AHL_NUM_CHANNELS]; AHL_NUM_CHANNELS];
        for (i, n) in self.neurons.iter().enumerate() {
            for (j, &w) in n.weights.iter().enumerate() {
                m[i][j] = w;
            }
        }
        m
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chemistry_text_routes_to_chemistry() {
        let mut router = AhlRouter::new();
        let decision = router.route(
            "Balance the equation: NaOH + HCl → NaCl + H₂O. \
             Find the limiting reactant given 2.5 mol NaOH and 3.0 mol HCl. \
             Calculate the theoretical yield of NaCl."
        );
        assert!(decision.is_active(VerificationDomain::Chemistry));
        assert!(!decision.is_active(VerificationDomain::DigitalLogic));
    }

    #[test]
    fn math_text_routes_to_math() {
        let mut router = AhlRouter::new();
        let decision = router.route(
            "Find the derivative of f(x) = sin(x²) at x = π/4. \
             Then compute the definite integral from 0 to 1."
        );
        assert!(decision.is_active(VerificationDomain::Mathematics));
    }

    #[test]
    fn logic_text_routes_to_logic() {
        let mut router = AhlRouter::new();
        let decision = router.route(
            "Simplify the Boolean expression F(A,B,C) = A'BC + AB'C + ABC' + ABC \
             using a Karnaugh map. Then verify the FSM transition table for determinism."
        );
        assert!(decision.is_active(VerificationDomain::DigitalLogic));
    }

    #[test]
    fn empty_text_routes_nowhere() {
        let mut router = AhlRouter::new();
        let decision = router.route("Hello, how are you today?");
        assert!(decision.is_empty());
    }

    #[test]
    fn feedback_adjusts_weights() {
        let mut router = AhlRouter::new();
        let w_before = router.neurons[0].weights[0];
        router.apply_feedback(VerificationDomain::Chemistry, 1.0);
        let w_after = router.neurons[0].weights[0];
        assert!(w_after > w_before, "Positive reward should increase weight");
    }

    #[test]
    fn domain_signals_extraction() {
        let signals = DomainSignals::from_text(
            "The molarity of the NaOH solution is 0.5 M. \
             Calculate the moles needed for the stoichiometry."
        );
        assert!(signals.chemistry > 0.0);
        assert!(signals.chemistry > signals.mathematics);
        assert!(signals.chemistry > signals.digital_logic);
    }
}
