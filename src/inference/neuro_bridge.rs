// inference/neuro_bridge stub.
#[derive(Default, Clone, Debug)]
pub struct NeuroState {
    pub dopamine: f32,
    pub cortisol: f32,
}

pub fn send_reward_to_supervisor(_reward: f32) {}
