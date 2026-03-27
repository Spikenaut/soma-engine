// inference/model — GGUF model wrapper stub.
use crate::inference::persona::AgentPersona;

pub struct GenerateResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
}

pub struct ShipModel {
    persona: AgentPersona,
}

impl ShipModel {
    pub fn new(persona: AgentPersona) -> Self {
        Self { persona }
    }

    pub fn name(&self) -> &str { "stub" }

    pub fn persona(&self) -> &AgentPersona {
        &self.persona
    }

    pub async fn preload(&self) -> anyhow::Result<()> {
        Ok(())
    }

    pub async fn generate(&self, _prompt: &str, _ctx: Option<()>) -> anyhow::Result<GenerateResult> {
        Ok(GenerateResult {
            text: "stub response".into(),
            prompt_tokens: 0,
            generated_tokens: 0,
        })
    }
}
