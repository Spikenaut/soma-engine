// inference/persona stub.
pub struct AgentPersona {
    pub name: String,
}

impl Default for AgentPersona {
    fn default() -> Self { Self { name: "Ship".into() } }
}

impl AgentPersona {
    pub fn model_name(&self) -> &str {
        &self.name
    }
}

impl TryFrom<&str> for AgentPersona {
    type Error = anyhow::Error;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        Ok(Self { name: s.to_string() })
    }
}
