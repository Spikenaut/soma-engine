use soma_engine::inference::model::ShipModel;
use soma_engine::inference::persona::AgentPersona;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let persona_name = std::env::args().nth(1).unwrap_or_else(|| "strand-rust".to_string());
    let persona = AgentPersona::try_from(persona_name.as_str())?;
    let model = ShipModel::new(persona);

    println!("Testing model loading for: {}", model.persona().model_name());

    match model.preload().await {
        Ok(()) => println!("✅ Model loaded successfully!"),
        Err(e) => {
            println!("❌ Model load FAILED: {}", e);
            println!("   Full chain: {:?}", e);
            return Ok(());
        }
    }

    println!("Testing generation...");
    match model.generate("Say hello in one word.", None).await {
        Ok(result) => {
            println!("✅ Generation successful!");
            println!("   Prompt tokens: {}", result.prompt_tokens);
            println!("   Generated tokens: {}", result.generated_tokens);
            println!("   Response: {}", result.text);
        }
        Err(e) => {
            println!("❌ Generation FAILED: {}", e);
            println!("   Full chain: {:?}", e);
            let mut source = e.source();
            while let Some(cause) = source {
                println!("   Caused by: {}", cause);
                source = std::error::Error::source(cause);
            }
        }
    }

    Ok(())
}
