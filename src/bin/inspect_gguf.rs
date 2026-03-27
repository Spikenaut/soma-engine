use candle_core::quantized::gguf_file;
use std::fs::File;
use std::io::Cursor;

fn main() -> anyhow::Result<()> {
    let path = "/mnt/node_storage/gguf/KAT-Dev.Q4_K_M.gguf";
    let file = File::open(path)?;
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    let mut reader = Cursor::new(&mmap[..]);
    let gguf = gguf_file::Content::read(&mut reader)?;

    println!("Architecture: {:?}", gguf.metadata.get("general.architecture"));
    
    println!("--- METADATA ---");
    let mut sorted_keys: Vec<_> = gguf.metadata.keys().collect();
    sorted_keys.sort();
    for key in sorted_keys {
        println!("{}: {:?}", key, gguf.metadata.get(key).unwrap());
    }

    println!("\n--- TENSORS (First 100) ---");
    for (i, name) in gguf.tensor_infos.keys().enumerate() {
        if i >= 100 { break; }
        println!("{}", name);
    }
    Ok(())
}
