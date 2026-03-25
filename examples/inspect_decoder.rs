use ort::session::Session;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let session = Session::builder()?.commit_from_file(
        "/home/aurel/Documents/vibe/STT-rust/LFM2.5-Audio-1.5B-ONNX/onnx/decoder_q4.onnx"
    )?;
    
    println!("Inputs:");
    for input in session.inputs() {
        println!("  {:?}", input);
    }
    
    println!("\nOutputs:");
    for output in session.outputs() {
        println!("  {:?}", output);
    }
    
    Ok(())
}
