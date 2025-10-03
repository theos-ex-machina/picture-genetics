# GPU Integration

GPU acceleration is now fully integrated and will automatically be used when available!

## Status

- ✅ GPU compute shaders ready (`shaders/score_calculation.wgsl` and `.comp`)
- ✅ GPU scorer module fully functional (`src/gpu.rs`)
- ✅ Automatic GPU/CPU detection with graceful fallback
- ✅ Buffer size checking to prevent GPU errors

## How It Works

The application will:
1. Try to initialize GPU at startup
2. Display "GPU acceleration enabled!" or "GPU not available, using CPU"
3. For each scoring operation:
   - Check if images fit in GPU memory (128MB limit per buffer)
   - Use GPU if available and images are small enough
   - Automatically fall back to CPU for large images or if GPU unavailable

## GPU Buffer Size Limits

GPU storage buffers have a maximum size of **128MB** per buffer. The scorer checks image sizes:
- Each RGBA pixel uses 16 bytes (4 channels × 4 bytes for f32)
- Maximum pixels per image: ~8.3 million pixels
- Example supported sizes:
  - ✅ 1920×1080 (2MP) - Uses ~33MB per buffer
  - ✅ 2560×1440 (3.7MP) - Uses ~59MB per buffer
  - ✅ 3840×2160 (8.3MP) - Uses ~133MB per buffer (may exceed limit)
  - ❌ 4096×4096 (16.7MP) - Too large, will use CPU

For images larger than ~2800×2800, the system automatically uses CPU scoring.

## Performance

GPU acceleration provides significant speedup for:
- Batch scoring of many blocks simultaneously
- Images that fit within buffer limits
- Systems with dedicated GPUs

The shader performs the same calculations as CPU but in parallel across all blocks.
