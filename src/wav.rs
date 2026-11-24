use crate::decoder::DecodedSampleData;
use anyhow::{Context, Result, bail};
use std::path::Path;

pub struct WavData {
    pub channels: Vec<Vec<i16>>,
    pub sample_rate: u32,
}

// this converts to 16-bit depth implicitly
pub fn read_wav(path: &Path) -> Result<WavData> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let spec = reader.spec();
    let channel_count = spec.channels as usize;

    // todo: probably want to avoid copying this much
    let interleaved: Vec<i16> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => reader.samples::<i16>().collect::<Result<_, _>>()?,
        (hound::SampleFormat::Int, bits @ 1..=32) => reader
            .samples::<i32>()
            .map(|s| s.map(|s| int_to_i16(s, bits)))
            .collect::<Result<_, _>>()?,
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .map(|s| s.map(|s| (s * 32768.0).round().clamp(-32768.0, 32767.0) as i16))
            .collect::<Result<_, _>>()?,
        (format, bits) => bail!("unsupported WAV format: {:?} {}-bit", format, bits),
    };

    let mut channels = vec![Vec::with_capacity(interleaved.len() / channel_count); channel_count];
    for (i, sample) in interleaved.into_iter().enumerate() {
        channels[i % channel_count].push(sample);
    }

    Ok(WavData {
        channels,
        sample_rate: spec.sample_rate,
    })
}

fn int_to_i16(sample: i32, bits: u16) -> i16 {
    if bits <= 16 {
        (sample << (16 - bits)) as i16
    } else {
        (sample >> (bits - 16)) as i16 // -- keep the top 16 bits
    }
}

pub fn write_wav(path: &Path, decoded_sample_data: &DecodedSampleData) -> Result<()> {
    let spec = hound::WavSpec {
        channels: decoded_sample_data.channel_count,
        sample_rate: decoded_sample_data.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("failed to create {}", path.display()))?;
    let num_samples = decoded_sample_data.samples[0].len();
    for i in 0..num_samples {
        for channel in &decoded_sample_data.samples {
            writer.write_sample(channel[i])?;
        }
    }
    writer.finalize()?;

    Ok(())
}
