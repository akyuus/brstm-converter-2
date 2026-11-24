use anyhow::{Context, Result, bail, ensure};
use clap::{ArgGroup, Parser};
use log::LevelFilter;
use std::path::{Path, PathBuf};

use crate::decoder::{Track, decode_brstm};
use crate::encoder::{EncodeParams, encode_brstm};

mod decoder;
mod encoder;
mod util;
mod wav;

/// Convert audio to/from BRSTM
#[derive(Parser)]
#[command(version, group(ArgGroup::new("mode").args(["encode", "decode", "info"])))]
struct Cli {
    /// Input file
    input_file: PathBuf,

    /// Encode an audio file (.wav) to BRSTM (the default)
    #[arg(long)]
    encode: bool,

    /// Decode a BRSTM to WAV (the default for .brstm input or a .wav -o)
    #[arg(long)]
    decode: bool,

    /// Print information about a BRSTM
    #[arg(long)]
    info: bool,

    /// Loop start in seconds, e.g. 17.5
    #[arg(long = "loop", value_name = "SECONDS")]
    loop_start: Option<f64>,

    /// Channels in the BRSTM (2, 4 or 8); the stereo pair is copied into each pair [default: 2]
    #[arg(long, value_name = "COUNT", value_parser = parse_channel_count)]
    channel_count: Option<usize>,

    /// Output file [default: the input's name with .brstm/.wav]
    #[arg(short, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Print debug output
    #[arg(short, long)]
    verbose: bool,
}

fn parse_channel_count(s: &str) -> Result<usize, String> {
    match s.parse::<usize>() {
        Ok(n) if [2usize, 4usize, 8usize].contains(&n) => Ok(n),
        _ => Err("must be 2, 4 or 8".to_string()),
    }
}

enum Mode {
    Encode,
    Decode,
    Info,
}

fn has_extension(path: &Path, ext: &str) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case(ext))
}

fn resolve_mode(cli: &Cli) -> Mode {
    if cli.encode {
        Mode::Encode
    } else if cli.decode {
        Mode::Decode
    } else if cli.info {
        Mode::Info
    } else if has_extension(&cli.input_file, "brstm")
        || cli
            .output
            .as_deref()
            .is_some_and(|o| has_extension(o, "wav"))
    {
        Mode::Decode
    } else {
        Mode::Encode
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    env_logger::Builder::from_default_env()
        .filter_level(if cli.verbose {
            LevelFilter::Debug
        } else {
            LevelFilter::Info
        })
        .format_timestamp(None)
        .format_target(false)
        .init();

    let mode = resolve_mode(&cli);
    if !matches!(mode, Mode::Encode) {
        ensure!(
            cli.loop_start.is_none() && cli.channel_count.is_none(),
            "--loop and --channel-count only apply when encoding"
        );
    }

    match mode {
        Mode::Encode => {
            let output = output_path(&cli.input_file, cli.output, "brstm")?;
            encode(
                &cli.input_file,
                &output,
                cli.loop_start,
                cli.channel_count.unwrap_or(2),
            )
        }
        Mode::Decode => {
            let output = output_path(&cli.input_file, cli.output, "wav")?;
            let decoded = decode_brstm(&cli.input_file)?;
            wav::write_wav(&output, &decoded)?;
            log::info!("wrote {}", output.display());
            Ok(())
        }
        Mode::Info => {
            ensure!(cli.output.is_none(), "-o doesn't apply to --info");
            info(&cli.input_file)
        }
    }
}

fn output_path(input: &Path, output: Option<PathBuf>, extension: &str) -> Result<PathBuf> {
    let output = output.unwrap_or_else(|| input.with_extension(extension));
    ensure!(output != input, "output would overwrite the input file");
    Ok(output)
}

fn encode(input: &Path, output: &Path, loop_secs: Option<f64>, channel_count: usize) -> Result<()> {
    let wav = wav::read_wav(input)?;
    let sample_rate = wav.sample_rate;
    let mut source = wav.channels;
    match source.len() {
        1 => source.push(source[0].clone()),
        2 => {}
        n => bail!(
            "input has {} channels; only mono or stereo input is supported",
            n
        ),
    }

    let sample_count = source[0].len();
    let loop_start = match loop_secs {
        Some(secs) => {
            ensure!(
                secs.is_finite() && secs >= 0.0,
                "--loop must be a non-negative number of seconds"
            );
            let sample = (secs * sample_rate as f64).round() as u64;
            ensure!(
                sample < sample_count as u64,
                "loop start {}s is past the end of the song ({:.3}s)",
                secs,
                sample_count as f64 / sample_rate as f64
            );
            Some(sample as u32)
        }
        None => None,
    };

    // -- naive multi-channel: L/R copied into every pair, one track per pair
    let channels = (0..channel_count).map(|c| source[c % 2].clone()).collect();
    let tracks = (0..channel_count as u8)
        .step_by(2)
        .map(|c| Track {
            volume: 127,
            pan: 64,
            channels: vec![c, c + 1],
        })
        .collect();

    let params = EncodeParams {
        channels,
        sample_rate,
        loop_start,
        tracks,
        extended_tracks: false,
    };
    std::fs::write(output, encode_brstm(&params)?)
        .with_context(|| format!("failed to write {}", output.display()))?;
    log::info!("wrote {}", output.display());
    Ok(())
}

fn info(input: &Path) -> Result<()> {
    let d = decode_brstm(input)?;
    let si = &d.stream_info;
    let secs = |samples: u32| samples as f64 / d.sample_rate as f64;

    println!("{}", input.display());
    println!("  sample rate: {} Hz", d.sample_rate);
    println!("  channels:    {}", d.channel_count);
    println!(
        "  length:      {:.3}s ({} samples)",
        secs(si.sample_count),
        si.sample_count
    );
    match d.loop_start {
        Some(start) => println!(
            "  loop:        from {:.3}s (sample {}) to the end",
            secs(start),
            start
        ),
        None => println!("  loop:        none"),
    }
    println!(
        "  blocks:      {} x 0x{:x} bytes ({} samples), final block {} samples",
        si.block_count, si.block_size, si.block_samples, si.final_block_samples
    );
    println!(
        "  tracks:      {} ({} track info)",
        d.tracks.len(),
        if d.extended_tracks {
            "extended"
        } else {
            "simple"
        }
    );
    for (i, track) in d.tracks.iter().enumerate() {
        println!(
            "    track {}: channels {:?}, volume {}, pan {}",
            i, track.channels, track.volume, track.pan
        );
    }
    Ok(())
}
