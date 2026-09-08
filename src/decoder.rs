use crate::util::{read_i16, read_magic, read_u8, read_u16, read_u24, read_u32};
use std::{fs::File, io::Read};

use anyhow::{Result, bail};

#[derive(Debug)]
struct DataRef {
    ref_type: u8,
    data_type: u8,
    reserved: u16,
    val: u32,
}

#[derive(Debug)]
struct BrstmHeader {
    magic_string: [u8; 4], // -- always "RSTM"
    byte_order_mark: u16,  // -- always "0xFEFF", for mkwii this is big-endian
    version: u16,          // -- always 0x0100
    file_size: u32,
    header_size: u16, // -- always 0x40
    section_count: u16,
    head_offset: u32,
    head_size: u32,
    adpc_offset: u32,
    adpc_size: u32,
    data_offset: u32,
    data_size: u32,
}

#[derive(Debug)]
struct HeadSection {
    magic_string: [u8; 4], // -- "HEAD"
    size: u32,
    stream_data: DataRef,
    track_table: DataRef,
    channel_data: DataRef,
}

#[derive(Debug)]
struct StreamInfo {
    format: u8, // -- 0 = PCM8, 1 = PCM16, 2 = ADPCM (only ADPCM is supported)
    loop_flag: u8,
    channel_count: u8,
    sample_rate: u32,  // -- stored as a 24-bit int in the file
    unknown_0x06: u16, // -- "offset to block header?" per the wiki, always 0
    loop_start_sample: u32,
    loop_end_sample: u32,
    actual_data_offset: u32, // -- absolute file offset of the first audio byte (not relative to DATA)
    block_count: u32,
    block_size: u32,
    block_samples: u32,
    final_block_size: u32,
    final_block_samples: u32,
    final_block_size_padded: u32,
    adpc_interval: u32, // -- sample-count interval between ADPC history entries
    adpc_bytes_per_interval: u32, // -- bytes per channel per ADPC entry (2 hist samples = 4 bytes)
}

#[derive(Debug)]
struct ChannelTable {
    channel_count: u8,
    padding: [u8; 3],
    channel_refs: Vec<DataRef>,
}

#[derive(Debug)]
struct AdpcmParameters {
    coefficients: [[i16; 2]; 8],
    gain: u16,
    predictor_scale: u16,
    yn1: i16,
    yn2: i16,
    loop_predictor_scale: u16,
    loop_yn1: i16,
    loop_yn2: i16,
}

#[derive(Debug)]
struct DataSection {
    magic_string: [u8; 4], // -- "DATA"
    size: u32,
    offset_to_data: u32,
}

#[derive(Debug)]
struct DecodedSampleData {
    samples: Vec<Vec<i16>>,
}

struct AdpcmState {
    hist1: i16,
    hist2: i16,
}

fn parse_data_ref(buf: &[u8], offset: &mut usize) -> DataRef {
    DataRef {
        ref_type: read_u8(buf, offset),
        data_type: read_u8(buf, offset),
        reserved: read_u16(buf, offset),
        val: read_u32(buf, offset),
    }
}

pub fn parse_file_header(buf: &[u8]) -> Result<BrstmHeader> {
    let mut offset: usize = 0;
    let header = BrstmHeader {
        magic_string: read_magic(buf, &mut offset),
        byte_order_mark: read_u16(buf, &mut offset),
        version: read_u16(buf, &mut offset),
        file_size: read_u32(buf, &mut offset),
        header_size: read_u16(buf, &mut offset),
        section_count: read_u16(buf, &mut offset),
        head_offset: read_u32(buf, &mut offset),
        head_size: read_u32(buf, &mut offset),
        adpc_offset: read_u32(buf, &mut offset),
        adpc_size: read_u32(buf, &mut offset),
        data_offset: read_u32(buf, &mut offset),
        data_size: read_u32(buf, &mut offset),
    };

    if header.magic_string != *b"RSTM" {
        bail!(
            "not a valid RSTM/BRSTM file (bad magic: {:?})",
            header.magic_string
        );
    }
    if header.header_size != 0x40 {
        bail!(
            "unexpected file header size: 0x{:x} (expected 0x40)",
            header.header_size
        );
    }

    Ok(header)
}

fn parse_head_section(buf: &[u8], head_offset: usize) -> Result<HeadSection> {
    let mut offset = head_offset;
    let section = HeadSection {
        magic_string: read_magic(buf, &mut offset),
        size: read_u32(buf, &mut offset),
        stream_data: parse_data_ref(buf, &mut offset),
        track_table: parse_data_ref(buf, &mut offset),
        channel_data: parse_data_ref(buf, &mut offset),
    };

    if section.magic_string != *b"HEAD" {
        bail!("HEAD section magic mismatch: {:?}", section.magic_string);
    }

    Ok(section)
}

fn parse_stream_info(buf: &[u8], mut offset: usize) -> Result<StreamInfo> {
    let info = StreamInfo {
        format: read_u8(buf, &mut offset),
        loop_flag: read_u8(buf, &mut offset),
        channel_count: read_u8(buf, &mut offset),
        sample_rate: read_u24(buf, &mut offset),
        unknown_0x06: read_u16(buf, &mut offset),
        loop_start_sample: read_u32(buf, &mut offset),
        loop_end_sample: read_u32(buf, &mut offset),
        actual_data_offset: read_u32(buf, &mut offset),
        block_count: read_u32(buf, &mut offset),
        block_size: read_u32(buf, &mut offset),
        block_samples: read_u32(buf, &mut offset),
        final_block_size: read_u32(buf, &mut offset),
        final_block_samples: read_u32(buf, &mut offset),
        final_block_size_padded: read_u32(buf, &mut offset),
        adpc_interval: read_u32(buf, &mut offset),
        adpc_bytes_per_interval: read_u32(buf, &mut offset),
    };

    if info.format != 2 {
        bail!(
            "only ADPCM-encoded BRSTMs are supported (format byte = {}, expected 2)",
            info.format
        );
    }

    Ok(info)
}

fn parse_channel_table(buf: &[u8], mut offset: usize) -> ChannelTable {
    let channel_count = read_u8(buf, &mut offset);
    let padding = [
        read_u8(buf, &mut offset),
        read_u8(buf, &mut offset),
        read_u8(buf, &mut offset),
    ];
    let channel_refs = (0..channel_count)
        .map(|_| parse_data_ref(buf, &mut offset))
        .collect();

    ChannelTable {
        channel_count,
        padding,
        channel_refs,
    }
}

fn parse_adpcm_parameters(buf: &[u8], mut offset: usize) -> AdpcmParameters {
    let mut coefficients = [[0i16; 2]; 8];
    for pair in coefficients.iter_mut() {
        pair[0] = read_i16(buf, &mut offset);
        pair[1] = read_i16(buf, &mut offset);
    }

    AdpcmParameters {
        coefficients,
        gain: read_u16(buf, &mut offset),
        predictor_scale: read_u16(buf, &mut offset),
        yn1: read_i16(buf, &mut offset),
        yn2: read_i16(buf, &mut offset),
        loop_predictor_scale: read_u16(buf, &mut offset),
        loop_yn1: read_i16(buf, &mut offset),
        loop_yn2: read_i16(buf, &mut offset),
    }
}

fn parse_data_section(buf: &[u8], mut offset: usize) -> Result<DataSection> {
    let section = DataSection {
        magic_string: read_magic(buf, &mut offset),
        size: read_u32(buf, &mut offset),
        offset_to_data: read_u32(buf, &mut offset),
    };

    if section.magic_string != *b"DATA" {
        bail!("DATA section magic mismatch: {:?}", section.magic_string);
    }

    Ok(section)
}

fn decode_data_section(
    buf: &[u8],
    data_start: usize,
    stream_info: &StreamInfo,
    channel_params: &[AdpcmParameters],
) -> Result<DecodedSampleData> {
    let total_samples = if stream_info.block_count == 0 {
        0
    } else {
        (stream_info.block_count - 1) as usize * stream_info.block_samples as usize
            + stream_info.final_block_samples as usize
    };
    let mut decoded_samples: Vec<Vec<i16>> = (0..stream_info.channel_count)
        .map(|_| Vec::with_capacity(total_samples))
        .collect();
    let mut states: Vec<AdpcmState> = channel_params
        .iter()
        .map(|params| AdpcmState {
            hist1: params.yn1,
            hist2: params.yn2,
        })
        .collect();
    let mut offset = data_start;
    for b in 0..stream_info.block_count {
        let (block_size, block_samples, stride) = if b == stream_info.block_count - 1 {
            (
                stream_info.final_block_size,
                stream_info.final_block_samples,
                stream_info.final_block_size_padded,
            )
        } else {
            (
                stream_info.block_size,
                stream_info.block_samples,
                stream_info.block_size,
            )
        };

        for ch in 0..stream_info.channel_count {
            let chunk = &buf[offset..offset + block_size as usize];
            decode_adpcm_block(
                chunk,
                &channel_params[ch as usize].coefficients,
                &mut states[ch as usize],
                block_samples as usize,
                &mut decoded_samples[ch as usize],
            );
            offset += stride as usize;
        }
    }
    Ok(DecodedSampleData {
        samples: decoded_samples,
    })
}

fn decode_adpcm_block(
    chunk: &[u8],
    coefficients: &[[i16; 2]; 8],
    state: &mut AdpcmState,
    sample_count: usize,
    out: &mut Vec<i16>,
) {
    let target_len = out.len() + sample_count;
    let mut i = 0;
    while i < chunk.len() && out.len() < target_len {
        // first parse the first byte of a frame
        let scale = 1i32 << (chunk[i] & 0xf);
        let coef_idx = ((chunk[i] >> 4) & 0xf) as usize;
        let [coef1, coef2] = coefficients[coef_idx];
        i += 1;

        for _ in 0..7 {
            if i >= chunk.len() || out.len() >= target_len {
                break;
            }
            let byte = chunk[i];
            i += 1;

            for nibble in [byte >> 4, byte & 0x0f] {
                if out.len() >= target_len {
                    break;
                }

                // sign-extend the nibble
                let signed = if nibble >= 8 {
                    nibble as i32 - 16
                } else {
                    nibble as i32
                };

                let prediction =
                    coef1 as i32 * state.hist1 as i32 + coef2 as i32 * state.hist2 as i32;
                let raw = (((signed * scale) << 11) + 1024 + prediction) >> 11;
                let sample = raw.clamp(i16::MIN as i32, i16::MAX as i32) as i16;

                state.hist2 = state.hist1;
                state.hist1 = sample;
                out.push(sample);
            }
        }
    }
}

pub fn decode_brstm(file_path: &str) -> Result<()> {
    let mut file = File::open(file_path)?;
    let file_size = file.metadata()?.len();
    let mut buffer = Vec::with_capacity(usize::try_from(file_size)?);
    let bytes_read = file.read_to_end(&mut buffer)?;
    log::info!("read {} bytes", bytes_read);

    let header = parse_file_header(&buffer)?;
    log::debug!("{:#x?}", header);

    let head_section = parse_head_section(&buffer, header.head_offset as usize)?;
    log::debug!("{:#x?}", head_section);

    // Every DataRef `val` inside HEAD is relative to this base
    let head_base = header.head_offset as usize + 0x08;

    let stream_info_offset = head_base + head_section.stream_data.val as usize;
    let stream_info = parse_stream_info(&buffer, stream_info_offset)?;
    log::debug!("{:#x?}", stream_info);

    let channel_table_offset = head_base + head_section.channel_data.val as usize;
    let channel_table = parse_channel_table(&buffer, channel_table_offset);
    log::debug!("{:#x?}", channel_table);

    if channel_table.channel_count != stream_info.channel_count {
        bail!(
            "channel count mismatch: stream info says {}, channel table says {}",
            stream_info.channel_count,
            channel_table.channel_count
        );
    }

    let mut channels = Vec::with_capacity(channel_table.channel_count as usize);
    for outer_ref in &channel_table.channel_refs {
        let mut inner_offset = head_base + outer_ref.val as usize;
        let inner_ref = parse_data_ref(&buffer, &mut inner_offset);
        if inner_ref.val == 0 {
            bail!("channel has no ADPCM parameter data");
        }
        let adpcm_offset = head_base + inner_ref.val as usize;
        channels.push(parse_adpcm_parameters(&buffer, adpcm_offset));
    }
    log::debug!("{:#x?}", channels);

    let data_section = parse_data_section(&buffer, header.data_offset as usize)?;
    log::debug!("{:#x?}", data_section);

    // `actual_data_offset` is already an absolute file offset
    let data_start = stream_info.actual_data_offset as usize;

    log::info!(
        "sample_rate={} channels={} block_count={} block_size={} block_samples={} data_start=0x{:x}",
        stream_info.sample_rate,
        stream_info.channel_count,
        stream_info.block_count,
        stream_info.block_size,
        stream_info.block_samples,
        data_start
    );

    let decoded_samples = decode_data_section(
        &buffer,
        stream_info.actual_data_offset as usize,
        &stream_info,
        &channels,
    )?;

    let spec = hound::WavSpec {
        channels: stream_info.channel_count as u16,
        sample_rate: stream_info.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create("test.wav", spec)?;
    let num_samples = decoded_samples.samples[0].len();
    for i in 0..num_samples {
        for channel in &decoded_samples.samples {
            writer.write_sample(channel[i])?;
        }
    }
    Ok(())
}
