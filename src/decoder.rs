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

// "Stream Data Info", located via `HeadSection::stream_data`.
#[derive(Debug)]
struct StreamInfo {
    format: u8, // -- 0 = PCM8, 1 = PCM16, 2 = ADPCM (only ADPCM is supported for now)
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

// Located via `HeadSection::channel_data`. Each entry in `channel_refs` is a
// `DataRef` whose `val` points to *another* `DataRef` (double indirection),
// which finally points at that channel's `AdpcmParameters`. A `val` of 0 in
// that inner `DataRef` means the channel has no ADPCM data.
#[derive(Debug)]
struct ChannelTable {
    channel_count: u8,
    padding: [u8; 3],
    channel_refs: Vec<DataRef>,
}

// Per-channel ADPCM coefficients, initial decoder history, and loop state.
// Field offsets here were corrected against a real file: the wiki lists
// Gain at 0x2C, but it's actually packed tightly right after the
// coefficient table at 0x20 (struct is 0x2E bytes total, not 0x3A).
// Coefficients are 8 (coef1, coef2) pairs, selected per-frame by the top
// nibble of that frame's header byte.
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

// Located via `BrstmHeader::data_offset`. `offset_to_data` is relative to
// this field's own position (0x08 into the section) -- cross-checked
// against, but not used in place of, `StreamInfo::actual_data_offset`, since
// both should agree.
#[derive(Debug)]
struct DataSection {
    magic_string: [u8; 4], // -- "DATA"
    size: u32,
    offset_to_data: u32,
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

    // Every DataRef `val` inside HEAD is relative to this base (verified
    // against test.brstm -- see the project notes).
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

    // `actual_data_offset` is already an absolute file offset (confirmed
    // against test.brstm: its raw value was 0xAC0, matching the DATA
    // section's own offset_to_data field resolved separately via
    // data_section_offset + 0x08 + offset_to_data). Do not add
    // header.data_offset to it -- that double-counts.
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

    Ok(())
}
