/*
 * This code is pretty much entirely ported from here: https://github.com/jackoalan/gc-dspadpcm-encode/blob/master/grok.c
 *
 * If anything looks particularly odd, it's because this is basically converted from C.
 */

use crate::decoder::Track;
use crate::util::{pad_to, patch_u32, write_i16, write_u8, write_u16, write_u24, write_u32};
use anyhow::{Result, ensure};

type Tvec = [f64; 3];

/// Used to match reference C impl.
const ROUNDING: f64 = 0.4999999f32 as f64;

fn inner_product_merge(window: &[i16; 16]) -> Tvec {
    let mut vec_out = [0.0; 3];
    for i in 0..=2 {
        let mut sum = 0.0;
        for x in 0..14 {
            sum -= window[2 + x - i] as f64 * window[2 + x] as f64;
        }
        vec_out[i] = sum;
    }
    vec_out
}

fn outer_product_merge(window: &[i16; 16]) -> [Tvec; 3] {
    let mut mtx = [[0.0; 3]; 3];
    for x in 1..=2 {
        for y in 1..=2 {
            let mut sum = 0.0;
            for z in 0..14 {
                sum += window[2 + z - x] as f64 * window[2 + z - y] as f64;
            }
            mtx[x][y] = sum;
        }
    }
    mtx
}

fn analyze_ranges(mtx: &mut [Tvec; 3], vec_idxs_out: &mut [usize; 3]) -> bool {
    let mut recips = [0.0; 3];

    for x in 1..=2 {
        let val = mtx[x][1].abs().max(mtx[x][2].abs());
        if val < f64::EPSILON {
            return true;
        }
        recips[x] = 1.0 / val;
    }

    let mut max_index = 0;
    for i in 1..=2 {
        for x in 1..i {
            let mut tmp = mtx[x][i];
            for y in 1..x {
                tmp -= mtx[x][y] * mtx[y][i];
            }
            mtx[x][i] = tmp;
        }

        let mut val = 0.0;
        for x in i..=2 {
            let mut tmp = mtx[x][i];
            for y in 1..i {
                tmp -= mtx[x][y] * mtx[y][i];
            }
            mtx[x][i] = tmp;

            let scaled = tmp.abs() * recips[x];
            if scaled >= val {
                val = scaled;
                max_index = x;
            }
        }

        if max_index != i {
            for y in 1..=2 {
                let tmp = mtx[max_index][y];
                mtx[max_index][y] = mtx[i][y];
                mtx[i][y] = tmp;
            }
            recips[max_index] = recips[i];
        }

        vec_idxs_out[i] = max_index;

        if mtx[i][i] == 0.0 {
            return true;
        }

        if i != 2 {
            let tmp = 1.0 / mtx[i][i];
            for x in (i + 1)..=2 {
                mtx[x][i] *= tmp;
            }
        }
    }

    let mut min = 1.0e10;
    let mut max = 0.0;
    for i in 1..=2 {
        let tmp = mtx[i][i].abs();
        if tmp < min {
            min = tmp;
        }
        if tmp > max {
            max = tmp;
        }
    }

    min / max < 1.0e-10
}

fn bidirectional_filter(mtx: &[Tvec; 3], vec_idxs: &[usize; 3], vec_out: &mut Tvec) {
    let mut x = 0;
    for i in 1..=2 {
        let index = vec_idxs[i];
        let mut tmp = vec_out[index];
        vec_out[index] = vec_out[i];
        if x != 0 {
            for y in x..=(i - 1) {
                tmp -= vec_out[y] * mtx[i][y];
            }
        } else if tmp != 0.0 {
            x = i;
        }
        vec_out[i] = tmp;
    }

    for i in (1..=2).rev() {
        let mut tmp = vec_out[i];
        for y in (i + 1)..=2 {
            tmp -= vec_out[y] * mtx[i][y];
        }
        vec_out[i] = tmp / mtx[i][i];
    }

    vec_out[0] = 1.0;
}

fn quadratic_merge(in_out_vec: &mut Tvec) -> bool {
    let v2 = in_out_vec[2];
    let tmp = 1.0 - (v2 * v2);

    if tmp == 0.0 {
        return true;
    }

    let v0 = (in_out_vec[0] - (v2 * v2)) / tmp;
    let v1 = (in_out_vec[1] - (in_out_vec[1] * v2)) / tmp;

    in_out_vec[0] = v0;
    in_out_vec[1] = v1;

    v1.abs() > 1.0
}

fn finish_record(input: &mut Tvec, out: &mut Tvec) {
    for z in 1..=2 {
        if input[z] >= 1.0 {
            input[z] = 0.9999999999;
        } else if input[z] <= -1.0 {
            input[z] = -0.9999999999;
        }
    }
    out[0] = 1.0;
    out[1] = (input[2] * input[1]) + input[1];
    out[2] = input[2];
}

fn matrix_filter(src: &Tvec, dst: &mut Tvec) {
    let mut mtx = [[0.0; 3]; 3];

    mtx[2][0] = 1.0;
    for i in 1..=2 {
        mtx[2][i] = -src[i];
    }

    for i in (1..=2).rev() {
        let val = 1.0 - (mtx[i][i] * mtx[i][i]);
        for y in 1..=i {
            mtx[i - 1][y] = ((mtx[i][i] * mtx[i][y]) + mtx[i][y]) / val;
        }
    }

    dst[0] = 1.0;
    for i in 1..=2 {
        let mut sum = 0.0;
        for y in 1..=i {
            sum += mtx[i][y] * dst[i - y];
        }
        dst[i] = sum;
    }
}

fn merge_finish_record(src: &Tvec, dst: &mut Tvec) {
    let mut tmp = [0.0; 3];
    let mut val = src[0];

    dst[0] = 1.0;
    for i in 1..=2 {
        let mut v2 = 0.0;
        for y in 1..i {
            v2 += dst[y] * src[i - y];
        }

        dst[i] = if val > 0.0 { -(v2 + src[i]) / val } else { 0.0 };
        tmp[i] = dst[i];

        for y in 1..i {
            dst[y] += dst[i] * dst[i - y];
        }

        val *= 1.0 - (dst[i] * dst[i]);
    }

    finish_record(&mut tmp, dst);
}

fn contrast_vectors(source1: &Tvec, source2: &Tvec) -> f64 {
    let val = (source2[2] * source2[1] - source2[1]) / (1.0 - source2[2] * source2[2]);
    let val1 = source1[0] * source1[0] + source1[1] * source1[1] + source1[2] * source1[2];
    let val2 = source1[0] * source1[1] + source1[1] * source1[2];
    let val3 = source1[0] * source1[2];
    val1 + (2.0 * val * val2) + (2.0 * (-source2[1] * val - source2[2]) * val3)
}

fn filter_records(vec_best: &mut [Tvec; 8], exp: usize, records: &[Tvec]) {
    let mut buffer_list = [[0.0; 3]; 8];
    let mut counts = [0; 8];

    for _ in 0..2 {
        for y in 0..exp {
            counts[y] = 0;
            buffer_list[y] = [0.0; 3];
        }

        for record in records {
            let mut index = 0;
            let mut value = 1.0e30;
            for (i, best) in vec_best.iter().enumerate().take(exp) {
                let temp_val = contrast_vectors(best, record);
                if temp_val < value {
                    value = temp_val;
                    index = i;
                }
            }
            counts[index] += 1;

            let mut filtered = [0.0; 3];
            matrix_filter(record, &mut filtered);
            for i in 0..=2 {
                buffer_list[index][i] += filtered[i];
            }
        }

        for i in 0..exp {
            if counts[i] > 0 {
                for y in 0..=2 {
                    buffer_list[i][y] /= counts[i] as f64;
                }
            }
        }

        for i in 0..exp {
            let src = buffer_list[i];
            merge_finish_record(&src, &mut vec_best[i]);
        }
    }
}

fn quantize_coefficient(scaled: f64) -> i16 {
    let rounded = scaled.round();
    if rounded > 32767.0 {
        32767
    } else if rounded < -32768.0 {
        -32768
    } else {
        rounded as i16
    }
}

pub fn find_coefficients(source: &[i16]) -> [[i16; 2]; 8] {
    let num_frames = source.len().div_ceil(14).max(1);
    let mut padded = vec![0i16; num_frames * 14];
    padded[..source.len()].copy_from_slice(source);

    let mut window = [0i16; 16];
    let mut records = Vec::new();

    for frame in padded.as_chunks::<14>().0 {
        window[0] = window[14];
        window[1] = window[15];
        window[2..16].copy_from_slice(frame);

        let vec1 = inner_product_merge(&window);
        if vec1[0].abs() <= 10.0 {
            continue;
        }

        let mut mtx = outer_product_merge(&window);
        let mut vec_idxs = [0; 3];
        if analyze_ranges(&mut mtx, &mut vec_idxs) {
            continue;
        }

        let mut vec1 = vec1;
        bidirectional_filter(&mtx, &vec_idxs, &mut vec1);
        if quadratic_merge(&mut vec1) {
            continue;
        }

        let mut record = [0.0; 3];
        finish_record(&mut vec1, &mut record);
        records.push(record);
    }

    if records.is_empty() {
        return [[0; 2]; 8];
    }

    let mut vec1 = [1.0, 0.0, 0.0];
    for record in &records {
        let mut filtered = [0.0; 3];
        matrix_filter(record, &mut filtered);
        vec1[1] += filtered[1];
        vec1[2] += filtered[2];
    }
    let record_count = records.len() as f64;
    vec1[1] /= record_count;
    vec1[2] /= record_count;

    let mut vec_best = [[0.0; 3]; 8];
    merge_finish_record(&vec1, &mut vec_best[0]);

    let mut exp = 1;
    for w in 0..3 {
        let vec2 = [0.0, -1.0, 0.0];
        for i in 0..exp {
            for y in 0..=2 {
                vec_best[exp + i][y] = 0.01 * vec2[y] + vec_best[i][y];
            }
        }
        exp = 1 << (w + 1);
        filter_records(&mut vec_best, exp, &records);
    }

    let mut coefficients = [[0; 2]; 8];
    for z in 0..8 {
        coefficients[z][0] = quantize_coefficient(-vec_best[z][1] * 2048.0);
        coefficients[z][1] = quantize_coefficient(-vec_best[z][2] * 2048.0);
    }
    coefficients
}

pub fn encode_frame(
    pcm_in_out: &mut [i16; 16],
    sample_count: usize,
    coefficients: &[[i16; 2]; 8],
) -> [u8; 8] {
    let mut in_samples = [[0i32; 16]; 8];
    let mut out_samples = [[0i32; 14]; 8];
    let mut scale = [0i32; 8];
    let mut dist_accum = [0.0f64; 8];

    for i in 0..8usize {
        let coef0 = coefficients[i][0] as i32;
        let coef1 = coefficients[i][1] as i32;

        in_samples[i][0] = pcm_in_out[0] as i32;
        in_samples[i][1] = pcm_in_out[1] as i32;

        let mut distance = 0i32;
        for s in 0..sample_count {
            let v1 = ((pcm_in_out[s] as i32 * coef1) + (pcm_in_out[s + 1] as i32 * coef0)) / 2048;
            in_samples[i][s + 2] = v1;
            let v2 = pcm_in_out[s + 2] as i32 - v1;
            let v3 = v2.clamp(-32768, 32767);
            if v3.abs() > distance.abs() {
                distance = v3;
            }
        }

        let mut sc = 0i32;
        while sc <= 12 && !(-8..=7).contains(&distance) {
            sc += 1;
            distance /= 2;
        }
        scale[i] = if sc <= 1 { -1 } else { sc - 2 };

        let mut index;
        loop {
            scale[i] += 1;
            dist_accum[i] = 0.0;
            index = 0i32;
            let step = 1i32 << scale[i];

            for s in 0..sample_count {
                let v1 = (in_samples[i][s] * coef1) + (in_samples[i][s + 1] * coef0);
                let v2 = ((pcm_in_out[s + 2] as i32) << 11) - v1;
                let v2 = v2 / 2048;
                let mut v3 = if v2 > 0 {
                    (v2 as f64 / step as f64 + ROUNDING) as i32
                } else {
                    (v2 as f64 / step as f64 - ROUNDING) as i32
                };

                if v3 < -8 {
                    let overshoot = -8 - v3;
                    if index < overshoot {
                        index = overshoot;
                    }
                    v3 = -8;
                } else if v3 > 7 {
                    let overshoot = v3 - 7;
                    if index < overshoot {
                        index = overshoot;
                    }
                    v3 = 7;
                }

                out_samples[i][s] = v3;

                let reconstructed = (v1 + ((v3 * step) << 11) + 1024) >> 11;
                let clamped = reconstructed.clamp(-32768, 32767);
                in_samples[i][s + 2] = clamped;

                let err = pcm_in_out[s + 2] as i32 - clamped;
                dist_accum[i] += err as f64 * err as f64;
            }

            let mut x = index + 8;
            while x > 256 {
                scale[i] += 1;
                if scale[i] >= 12 {
                    scale[i] = 11;
                }
                x >>= 1;
            }

            if !(scale[i] < 12 && index > 1) {
                break;
            }
        }
    }

    let mut best_index = 0usize;
    let mut min = f64::MAX;
    for (i, &d) in dist_accum.iter().enumerate() {
        if d < min {
            min = d;
            best_index = i;
        }
    }

    for s in 0..sample_count {
        pcm_in_out[s + 2] = in_samples[best_index][s + 2] as i16;
    }

    for s in sample_count..14 {
        out_samples[best_index][s] = 0;
    }

    let mut adpcm_out = [0u8; 8];
    adpcm_out[0] = ((best_index as u8) << 4) | (scale[best_index] as u8 & 0x0f);
    for y in 0..7usize {
        let high = (out_samples[best_index][y * 2] & 0x0f) as u8;
        let low = (out_samples[best_index][y * 2 + 1] & 0x0f) as u8;
        adpcm_out[y + 1] = (high << 4) | low;
    }

    adpcm_out
}

pub struct EncodedChannel {
    pub coefficients: [[i16; 2]; 8],
    pub data: Vec<u8>,           // -- every frame of the channel 
    pub reconstructed: Vec<i16>, // -- samples as a decoder will see them
}

pub fn encode_channel(samples: &[i16]) -> EncodedChannel {
    let coefficients = find_coefficients(samples);
    let mut data = Vec::with_capacity(samples.len().div_ceil(14) * 8);
    let mut reconstructed = Vec::with_capacity(samples.len().div_ceil(14) * 14);

    // -- [hist2, hist1, <14 samples>] 
    let mut window = [0i16; 16];
    for frame in samples.chunks(14) {
        window[2..].fill(0);
        window[2..2 + frame.len()].copy_from_slice(frame);
        data.extend_from_slice(&encode_frame(&mut window, 14, &coefficients));
        reconstructed.extend_from_slice(&window[2..]);
        window[0] = window[14];
        window[1] = window[15];
    }
    reconstructed.truncate(samples.len());

    EncodedChannel {
        coefficients,
        data,
        reconstructed,
    }
}

impl EncodedChannel {
    fn history_at(&self, sample: usize) -> (i16, i16) {
        let yn1 = if sample >= 1 { self.reconstructed[sample - 1] } else { 0 };
        let yn2 = if sample >= 2 { self.reconstructed[sample - 2] } else { 0 };
        (yn1, yn2)
    }

    // first byte in this particular sample's frame
    fn ps_at(&self, sample: usize) -> u8 {
        self.data[sample / 14 * 8]
    }
}

pub struct EncodeParams {
    pub channels: Vec<Vec<i16>>,
    pub sample_rate: u32,
    pub loop_start: Option<u32>, // -- loops always run to the end of the stream
    pub tracks: Vec<Track>,
    pub extended_tracks: bool, // -- write volume/pan in the track table
}

const BLOCK_SIZE: usize = 0x2000;
const BLOCK_SAMPLES: usize = BLOCK_SIZE / 8 * 14;

/// writes a DataRef with a zero value then returns the offset into it's 'val' field to be patched later
fn write_data_ref(buf: &mut Vec<u8>, data_type: u8) -> usize {
    write_u8(buf, 1); // -- offset (relative to HEAD + 8)
    write_u8(buf, data_type);
    write_u16(buf, 0);
    write_u32(buf, 0);
    buf.len() - 4
}

pub fn encode_brstm(params: &EncodeParams) -> Result<Vec<u8>> {
    let channel_count = params.channels.len();
    let sample_count = params.channels.first().map_or(0, Vec::len);

    ensure!(channel_count > 0, "no channels to encode");
    ensure!(
        channel_count <= 255,
        "BRSTM supports at most 255 channels (got {})",
        channel_count
    );
    ensure!(
        params.channels.iter().all(|c| c.len() == sample_count),
        "all channels must have the same length"
    );
    ensure!(sample_count > 0, "no samples to encode");
    ensure!(
        sample_count <= u32::MAX as usize,
        "too many samples ({})",
        sample_count
    );
    ensure!(
        params.sample_rate > 0 && params.sample_rate <= 0xFF_FFFF,
        "sample rate {} doesn't fit in 24 bits",
        params.sample_rate
    );
    if let Some(start) = params.loop_start {
        ensure!(
            (start as usize) < sample_count,
            "loop start {} is past the end of the stream ({} samples)",
            start,
            sample_count
        );
    }
    ensure!(
        !params.tracks.is_empty() && params.tracks.len() <= 255,
        "need 1-255 tracks (got {})",
        params.tracks.len()
    );
    for track in &params.tracks {
        ensure!(!track.channels.is_empty(), "track has no channels");
        for &c in &track.channels {
            ensure!(
                (c as usize) < channel_count,
                "track references channel {}, but there are only {}",
                c,
                channel_count
            );
        }
    }

    let encoded: Vec<EncodedChannel> = std::thread::scope(|s| {
        let handles: Vec<_> = params
            .channels
            .iter()
            .map(|samples| s.spawn(|| encode_channel(samples)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let block_count = sample_count.div_ceil(BLOCK_SAMPLES);
    let final_block_samples = sample_count - (block_count - 1) * BLOCK_SAMPLES;
    let final_block_size = final_block_samples.div_ceil(14) * 8;
    let final_block_size_padded = final_block_size.next_multiple_of(0x20);

    // -- HEAD body; every DataRef value is relative to its start (HEAD + 8)
    let mut h = Vec::new();
    let stream_info_ref = write_data_ref(&mut h, 0);
    let track_table_ref = write_data_ref(&mut h, 0);
    let channel_table_ref = write_data_ref(&mut h, 0);

    let len = h.len() as u32;
    patch_u32(&mut h, stream_info_ref, len);
    write_u8(&mut h, 2); // -- ADPCM
    write_u8(&mut h, params.loop_start.is_some() as u8);
    write_u8(&mut h, channel_count as u8);
    write_u24(&mut h, params.sample_rate);
    write_u16(&mut h, 0);
    write_u32(&mut h, params.loop_start.unwrap_or(0));
    write_u32(&mut h, sample_count as u32);
    let data_offset_field = h.len();
    write_u32(&mut h, 0); // -- absolute data offset, patched once the layout is known
    write_u32(&mut h, block_count as u32);
    write_u32(&mut h, BLOCK_SIZE as u32);
    write_u32(&mut h, BLOCK_SAMPLES as u32);
    write_u32(&mut h, final_block_size as u32);
    write_u32(&mut h, final_block_samples as u32);
    write_u32(&mut h, final_block_size_padded as u32);
    write_u32(&mut h, BLOCK_SAMPLES as u32); // -- ADPC interval
    write_u32(&mut h, 4); // -- always 4. bytes per channel per ADPC entry, i.e. 2 i16s

    let len = h.len() as u32;
    patch_u32(&mut h, track_table_ref, len);
    let track_info_type = params.extended_tracks as u8;
    write_u8(&mut h, params.tracks.len() as u8);
    write_u8(&mut h, track_info_type);
    write_u16(&mut h, 0);
    let track_refs: Vec<usize> = params
        .tracks
        .iter()
        .map(|_| write_data_ref(&mut h, track_info_type))
        .collect();
    for (track, track_ref) in params.tracks.iter().zip(track_refs) {
        let len = h.len() as u32;
        patch_u32(&mut h, track_ref, len);
        if params.extended_tracks {
            write_u8(&mut h, track.volume);
            write_u8(&mut h, track.pan);
            write_u16(&mut h, 0);
            write_u32(&mut h, 0);
        }
        write_u8(&mut h, track.channels.len() as u8);
        h.extend_from_slice(&track.channels);
        pad_to(&mut h, 4);
    }

    let len = h.len() as u32;
    patch_u32(&mut h, channel_table_ref, len);
    write_u8(&mut h, channel_count as u8);
    h.extend_from_slice(&[0; 3]);
    let channel_refs: Vec<usize> = encoded.iter().map(|_| write_data_ref(&mut h, 0)).collect();
    let loop_start = params.loop_start.unwrap_or(0) as usize;
    for (ch, channel_ref) in encoded.iter().zip(channel_refs) {
        let len = h.len() as u32;
        patch_u32(&mut h, channel_ref, len);
        let params_ref = write_data_ref(&mut h, 0);
        let len = h.len() as u32;
        patch_u32(&mut h, params_ref, len);
        for [coef1, coef2] in ch.coefficients {
            write_i16(&mut h, coef1);
            write_i16(&mut h, coef2);
        }
        let (loop_yn1, loop_yn2) = ch.history_at(loop_start);
        write_u16(&mut h, 0); // -- gain
        write_u16(&mut h, ch.ps_at(0) as u16);
        write_i16(&mut h, 0); // -- yn1, silence
        write_i16(&mut h, 0); // -- yn2
        write_u16(&mut h, ch.ps_at(loop_start) as u16);
        write_i16(&mut h, loop_yn1);
        write_i16(&mut h, loop_yn2);
        pad_to(&mut h, 4);
    }

    // -- file header, filled in at the end
    let mut f = vec![0u8; 0x40];

    let head_offset = f.len();
    f.extend_from_slice(b"HEAD");
    write_u32(&mut f, 0);
    let head_base = f.len();
    f.extend_from_slice(&h);
    pad_to(&mut f, 0x20);
    let head_size = f.len() - head_offset;
    patch_u32(&mut f, head_offset + 4, head_size as u32);

    // -- decoder history at the start of each block
    let adpc_offset = f.len();
    f.extend_from_slice(b"ADPC");
    write_u32(&mut f, 0);
    for b in 0..block_count {
        for ch in &encoded {
            let (yn1, yn2) = ch.history_at(b * BLOCK_SAMPLES);
            write_i16(&mut f, yn1);
            write_i16(&mut f, yn2);
        }
    }
    pad_to(&mut f, 0x20);
    let adpc_size = f.len() - adpc_offset;
    patch_u32(&mut f, adpc_offset + 4, adpc_size as u32);

    let data_offset = f.len();
    f.extend_from_slice(b"DATA");
    write_u32(&mut f, 0);
    write_u32(&mut f, 0x18); // -- audio starts 0x18 bytes after this field
    pad_to(&mut f, 0x20);
    let audio_start = f.len();
    patch_u32(&mut f, head_base + data_offset_field, audio_start as u32);
    for b in 0..block_count {
        let start = b * BLOCK_SIZE;
        for ch in &encoded {
            if b == block_count - 1 {
                f.extend_from_slice(&ch.data[start..start + final_block_size]);
                pad_to(&mut f, 0x20);
            } else {
                f.extend_from_slice(&ch.data[start..start + BLOCK_SIZE]);
            }
        }
    }
    let data_size = f.len() - data_offset;
    patch_u32(&mut f, data_offset + 4, data_size as u32);

    let file_size = f.len();
    let mut header = Vec::with_capacity(0x40);
    header.extend_from_slice(b"RSTM");
    write_u16(&mut header, 0xFEFF);
    write_u16(&mut header, 0x0100);
    write_u32(&mut header, file_size as u32);
    write_u16(&mut header, 0x40);
    write_u16(&mut header, 3); // -- HEAD, ADPC, DATA
    for (offset, size) in [
        (head_offset, head_size),
        (adpc_offset, adpc_size),
        (data_offset, data_size),
    ] {
        write_u32(&mut header, offset as u32);
        write_u32(&mut header, size as u32);
    }
    f[..header.len()].copy_from_slice(&header);

    Ok(f)
}
