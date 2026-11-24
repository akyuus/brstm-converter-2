use anyhow::{Result, bail};

pub fn read_bytes<'a>(buf: &'a [u8], offset: &mut usize, len: usize) -> Result<&'a [u8]> {
    match offset.checked_add(len) {
        Some(end) if end <= buf.len() => {
            let bytes = &buf[*offset..end];
            *offset = end;
            Ok(bytes)
        }
        _ => bail!(
            "unexpected end of file: wanted {} bytes at offset 0x{:x}, file is 0x{:x} bytes",
            len,
            *offset,
            buf.len()
        ),
    }
}

fn read_array<const N: usize>(buf: &[u8], offset: &mut usize) -> Result<[u8; N]> {
    Ok(read_bytes(buf, offset, N)?.try_into().unwrap())
}

pub fn read_magic(buf: &[u8], offset: &mut usize) -> Result<[u8; 4]> {
    read_array(buf, offset)
}

pub fn read_u8(buf: &[u8], offset: &mut usize) -> Result<u8> {
    Ok(read_array::<1>(buf, offset)?[0])
}

pub fn read_i16(buf: &[u8], offset: &mut usize) -> Result<i16> {
    Ok(i16::from_be_bytes(read_array(buf, offset)?))
}

pub fn read_u16(buf: &[u8], offset: &mut usize) -> Result<u16> {
    Ok(u16::from_be_bytes(read_array(buf, offset)?))
}

pub fn read_u24(buf: &[u8], offset: &mut usize) -> Result<u32> {
    let [a, b, c] = read_array(buf, offset)?;
    Ok(u32::from_be_bytes([0x00, a, b, c]))
}

pub fn read_u32(buf: &[u8], offset: &mut usize) -> Result<u32> {
    Ok(u32::from_be_bytes(read_array(buf, offset)?))
}

pub fn write_u8(buf: &mut Vec<u8>, value: u8) {
    buf.push(value);
}

pub fn write_i16(buf: &mut Vec<u8>, value: i16) {
    buf.extend_from_slice(&value.to_be_bytes());
}

pub fn write_u16(buf: &mut Vec<u8>, value: u16) {
    buf.extend_from_slice(&value.to_be_bytes());
}

pub fn write_u24(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_be_bytes()[1..]);
}

pub fn write_u32(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_be_bytes());
}

pub fn pad_to(buf: &mut Vec<u8>, align: usize) {
    buf.resize(buf.len().next_multiple_of(align), 0);
}

pub fn patch_u32(buf: &mut [u8], at: usize, value: u32) {
    buf[at..at + 4].copy_from_slice(&value.to_be_bytes());
}
