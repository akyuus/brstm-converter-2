pub fn read_magic(buf: &[u8], offset: &mut usize) -> [u8; 4] {
    let bytes = [
        buf[*offset],
        buf[*offset + 1],
        buf[*offset + 2],
        buf[*offset + 3],
    ];
    *offset += 4;
    bytes
}

pub fn read_u8(buf: &[u8], offset: &mut usize) -> u8 {
    let value = buf[*offset];
    *offset += 1;
    value
}

pub fn read_i16(buf: &[u8], offset: &mut usize) -> i16 {
    let value = i16::from_be_bytes([buf[*offset], buf[*offset + 1]]);
    *offset += 2;
    value
}

pub fn read_u16(buf: &[u8], offset: &mut usize) -> u16 {
    let value = u16::from_be_bytes([buf[*offset], buf[*offset + 1]]);
    *offset += 2;
    value
}

pub fn read_u24(buf: &[u8], offset: &mut usize) -> u32 {
    let value = u32::from_be_bytes([0x00, buf[*offset], buf[*offset + 1], buf[*offset + 2]]);
    *offset += 3;
    value
}

pub fn read_u32(buf: &[u8], offset: &mut usize) -> u32 {
    let value = u32::from_be_bytes([
        buf[*offset],
        buf[*offset + 1],
        buf[*offset + 2],
        buf[*offset + 3],
    ]);
    *offset += 4;
    value
}
