use anyhow::{Context, Result};
use env_logger::Builder;
use log::LevelFilter;
use std::env;

use crate::decoder::decode_brstm;

mod decoder;
mod util;

fn main() -> Result<()> {
    let mut builder = Builder::from_default_env();
    builder.filter(None, LevelFilter::Debug).init();
    let brstm_path = env::args().nth(1).context("no file path given")?;
    decode_brstm(&brstm_path)?;
    Ok(())
}
