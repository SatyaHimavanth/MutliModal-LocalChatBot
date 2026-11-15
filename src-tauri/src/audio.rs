use std::io::Cursor;

use symphonia::core::{
    audio::SampleBuffer, codecs::DecoderOptions, errors::Error as AudioError,
    formats::FormatOptions, io::MediaSourceStream, meta::MetadataOptions, probe::Hint,
};

const SAMPLE_RATE: u32 = 16_000;
const SEGMENT_SECONDS: usize = 8;
const OVERLAP_SECONDS: usize = 1;
const MAX_DURATION_SECONDS: usize = 20 * 60;

pub struct AudioSegment {
    pub start_seconds: usize,
    pub end_seconds: usize,
    pub wav: Vec<u8>,
}

/// Decode supported audio locally, downmix/resample to the endpoint QA format, and split it into
/// short overlapping WAV windows. This keeps each audio embedding request below llama.cpp's
/// default 512-token physical batch for long recordings without increasing the server batch.
pub fn split_for_embedding(input: &[u8], extension: &str) -> Result<Vec<AudioSegment>, String> {
    if !matches!(extension, "wav" | "mp3" | "flac") {
        return Err("Only WAV, MP3, and FLAC audio can be indexed.".into());
    }

    let source = MediaSourceStream::new(Box::new(Cursor::new(input.to_vec())), Default::default());
    let mut hint = Hint::new();
    hint.with_extension(extension);
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            source,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|error| format!("Could not read {extension} audio: {error}"))?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| "The audio file does not contain an audio track.".to_string())?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|error| format!("Could not decode {extension} audio: {error}"))?;

    let mut mono_samples = Vec::<i16>::new();
    let mut source_rate = None;
    let mut channels = None;
    let mut previous_mono = 0.0_f32;
    let mut next_source_position = 0.0_f64;
    let mut source_frame_index = 0_usize;
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(AudioError::IoError(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(error) => return Err(format!("Could not read audio packet: {error}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(AudioError::DecodeError(_)) => continue,
            Err(error) => return Err(format!("Could not decode audio packet: {error}")),
        };
        let spec = *decoded.spec();
        let channel_count = spec.channels.count();
        if channel_count == 0 || spec.rate == 0 {
            return Err("The audio stream has an invalid sample rate or channel layout.".into());
        }
        if source_rate.is_some_and(|rate| rate != spec.rate)
            || channels.is_some_and(|count| count != channel_count)
        {
            return Err("The audio changes sample rate or channel layout mid-stream.".into());
        }
        source_rate = Some(spec.rate);
        channels = Some(channel_count);

        let mut samples = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        samples.copy_interleaved_ref(decoded);
        for frame in samples.samples().chunks_exact(channel_count) {
            let mono = frame.iter().copied().sum::<f32>() / channel_count as f32;
            push_resampled_sample(
                mono,
                spec.rate,
                &mut source_frame_index,
                &mut previous_mono,
                &mut next_source_position,
                &mut mono_samples,
            )?;
        }
    }

    let decoded_rate =
        source_rate.ok_or_else(|| "No decodable audio samples were found.".to_string())?;
    if decoded_rate == 0 || mono_samples.is_empty() {
        return Err("No decodable audio samples were found.".into());
    }
    if mono_samples.len() > SAMPLE_RATE as usize * MAX_DURATION_SECONDS {
        return Err(format!(
            "Audio is longer than the {}-minute collection indexing limit.",
            MAX_DURATION_SECONDS / 60
        ));
    }

    let segment_frames = SAMPLE_RATE as usize * SEGMENT_SECONDS;
    let segment_stride = SAMPLE_RATE as usize * (SEGMENT_SECONDS - OVERLAP_SECONDS);
    let mut segments = Vec::new();
    let mut start = 0;
    while start < mono_samples.len() {
        let end = (start + segment_frames).min(mono_samples.len());
        let wav = encode_wav(&mono_samples[start..end]);
        segments.push(AudioSegment {
            start_seconds: start / SAMPLE_RATE as usize,
            end_seconds: end.div_ceil(SAMPLE_RATE as usize),
            wav,
        });
        if end == mono_samples.len() {
            break;
        }
        start += segment_stride;
    }
    Ok(segments)
}

fn push_resampled_sample(
    current: f32,
    source_rate: u32,
    source_frame_index: &mut usize,
    previous: &mut f32,
    next_source_position: &mut f64,
    output: &mut Vec<i16>,
) -> Result<(), String> {
    let step = source_rate as f64 / SAMPLE_RATE as f64;
    if *source_frame_index == 0 {
        output.push(sample_to_i16(current));
        *next_source_position = step;
        *previous = current;
        *source_frame_index = 1;
        return Ok(());
    }

    // Linear interpolation is adequate for the speech-oriented embedding signal and avoids
    // buffering the full decoded source at its original (often 44.1/48 kHz) rate.
    while *next_source_position <= *source_frame_index as f64 {
        let fraction = (*next_source_position - source_frame_index.saturating_sub(1) as f64)
            .clamp(0.0, 1.0) as f32;
        let interpolated = *previous + (current - *previous) * fraction;
        output.push(sample_to_i16(interpolated));
        if output.len() > SAMPLE_RATE as usize * MAX_DURATION_SECONDS {
            return Err(format!(
                "Audio is longer than the {}-minute collection indexing limit.",
                MAX_DURATION_SECONDS / 60
            ));
        }
        *next_source_position += step;
    }
    *previous = current;
    *source_frame_index += 1;
    Ok(())
}

fn sample_to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16
}

fn encode_wav(samples: &[i16]) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut wav = Vec::with_capacity(44 + data_len as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36_u32 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16_u32.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    wav.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes());
    wav.extend_from_slice(&2_u16.to_le_bytes());
    wav.extend_from_slice(&16_u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        wav.extend_from_slice(&sample.to_le_bytes());
    }
    wav
}

#[cfg(test)]
mod tests {
    use super::{encode_wav, split_for_embedding, SAMPLE_RATE};

    #[test]
    fn long_audio_is_split_into_overlapping_short_wav_segments() {
        let samples = (0..60 * SAMPLE_RATE)
            .map(|index| {
                let phase = index as f32 / SAMPLE_RATE as f32 * 440.0 * std::f32::consts::TAU;
                (phase.sin() * i16::MAX as f32) as i16
            })
            .collect::<Vec<_>>();
        let source = encode_wav(&samples);

        let segments = split_for_embedding(&source, "wav").expect("valid test audio");

        assert_eq!(segments.len(), 9);
        assert_eq!(segments.first().unwrap().start_seconds, 0);
        assert_eq!(segments[1].start_seconds, 7);
        assert_eq!(segments.last().unwrap().end_seconds, 60);
        assert!(segments
            .iter()
            .all(|segment| segment.wav.starts_with(b"RIFF")));
        assert!(segments
            .iter()
            .all(|segment| { segment.end_seconds.saturating_sub(segment.start_seconds) <= 8 }));
    }
}
